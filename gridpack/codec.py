"""gridpack.codec -- 格点数据无损编解码核心。

流水线（每一级都严格可逆，quant_bits=0 时 decode(encode(x)) == x 逐字节相等）：

  1) 表示层 (representation)
       a) 精确格点检测：全体取值是否都落在同一个 2**E 上（见 lattice.py）。
          命中则改写成整数 k = v / 2**E —— 这是 ERA5 这类量化格点数据最主要的收益来源；
       b) 未命中则退化为位序重映射 (order map)：IEEE-754 位型 -> 保序整数。
  2) 背景场剔除 + 残差编码 : 时间维 + 纬向 + 经向 依次一阶差分（可分离的三维 Lorenzo）
  3) 折叠 (zig-zag)        : 有符号残差 -> 无符号小整数
  4) 位面重排 (byte-plane) : 按字节拆成平面（等价 shuffle）
  5) 熵编码                : zstd（默认）/ zlib / rc0 / rc1 / huff，见 entropy.py

时间模型可选（mode）：
  'none'    不做任何预测，只做位面重排（稀疏/零值多的场，如降水）
  'spatial' 只做纬向+经向差分
  'dt'      时间一阶差分 + 空间差分（适合逐小时连续场）
  'lagN'    时间 N 步差分 + 空间差分（N 由数据节律决定，如逐小时对齐日循环用 N=24）
  'auto'    逐块试上面几种，取最小的

quant_bits>0 为有容差模式（丢保序整数低 quant_bits 位），默认 0 = 严格无损。

负零（-0.0）在格点路径下无法用整数区分，故单独记录稀疏位图随块存储，保证逐位无损。
"""

from __future__ import annotations

import struct

import numpy as np

from . import entropy as _entropy
from . import lattice as _lattice

CODEC_VERSION = 4

__all__ = ["encode_array", "decode_array", "encode_bytes", "decode_bytes",
           "CODEC_VERSION", "MODES", "candidate_modes", "detect_lattice_e"]

MODES = ("auto", "none", "spatial", "dt")

_M_RAW = 0        # 非浮点兜底
_M_PRED = 1       # 保序整数 + 带时间滞后的预测编码
_M_NONE = 2       # 保序整数 + 仅位面重排
_M_LAT_NONE = 3   # 格点整数 + 仅位面重排
_M_LAT_PRED = 4   # 格点整数 + 预测编码
_BLK_TAG = b"GPZK"

_LAZY = object()


# --------------------------------------------------------------------------
# 基础可逆变换
# --------------------------------------------------------------------------
def _order_map(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype == np.float32:
        u = a.view(np.uint32)
        m = np.where((u >> np.uint32(31)).astype(bool),
                     np.uint32(0xFFFFFFFF) - u, u | np.uint32(0x80000000))
        return np.ascontiguousarray(m).view(np.int32)
    if a.dtype == np.float64:
        u = a.view(np.uint64)
        m = np.where((u >> np.uint64(63)).astype(bool),
                     np.uint64(0xFFFFFFFFFFFFFFFF) - u, u | np.uint64(0x8000000000000000))
        return np.ascontiguousarray(m).view(np.int64)
    raise TypeError("order map only supports float32/float64, got %s" % a.dtype)


def _order_unmap(m: np.ndarray, dtype) -> np.ndarray:
    m = np.ascontiguousarray(m)
    if dtype == np.float32:
        u = m.view(np.uint32)
        r = np.where((u >> np.uint32(31)).astype(bool),
                     u & np.uint32(0x7FFFFFFF), np.uint32(0xFFFFFFFF) - u)
        return np.ascontiguousarray(r).view(np.float32)
    if dtype == np.float64:
        u = m.view(np.uint64)
        r = np.where((u >> np.uint64(63)).astype(bool),
                     u & np.uint64(0x7FFFFFFFFFFFFFFF), np.uint64(0xFFFFFFFFFFFFFFFF) - u)
        return np.ascontiguousarray(r).view(np.float64)
    raise TypeError("order unmap only supports float32/float64, got %s" % dtype)


def _quantize_order(m: np.ndarray, quant_bits: int) -> np.ndarray:
    if quant_bits <= 0:
        return m
    u = m.view(np.dtype("u%d" % m.dtype.itemsize))
    half = np.array(1 << (quant_bits - 1), dtype=u.dtype)
    mask = np.array(((1 << (u.dtype.itemsize * 8)) - 1) ^ ((1 << quant_bits) - 1), dtype=u.dtype)
    return ((u + half) & mask).view(m.dtype)


def _zig(u: np.ndarray) -> np.ndarray:
    bits = u.dtype.itemsize * 8
    return (u << np.array(1, dtype=u.dtype)) ^ (
        np.array(0, dtype=u.dtype) - (u >> np.array(bits - 1, dtype=u.dtype)))


def _unzig(z: np.ndarray) -> np.ndarray:
    one = np.array(1, dtype=z.dtype)
    zero = np.array(0, dtype=z.dtype)
    return (z >> one) ^ (zero - (z & one))


def _dims3(shape):
    if len(shape) >= 3:
        return int(np.prod(shape[:-2])), int(shape[-2]), int(shape[-1])
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 1:
        return 1, 1, int(shape[0])
    return 1, 1, 1


def _diff_axis(a: np.ndarray, axis: int) -> np.ndarray:
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    src[axis] = slice(None, -1)
    dst[axis] = slice(1, None)
    a[tuple(dst)] -= a[tuple(src)].copy()
    return a


def _temporal_fwd(d: np.ndarray, lag: int) -> np.ndarray:
    """时间维一阶差分（滞后 lag 步）。lag<=0 表示不做时间差分。"""
    if lag <= 0 or d.shape[0] <= lag:
        return d
    d[lag:] -= d[:-lag].copy()
    return d


def _temporal_inv(d: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or d.shape[0] <= lag:
        return d
    t = d.shape[0]
    if lag == 1:
        return np.cumsum(d, axis=0, dtype=d.dtype)
    q, r = divmod(t, lag)
    if r:
        for i in range(lag, t):
            d[i] += d[i - lag]
        return d
    v = d.reshape(q, lag, *d.shape[1:])
    return np.cumsum(v, axis=0, dtype=d.dtype).reshape(d.shape)


def _predict_fwd(m: np.ndarray, lag: int) -> np.ndarray:
    d = m.copy()
    _temporal_fwd(d, lag)
    for ax in (2, 1):
        if d.ndim > ax and d.shape[ax] > 1:
            _diff_axis(d, ax)
    return d


def _predict_inv(d: np.ndarray, lag: int) -> np.ndarray:
    d = d.copy()
    for ax in (1, 2):
        if d.ndim > ax and d.shape[ax] > 1:
            d = np.cumsum(d, axis=ax, dtype=d.dtype)
    return _temporal_inv(d, lag)


def _byte_planes(u8: np.ndarray, width: int) -> np.ndarray:
    b = u8.reshape(-1, width)
    return np.concatenate([np.ascontiguousarray(b[:, i]) for i in range(width)])


def _unbyte_planes(buf: np.ndarray, width: int, n: int) -> np.ndarray:
    q = buf.reshape(width, n)
    out = np.empty((n, width), dtype=np.uint8)
    for i in range(width):
        out[:, i] = q[i]
    return np.ascontiguousarray(out).reshape(-1)


def candidate_modes(lags=()):
    """给定候选时间滞后（步数），返回去重后的 mode 列表。"""
    out = ["none", "spatial", "dt"]
    for L in lags:
        L = int(L)
        if L > 1:
            out.append("lag%d" % L)
    return out


def _parse_mode(mode):
    """返回 (use_prediction, lag)。"""
    if mode == "none":
        return False, 0
    if mode == "spatial":
        return True, 0
    if mode == "dt":
        return True, 1
    if mode.startswith("lag"):
        return True, int(mode[3:])
    raise ValueError("unknown mode %r" % mode)


def _lag_header(lag: int) -> bytes:
    if lag < 255:
        return bytes([lag])
    return bytes([255]) + struct.pack("<I", lag)


# --------------------------------------------------------------------------
# 单块编解码
# --------------------------------------------------------------------------
def _encode_block(a: np.ndarray, level: int, threads: int, quant_bits: int,
                  mode: str, ent: str, lat_e=_LAZY) -> bytes:
    if a.dtype not in (np.float32, np.float64):
        return bytes([_M_RAW, _entropy.eid("zstd")]) + _entropy.compress(
            a.tobytes(), "zstd", level, threads)

    t, y, x = _dims3(a.shape)
    src = a.reshape(t, y, x)

    if lat_e is _LAZY:
        lat_e = _lattice.detect_lattice(src) if quant_bits == 0 else None

    if mode == "auto":
        best = None
        for mm in ("none", "spatial", "dt"):
            c = _encode_block(a, level, threads, quant_bits, mm, ent, lat_e=lat_e)
            if best is None or len(c) < len(best):
                best = c
        return best

    use_pred, lag = _parse_mode(mode)
    eid = _entropy.eid(ent)

    if lat_e is not None:
        width = _lattice.INT_WIDTH
        ud = _lattice.UINT_DTYPE
        idt = _lattice.INT_DTYPE
        ints = _lattice.lattice_encode(src, lat_e)
        nz = _lattice.negative_zero_mask(src)
        head = bytes([_M_LAT_PRED if use_pred else _M_LAT_NONE, eid]) + \
            struct.pack("<hI", int(lat_e), len(nz))
        if use_pred:
            head += _lag_header(lag)
            d = _predict_fwd(ints, lag)
            del ints
            payload = _byte_planes(_zig(d.view(ud)).view(np.uint8), width)
            del d
        else:
            payload = _byte_planes(ints.view(ud).view(np.uint8), width)
            del ints
        return head + _entropy.compress(payload.tobytes(), ent, level, threads) + nz

    width = a.dtype.itemsize
    ud = np.dtype("u%d" % width)
    m = _quantize_order(_order_map(src), quant_bits)

    if not use_pred:
        payload = _byte_planes(m.view(ud).view(np.uint8), width)
        return bytes([_M_NONE, eid]) + _entropy.compress(payload.tobytes(), ent, level, threads)

    d = _predict_fwd(m, lag)
    del m
    payload = _byte_planes(_zig(d.view(ud)).view(np.uint8), width)
    del d
    return bytes([_M_PRED, eid]) + _lag_header(lag) + _entropy.compress(
        payload.tobytes(), ent, level, threads)


def _decode_block(buf: bytes, shape, dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    tag = buf[0]
    eid = buf[1]
    method = _entropy.name(eid)

    if tag == _M_RAW:
        raw = _entropy.decompress(buf[2:], method, int(np.prod(shape)) * dt.itemsize)
        return np.frombuffer(raw, dtype=dt).reshape(shape)

    t, y, x = _dims3(shape)
    n = t * y * x
    off = 2
    lat_e = None
    nzlen = 0
    if tag in (_M_LAT_NONE, _M_LAT_PRED):
        lat_e, nzlen = struct.unpack_from("<hI", buf, off)
        off += 6
    use_pred = tag in (_M_PRED, _M_LAT_PRED)
    lag = 0
    if use_pred:
        lag = buf[off]
        if lag == 255:
            lag = struct.unpack_from("<I", buf, off + 1)[0]
            off += 5
        else:
            off += 1
    end = len(buf) - nzlen
    body = buf[off:end]
    nz = buf[end:] if nzlen else b""

    if lat_e is not None:
        width = _lattice.INT_WIDTH
        ud = _lattice.UINT_DTYPE
        idt = _lattice.INT_DTYPE
    else:
        width = dt.itemsize
        ud = np.dtype("u%d" % width)
        idt = np.dtype("i%d" % width)

    payload = np.frombuffer(_entropy.decompress(body, method, n * width), dtype=np.uint8)
    u8 = _unbyte_planes(payload, width, n)

    if lat_e is not None:
        if use_pred:
            d = _unzig(u8.view(ud)).view(idt).reshape(t, y, x)
            ints = _predict_inv(d, lag)
        else:
            ints = u8.view(ud).view(idt).reshape(t, y, x)
        out = _lattice.lattice_decode(ints, lat_e, dt).reshape(shape)
        return _lattice.restore_negative_zero(out, nz, dt)

    if not use_pred:
        return _order_unmap(u8.view(ud).view(idt).reshape(shape), dt)
    d = _unzig(u8.view(ud)).view(idt)
    return _order_unmap(_predict_inv(d.reshape(t, y, x), lag), dt).reshape(shape)


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
def detect_lattice_e(a: np.ndarray):
    """返回数组所用格点指数 E；不适用时返回 None（供报告与基准使用）。"""
    return _lattice.detect_lattice(a)


def encode_array(a: np.ndarray, level: int = 12, chunk_t: int | None = None,
                 threads: int = 1, quant_bits: int = 0, mode: str = "auto",
                 entropy: str = _entropy.DEFAULT_METHOD) -> bytes:
    a = np.ascontiguousarray(a)
    if chunk_t is None or a.ndim == 0 or a.shape[0] <= chunk_t:
        return _encode_block(a, level, threads, quant_bits, mode, entropy)
    blocks = []
    for s in range(0, a.shape[0], chunk_t):
        blk = np.ascontiguousarray(a[s:s + chunk_t])
        c = _encode_block(blk, level, threads, quant_bits, mode, entropy)
        blocks.append(struct.pack("<IQ", blk.shape[0], len(c)) + c)
    return _BLK_TAG + struct.pack("<I", len(blocks)) + b"".join(blocks)


def decode_array(buf: bytes, shape, dtype, chunk_t: int | None = None) -> np.ndarray:
    dt = np.dtype(dtype)
    if buf[:4] == _BLK_TAG:
        off = 8
        nblk = struct.unpack_from("<I", buf, 4)[0]
        out = []
        for _ in range(nblk):
            tlen, clen = struct.unpack_from("<IQ", buf, off)
            off += 12
            sub = buf[off:off + clen]
            off += clen
            out.append(_decode_block(sub, (tlen,) + tuple(shape[1:]), dt))
        return np.concatenate(out, axis=0).reshape(shape)
    return _decode_block(buf, shape, dt)


def encode_bytes(raw: bytes, shape, dtype, level: int = 12, chunk_t: int | None = None,
                 threads: int = 1, quant_bits: int = 0, mode: str = "auto",
                 entropy: str = _entropy.DEFAULT_METHOD) -> bytes:
    dt = np.dtype(dtype)
    if dt in (np.float32, np.float64):
        return encode_array(np.frombuffer(raw, dtype=dt).reshape(shape), level=level,
                            chunk_t=chunk_t, threads=threads, quant_bits=quant_bits,
                            mode=mode, entropy=entropy)
    return bytes([_M_RAW, _entropy.eid("zstd")]) + _entropy.compress(
        raw, "zstd", level, threads)


def decode_bytes(buf: bytes, shape, dtype, chunk_t: int | None = None) -> bytes:
    dt = np.dtype(dtype)
    if buf[:4] == _BLK_TAG or buf[0] != _M_RAW:
        return decode_array(buf, shape, dt, chunk_t=chunk_t).tobytes()
    return _entropy.decompress(buf[2:], _entropy.name(buf[1]),
                               int(np.prod(shape)) * dt.itemsize)
