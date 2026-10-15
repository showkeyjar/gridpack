"""gridpack.lattice -- 精确格点表示层。

很多格点数据（如 ERA5 各要素）的浮点值并不是任意的 IEEE-754 数，而是**全部落在同一个
二进制格点 ``2**E`` 上**（例如 10m 风场量化到 0.0625 = 2**-4，总降水量化到 2**-14）。
此时可以把浮点无损改写成整数 ``k = v / 2**E``：整数的动态范围比原始 32/64 位位型小得多，
后续的时间/空间预测残差更小、字节平面更干净，熵编码收益很大。

判定是**精确**的，不做任何近似：

* float32 非零值可写成 ``v = sig * 2**(exp-150)``，``sig`` 是 24 位有效数（正规数为
  ``mant | 0x800000``，次正规数为 ``mant``，此时指数固定为 -149）。
* ``v`` 是 ``2**E`` 的整数倍 ⟺ ``tz(sig) + exp - 150 >= E``（``tz`` 为二进制尾零个数）。
* 因此全体取值可用的**最大**格点指数就是 ``E_max = min(tz(sig) + exp - 150)``。

取 ``E = E_max`` 是精确无损的；取更小的 ``E`` 同样无损（只是整数更大）。本模块在返回
``E`` 之前还会验证改写后的整数能安全放进 int32（给后续 zig-zag 与预测留足余量），
否则返回 ``None``，由调用方回退到保序映射路径。

负零（``-0.0``）与正零位型不同而整数表示无法区分，故单独用稀疏位图记录，
保证解码后**逐位一致**。
"""

from __future__ import annotations

import zlib

import numpy as np

__all__ = ["detect_lattice", "lattice_encode", "lattice_decode",
           "negative_zero_mask", "restore_negative_zero", "INT_WIDTH"]

# 格点整数一律用 int32 承载（动态范围已由 detect_lattice 限定在 2**29 以内）
INT_WIDTH = 4
INT_DTYPE = np.dtype("i4")
UINT_DTYPE = np.dtype("u4")

# 整数绝对值上限：给 zig-zag（翻倍）与一阶差分（再翻倍）留出余量
_K_LIMIT = 2.0 ** 29


def _trailing_zeros(x: np.ndarray, nbits: int) -> np.ndarray:
    """向量化二进制尾零个数（x 为非零非负整数数组）。"""
    n = np.zeros(x.shape, dtype=np.int64)
    y = x.astype(np.int64).copy()
    steps = []
    k = 1
    while k < nbits:
        steps.append(k)
        k *= 2
    for k in reversed(steps):
        mask = (1 << k) - 1
        z = (y & mask) == 0
        n += np.where(z, k, 0)
        y = np.where(z, y >> k, y)
    return n


def detect_lattice(a: np.ndarray):
    """返回全体取值共同格点的最大指数 E（int），无法使用格点表示时返回 None。"""
    v = np.ascontiguousarray(a)
    if v.dtype not in (np.float32, np.float64):
        return None
    flat = v.reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        return None
    nz = flat != 0
    if not nz.any():
        return None                      # 全零场：整数表示无意义
    f = flat[nz]

    if v.dtype == np.float32:
        u = f.view(np.uint32)
        exp = (u >> np.uint32(23)).astype(np.int64)
        mant = (u & np.uint32(0x7FFFFF)).astype(np.int64)
        sig = np.where(exp == 0, mant, mant | 0x800000)
        base = np.where(exp == 0, -149, exp - 150)
        nbits = 24
    else:
        u = f.view(np.uint64)
        exp = (u >> np.uint64(52)).astype(np.int64)
        mant = (u & np.uint64(0xFFFFFFFFFFFFF)).astype(np.int64)
        sig = np.where(exp == 0, mant, mant | (1 << 52))
        base = np.where(exp == 0, -1074, exp - 1075)
        nbits = 53

    e_max = int((_trailing_zeros(sig, nbits) + base).min())
    if not (-1074 <= e_max <= 40):
        return None
    if float(np.abs(f).max()) / (2.0 ** e_max) > _K_LIMIT:
        return None                      # 整数会溢出 int32，放弃格点路径
    return e_max


def lattice_encode(a: np.ndarray, e: int) -> np.ndarray:
    """浮点 -> 整数 k = v / 2**e（精确，无舍入误差）。"""
    v = np.ascontiguousarray(a)
    scale = float(2.0 ** (-e))
    if v.dtype == np.float32:
        q = np.rint(v.astype(np.float64) * scale)
    else:
        q = np.rint(v * scale)
    return q.astype(np.int32)


def lattice_decode(k: np.ndarray, e: int, dtype) -> np.ndarray:
    """整数 -> 浮点 v = k * 2**e。2 的幂缩放与 int32 取值范围在 float64 中都是精确的。"""
    scale = float(2.0 ** e)
    q = np.ascontiguousarray(k).astype(np.int64).astype(np.float64) * scale
    return q.astype(dtype)


def negative_zero_mask(a: np.ndarray) -> bytes:
    """记录 -0.0 的位置（稀疏位图，zlib 压缩）；没有 -0.0 时返回 b""。"""
    v = np.ascontiguousarray(a)
    if v.dtype == np.float32:
        m = v.view(np.uint32) == np.uint32(0x80000000)
    elif v.dtype == np.float64:
        m = v.view(np.uint64) == np.uint64(0x8000000000000000)
    else:
        return b""
    m = m.reshape(-1)
    if not m.any():
        return b""
    return zlib.compress(np.packbits(m).tobytes(), 6)


def restore_negative_zero(v: np.ndarray, nz: bytes, dtype) -> np.ndarray:
    """按位图把 -0.0 的符号位恢复回去（原地）。"""
    if not nz:
        return v
    mask = np.unpackbits(np.frombuffer(zlib.decompress(nz), dtype=np.uint8))[:v.size].astype(bool)
    v.reshape(-1)[mask] = np.dtype(dtype).type(-0.0)
    return v
