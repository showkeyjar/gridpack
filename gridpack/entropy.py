"""gridpack.entropy -- 熵编码后端。

生产默认用 ``zstd``（多线程、速度快），``zlib`` 作为无第三方依赖的兜底。
``rc0``/``rc1``/``huff`` 是研究性后端，纯 Python 实现，只适合小数据块，
用于横向对比"这类数据还能再压多少"：

* ``rc0`` 自适应二进制区间编码，上下文 = 位序号（8 个概率）
* ``rc1`` 自适应二进制区间编码，上下文 = 前一字节 × 位序号（2048 个概率）
* ``huff`` 静态 0 阶 Huffman（码长表随块存储）

区间编码器采用 LZMA 风格的无进位二进制 range coder：32 位 low/range，
11 位精度概率，移位 5 的自适应更新。解码端按同样的上下文顺序重建概率表，
因此不需要额外传输模型。
"""

from __future__ import annotations

import heapq
import zlib

import numpy as np

try:                                     # zstd 为可选依赖
    import zstandard as _zstd
    HAVE_ZSTD = True
except Exception:                        # pragma: no cover
    _zstd = None
    HAVE_ZSTD = False

__all__ = ["METHODS", "DEFAULT_METHOD", "SLOW_METHODS", "HAVE_ZSTD",
           "compress", "decompress", "eid", "name"]

METHODS = ("zstd", "zlib", "rc0", "rc1", "huff")
DEFAULT_METHOD = "zstd"
SLOW_METHODS = ("rc0", "rc1", "huff")

_EID = {m: i for i, m in enumerate(METHODS)}
_NAME = {i: m for m, i in _EID.items()}

_KTOP = 1 << 24
_PBITS = 11
_PMAX = 1 << _PBITS
_PINIT = _PMAX // 2
_MOVE = 5


def eid(method: str) -> int:
    try:
        return _EID[method]
    except KeyError:
        raise ValueError("unknown entropy method %r (choose from %s)"
                         % (method, ", ".join(METHODS)))


def name(ident: int) -> str:
    try:
        return _NAME[int(ident)]
    except KeyError:
        raise ValueError("unknown entropy id %r" % (ident,))


# --------------------------------------------------------------------------
# 自适应二进制区间编码
# --------------------------------------------------------------------------
class _RcEncoder:
    __slots__ = ("low", "rng", "cache", "cache_size", "out")

    def __init__(self):
        self.low = 0
        self.rng = 0xFFFFFFFF
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & 0xFFFFFFFF

    def bit(self, probs, i, b):
        p = probs[i]
        bound = (self.rng >> _PBITS) * p
        if b:
            self.low += bound
            self.rng -= bound
            probs[i] = p - (p >> _MOVE)
        else:
            self.rng = bound
            probs[i] = p + ((_PMAX - p) >> _MOVE)
        while self.rng < _KTOP:
            self.rng = (self.rng << 8) & 0xFFFFFFFF
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class _RcDecoder:
    __slots__ = ("data", "pos", "code", "rng")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 1                      # 第 1 个字节是编码器的初始 cache，丢弃
        self.code = 0
        self.rng = 0xFFFFFFFF
        for _ in range(4):
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF

    def _next(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def bit(self, probs, i) -> int:
        p = probs[i]
        bound = (self.rng >> _PBITS) * p
        if self.code < bound:
            self.rng = bound
            probs[i] = p + ((_PMAX - p) >> _MOVE)
            b = 0
        else:
            self.code -= bound
            self.rng -= bound
            probs[i] = p - (p >> _MOVE)
            b = 1
        while self.rng < _KTOP:
            self.rng = (self.rng << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF
        return b


def _rc_encode(buf: bytes, order1: bool) -> bytes:
    nctx = 2048 if order1 else 8
    probs = [_PINIT] * nctx
    enc = _RcEncoder()
    prev = 0
    for byte in buf:
        base = prev << 3 if order1 else 0
        for k in range(8):
            enc.bit(probs, base + k, (byte >> (7 - k)) & 1)
        prev = byte
    return enc.finish()


def _rc_decode(data: bytes, out_len: int, order1: bool) -> bytes:
    nctx = 2048 if order1 else 8
    probs = [_PINIT] * nctx
    dec = _RcDecoder(data)
    out = bytearray(out_len)
    prev = 0
    for j in range(out_len):
        base = prev << 3 if order1 else 0
        byte = 0
        for k in range(8):
            byte = (byte << 1) | dec.bit(probs, base + k)
        out[j] = byte
        prev = byte
    return bytes(out)


# --------------------------------------------------------------------------
# 静态 0 阶 Huffman
# --------------------------------------------------------------------------
def _huff_lengths(counts) -> list:
    lens = [0] * 256
    live = [(int(c), i, i) for i, c in enumerate(counts) if c > 0]
    if not live:
        return lens
    if len(live) == 1:
        lens[live[0][1]] = 1
        return lens
    heapq.heapify(live)
    order = 0
    while len(live) > 1:
        c1, _, n1 = heapq.heappop(live)
        c2, _, n2 = heapq.heappop(live)
        order += 1
        heapq.heappush(live, (c1 + c2, order, (n1, n2)))
    stack = [(live[0][2], 0)]
    while stack:
        node, d = stack.pop()
        if isinstance(node, int):
            lens[node] = max(d, 1)
        else:
            stack.append((node[0], d + 1))
            stack.append((node[1], d + 1))
    return lens


def _canonical(lens) -> list:
    maxlen = max(lens) if lens else 0
    if maxlen == 0:
        return [0] * 256
    bl_count = [0] * (maxlen + 1)
    for L in lens:
        if L:
            bl_count[L] += 1
    code = 0
    next_code = [0] * (maxlen + 2)
    for bits in range(1, maxlen + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = [0] * 256
    for i, L in enumerate(lens):
        if L:
            codes[i] = next_code[L]
            next_code[L] += 1
    return codes


def _huff_encode(buf: bytes) -> bytes:
    counts = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256)
    lens = _huff_lengths(counts)
    codes = _canonical(lens)
    out = bytearray(lens)
    acc = 0
    nbits = 0
    for byte in buf:
        L = lens[byte]
        acc = (acc << L) | codes[byte]
        nbits += L
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def _huff_decode(data: bytes, out_len: int) -> bytes:
    lens = list(data[:256])
    maxlen = max(lens) if lens else 0
    if maxlen == 0:
        return bytes(out_len)
    bl_count = [0] * (maxlen + 1)
    for L in lens:
        if L:
            bl_count[L] += 1
    code = 0
    next_code = [0] * (maxlen + 2)
    for bits in range(1, maxlen + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    lut = {}
    for i, L in enumerate(lens):
        if L:
            lut[(L, next_code[L])] = i
            next_code[L] += 1
    body = data[256:]
    out = bytearray(out_len)
    pos = 0
    total_bits = len(body) * 8
    for j in range(out_len):
        c = 0
        sym = -1
        for L in range(1, maxlen + 1):
            if pos >= total_bits:
                break
            c = (c << 1) | ((body[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
            s = lut.get((L, c))
            if s is not None:
                sym = s
                break
        out[j] = sym if sym >= 0 else 0
    return bytes(out)


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
def compress(buf: bytes, method: str = DEFAULT_METHOD, level: int = 12,
             threads: int = 1) -> bytes:
    if method == "zstd":
        if not HAVE_ZSTD:
            return zlib.compress(buf, 6)
        return _zstd.ZstdCompressor(level=level, threads=threads).compress(buf)
    if method == "zlib":
        return zlib.compress(buf, level if level <= 9 else 9)
    if method == "rc0":
        return _rc_encode(buf, False)
    if method == "rc1":
        return _rc_encode(buf, True)
    if method == "huff":
        return _huff_encode(buf)
    raise ValueError("unknown entropy method %r" % method)


def decompress(blob: bytes, method: str, out_len: int) -> bytes:
    if method == "zstd":
        if not HAVE_ZSTD:
            return zlib.decompress(blob)
        return _zstd.ZstdDecompressor().decompress(blob, max_output_size=max(out_len, 1))
    if method == "zlib":
        return zlib.decompress(blob)
    if method == "rc0":
        return _rc_decode(blob, out_len, False)
    if method == "rc1":
        return _rc_decode(blob, out_len, True)
    if method == "huff":
        return _huff_decode(blob, out_len)
    raise ValueError("unknown entropy method %r" % method)
