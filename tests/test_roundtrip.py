"""gridpack 自检：所有编解码路径都必须严格无损。

直接运行： python tests/test_roundtrip.py
真实数据测试： 设置环境变量 GRIDPACK_TEST_NC 指向一个 .nc 文件
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gridpack as gp                                        # noqa: E402
from gridpack import codec, entropy, lattice                 # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print("%-46s %s %s" % (name, "PASS" if cond else "FAIL", extra))
    if not cond:
        FAILED.append(name)


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------- 表示层
def test_order_map():
    for dt in (np.float32, np.float64):
        v = np.array([0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30,
                      np.inf, -np.inf, np.nan, np.finfo(dt).tiny,
                      np.finfo(dt).max, -np.finfo(dt).max, 3.5, -3.5], dtype=dt)
        back = codec._order_unmap(codec._order_map(v), dt)
        check("order map roundtrip (%s)" % np.dtype(dt).name, bits_equal(v, back))
    # 保序性：映射在无符号视角下单调（负浮点占低半区，正浮点占高半区）
    v = np.sort(np.random.RandomState(0).randn(10000).astype(np.float32))
    m = codec._order_map(v).view(np.uint32).astype(np.uint64)
    check("order map is monotone (unsigned)", bool(np.all(np.diff(m) >= 0)))


def test_lattice():
    rng = np.random.RandomState(1)
    for dt, e in ((np.float32, -4), (np.float32, -16), (np.float64, -8)):
        k = rng.randint(-5000, 5000, size=(3, 7, 5))
        v = (k.astype(np.float64) * 2.0 ** e).astype(dt)
        got = lattice.detect_lattice(v)
        check("detect_lattice E=%d (%s)" % (e, np.dtype(dt).name), got is not None and got >= e,
              "got=%s" % got)
        e_use = got if got is not None else e
        back = lattice.lattice_decode(lattice.lattice_encode(v, e_use), e_use, dt)
        check("lattice roundtrip E=%d (%s)" % (e, np.dtype(dt).name), bits_equal(v, back))

    v = rng.randn(1000).astype(np.float32)
    check("detect_lattice rejects non-lattice", lattice.detect_lattice(v) is None)
    v = np.array([0.0, 0.5, np.nan], dtype=np.float32)
    check("detect_lattice rejects NaN", lattice.detect_lattice(v) is None)


def test_negative_zero():
    v = np.array([0.0, -0.0, 1.0, -0.0, 0.0], dtype=np.float32)
    blob = codec.encode_array(v, mode="none")
    back = codec.decode_array(blob, v.shape, v.dtype)
    check("negative zero preserved (lattice path)", bits_equal(v, back))


# ---------------------------------------------------------------- 熵编码
def test_entropy():
    rng = np.random.RandomState(2)
    buf = bytes(rng.randint(0, 8, size=40000))          # 低熵，贴近位面数据
    for m in entropy.METHODS:
        if m == "zstd" and not entropy.HAVE_ZSTD:
            continue
        blob = entropy.compress(buf, m, level=6)
        back = entropy.decompress(blob, m, len(buf))
        check("entropy %-5s roundtrip" % m, back == buf,
              "%.3f -> %.3f" % (len(buf) / 1024, len(blob) / 1024))
    for m in ("zlib", "huff"):
        one = bytes([7] * 500)
        check("entropy %-5s single-symbol" % m,
              entropy.decompress(entropy.compress(one, m), m, 500) == one)
        check("entropy %-5s empty" % m, entropy.decompress(entropy.compress(b"", m), m, 0) == b"")


# ---------------------------------------------------------------- 数组级
def test_array_roundtrip():
    rng = np.random.RandomState(3)
    cases = {
        "random f32": rng.randn(4, 9, 11).astype(np.float32),
        "lattice f32": (rng.randint(-800, 800, size=(4, 9, 11)) * 0.0625).astype(np.float32),
        "constant": np.full((3, 5, 5), 7.25, dtype=np.float32),
        "with nan/inf": np.array([[[np.nan, np.inf, -np.inf]] * 3] * 2, dtype=np.float32),
        "float64 lattice": (rng.randint(-300, 300, size=(3, 4, 5)) * 2.0 ** -8),
        "int32 passthrough": rng.randint(-100, 100, size=(2, 3, 4)).astype(np.int32),
        "1d": rng.randn(101).astype(np.float32),
    }
    for name, arr in cases.items():
        for mode in ("none", "spatial", "dt", "auto"):
            blob = codec.encode_array(arr, mode=mode)
            back = codec.decode_array(blob, arr.shape, arr.dtype)
            check("array %-16s mode=%-8s" % (name, mode), bits_equal(arr, back))


def test_chunked():
    rng = np.random.RandomState(4)
    arr = (rng.randint(-500, 500, size=(23, 7, 6)) * 0.0625).astype(np.float32)
    for mode in ("none", "dt"):
        blob = codec.encode_array(arr, chunk_t=4, mode=mode)
        back = codec.decode_array(blob, arr.shape, arr.dtype, chunk_t=4)
        check("chunked roundtrip mode=%s" % mode, bits_equal(arr, back))


def test_encode_bytes():
    rng = np.random.RandomState(5)
    arr = rng.randn(5, 6, 7).astype(np.float32)
    blob = codec.encode_bytes(arr.tobytes(), arr.shape, arr.dtype)
    check("encode_bytes/decode_bytes",
          codec.decode_bytes(blob, arr.shape, arr.dtype) == arr.tobytes())
    i = rng.randint(0, 1000, size=(4, 4)).astype(np.int64)
    blob = codec.encode_bytes(i.tobytes(), i.shape, i.dtype)
    check("encode_bytes int passthrough",
          codec.decode_bytes(blob, i.shape, i.dtype) == i.tobytes())


# ---------------------------------------------------------------- 真实数据
def test_real_file():
    src = os.environ.get("GRIDPACK_TEST_NC")
    if not src or not os.path.exists(src):
        print("real-data test                         SKIP (set GRIDPACK_TEST_NC)")
        return
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="gridpack_")
    gpz = os.path.join(d, "t.gpz")
    st = gp.compress_file(src, gpz, verify=True)
    check("netCDF container compress", st["gpz_bytes"] > 0,
          "%.2fx vs raw" % st["ratio_vs_raw"])
    v = gp.verify_file(src, gpz)
    check("netCDF container lossless", bool(v["exact"]))
    out = os.path.join(d, "restored.nc")
    gp.decompress_file(gpz, out)
    check("netCDF container decompress", os.path.getsize(out) > 0)
    shutil.rmtree(d, ignore_errors=True)


def main():
    test_order_map()
    test_lattice()
    test_negative_zero()
    test_entropy()
    test_array_roundtrip()
    test_chunked()
    test_encode_bytes()
    test_real_file()
    print()
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
