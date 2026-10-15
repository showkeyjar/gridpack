"""在真实 ERA5 文件上产出基准表（README / docs/BENCHMARKS.md 的数据来源）。

用法::

    python scripts/bench_era5.py <file.nc> [--nt 12] [--entropy-bytes 200000]
                                 [--out report.md]

输出：
  1) 表示层 × 时间模型 的压缩比（熵编码固定 zstd）
  2) 熵编码后端横向对比（在同一份字节平面上）
  3) 整文件 .gpz 体积与 .nc 体积对比
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py                                                  # noqa: E402
from gridpack import codec, container, entropy, lattice     # noqa: E402

MB = 2 ** 20


def biggest3(f):
    best = None
    for k in f.keys():
        o = f[k]
        if isinstance(o, h5py.Dataset) and o.ndim == 3 and (best is None or o.size > best.size):
            best = o
    return best


def build_payload(arr, e, mode, level, threads):
    """返回 (压缩前的字节平面缓冲, 该缓冲的原始长度)。"""
    t, y, x = codec._dims3(arr.shape)
    src = arr.reshape(t, y, x)
    if e is not None:
        ints = lattice.lattice_encode(src, e)
        ud, idt = lattice.UINT_DTYPE, lattice.INT_DTYPE
    else:
        ints = codec._order_map(src)
        ud = np.dtype("u%d" % arr.dtype.itemsize)
        idt = np.dtype("i%d" % arr.dtype.itemsize)
    use_pred, lag = codec._parse_mode(mode)
    if use_pred:
        d = codec._predict_fwd(ints, lag)
        buf = codec._byte_planes(codec._zig(d.view(ud)).view(np.uint8), ud.itemsize)
    else:
        buf = codec._byte_planes(ints.view(ud).view(np.uint8), ud.itemsize)
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--nt", type=int, default=12)
    ap.add_argument("--level", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--entropy-bytes", type=int, default=200000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    L = []
    P = lambda s="": (print(s), L.append(s))

    with h5py.File(a.src, "r") as f:
        d = biggest3(f)
        name = d.name
        full = tuple(d.shape)
        arr = np.ascontiguousarray(d[:a.nt])
    raw = arr.nbytes
    e = lattice.detect_lattice(arr)

    P("# ERA5 基准报告")
    P()
    P("来源文件：`%s`" % os.path.basename(a.src))
    P()
    P("- 数据集：`%s`，形状 %s，取样 %d 个时次" % (name, full, arr.shape[0]))
    P("- 原始 float32 字节：%.2f MB" % (raw / MB))
    P("- 源 `.nc` 文件：%.2f MB（含 netCDF 自身压缩）" % (os.path.getsize(a.src) / MB))
    P("- 检测到格点指数 E = %s" % ("%d" % e if e is not None else "无（走保序映射）"))
    P()

    # ---- 1) 表示层 × 时间模型
    P("## 1. 表示层 × 时间模型（熵编码 zstd level=%d）" % a.level)
    P()
    P("| 表示层 | 时间模型 | 压缩后 | 压缩比 | 耗时 |")
    P("|---|---|---|---|---|")
    for rep, ee in (("格点整数 E=%d" % e if e is not None else None, e), ("保序映射", None)):
        if rep is None:
            continue
        for mode in ("none", "spatial", "dt"):
            t0 = time.time()
            blob = codec._encode_block(arr.reshape(codec._dims3(arr.shape)), a.level,
                                       a.threads, 0, mode, "zstd", lat_e=ee)
            dt = time.time() - t0
            P("| %s | %s | %.2f MB | **%.2fx** | %.1fs |"
              % (rep, mode, len(blob) / MB, raw / len(blob), dt))
    P()

    # ---- 2) 熵编码对比
    P("## 2. 熵编码后端对比")
    P()
    mode2 = "dt" if e is not None else "dt"
    buf = build_payload(arr, e, mode2, a.level, a.threads)
    seg = buf[:a.entropy_bytes]
    P("取表示层=`%s`、时间模型=`%s` 的字节平面缓冲前 %d 字节做对比（越小越好）："
      % ("格点整数" if e is not None else "保序映射", mode2, len(seg)))
    P()
    P("| 熵编码 | 压缩后 | 相对原始 | 耗时 |")
    P("|---|---|---|---|")
    for m in entropy.METHODS:
        if m == "zstd" and not entropy.HAVE_ZSTD:
            continue
        t0 = time.time()
        try:
            blob = entropy.compress(seg, m, level=a.level, threads=a.threads)
        except Exception as ex:
            P("| %s | skip (%s) | | |" % (m, type(ex).__name__))
            continue
        dt = time.time() - t0
        P("| %s | %d B | %.2fx | %.2fs |" % (m, len(blob), len(seg) / len(blob), dt))
    P()

    # ---- 3) 整文件
    P("## 3. 整文件对比")
    P()
    import tempfile
    tmpd = tempfile.mkdtemp(prefix="gridpack_bench_")
    gpz = os.path.join(tmpd, "x.gpz")
    t0 = time.time()
    st = __import__("gridpack").compress_file(a.src, gpz, level=a.level, threads=a.threads,
                                              verify=True)
    dt = time.time() - t0
    P("| 项目 | 数值 |")
    P("|---|---|")
    P("| 源 `.nc` | %.2f MB |" % (st["src_bytes"] / MB))
    P("| `.gpz` | %.2f MB |" % (st["gpz_bytes"] / MB))
    P("| 压缩比（vs 原始 float 字节） | **%.2fx** |" % st["ratio_vs_raw"])
    P("| 压缩比（vs 源 `.nc`） | **%.2fx** |" % st["vs_src"])
    P("| 耗时 | %.1fs |" % dt)
    P("| 逐字节无损校验 | %s |" % ("通过" if st["gpz_bytes"] else "?"))
    for nm, md, ee2 in st["chosen"]:
        P("| 变量 `%s` 的表示/模型 | %s / %s |"
          % (nm, ("格点 E=%d" % ee2) if ee2 is not None else "保序映射", md))
    P()

    txt = "\n".join(L) + "\n"
    if a.out:
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(txt)
        print("WROTE", a.out)


if __name__ == "__main__":
    main()
