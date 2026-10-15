"""gridpack 基准脚本：在真实格点文件上对比 表示层 × 时间模型 × 熵编码。

用法::

    python scripts/bench.py <file.nc> [--nt 48] [--level 12]
                            [--entropy zstd zlib] [--mode none spatial dt]

输出一张二维表（行 = 表示层/时间模型，列 = 熵编码），数值为压缩比（越大越好）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py                                                  # noqa: E402
from gridpack import codec, entropy, lattice                 # noqa: E402

MB = 2 ** 20


def pick_var(path, nt):
    with h5py.File(path, "r") as f:
        best = None
        for k in f.keys():
            o = f[k]
            if isinstance(o, h5py.Dataset) and o.ndim == 3 and (best is None or o.size > best.size):
                best = o
        if best is None:
            raise SystemExit("no 3-D dataset found in %s" % path)
        arr = np.ascontiguousarray(best[:nt])
        return best.name, tuple(best.shape), arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--nt", type=int, default=48, help="取样时次数")
    ap.add_argument("--level", type=int, default=12)
    ap.add_argument("--chunk-t", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--entropy", nargs="+", default=["zstd", "zlib"],
                    choices=list(entropy.METHODS))
    ap.add_argument("--mode", nargs="+", default=["none", "spatial", "dt"])
    a = ap.parse_args()

    name, full, arr = pick_var(a.src, a.nt)
    raw = arr.nbytes
    e = lattice.detect_lattice(arr)
    print("file      : %s" % os.path.basename(a.src))
    print("dataset   : %s  full=%s  used=%s" % (name, full, arr.shape))
    print("raw       : %.2f MB (%.1f%% of full file)" % (raw / MB, 100.0 * raw / (int(np.prod(full)) * 4)))
    print("格点指数 E: %s" % ("%d" % e if e is not None else "无（走保序映射）"))
    print()

    reps = []
    if e is not None:
        reps.append(("格点 E=%d" % e, True))
        reps.append(("保序映射", False))
    else:
        reps.append(("保序映射", False))

    hdr = "%-12s %-8s" % ("表示", "模型") + "".join("%14s" % m for m in a.entropy)
    print(hdr)
    print("-" * len(hdr))
    for rep_name, use_lat in reps:
        for mode in a.mode:
            cells = []
            for ent in a.entropy:
                t0 = time.time()
                try:
                    if use_lat:
                        blob = _encode_lattice(arr, a, mode, ent)
                    else:
                        blob = _encode_plain(arr, a, mode, ent)
                except Exception as ex:
                    cells.append("%14s" % ("skip:%s" % type(ex).__name__))
                    continue
                dt = time.time() - t0
                cells.append("%9.2fx%4.1fs" % (raw / len(blob), dt))
            print("%-12s %-8s" % (rep_name, mode) + "".join(cells))
    print()
    print("(压缩比 = 原始 float 字节 / 压缩后字节；越大越好)")


def _encode_lattice(arr, a, mode, ent):
    """强制走格点表示（把 detect_lattice 的结果直接喂进去）。"""
    e = lattice.detect_lattice(arr)
    if e is None:
        raise RuntimeError("no lattice")
    blk = codec._encode_block(arr.reshape(codec._dims3(arr.shape)), a.level,
                              a.threads, 0, mode, ent, lat_e=e)
    return blk


def _encode_plain(arr, a, mode, ent):
    """强制走保序映射（lat_e=None 表示不做格点检测）。"""
    return codec._encode_block(arr.reshape(codec._dims3(arr.shape)), a.level,
                               a.threads, 0, mode, ent, lat_e=None)


if __name__ == "__main__":
    main()
