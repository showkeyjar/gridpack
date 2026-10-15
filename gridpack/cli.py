"""gridpack 命令行入口。

  python -m gridpack compress   <in.nc> <out.gpz> [--level 12] [--mode auto]
                                [--entropy zstd] [--threads 4] [--quant-bits 0]
                                [--file-chunk N]
  python -m gridpack decompress <in.gpz> <out.nc>
  python -m gridpack verify     <in.nc> <in.gpz>
  python -m gridpack info       <in.gpz>
  python -m gridpack bench      <in.nc> [--nt 48] [--level 12] [--entropy zstd zlib]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import codec, container, entropy, netcdf_io

MB = 2 ** 20


def _fmt(n):
    return "%.2f MB" % (n / MB)


def _rep(lat_e):
    return ("格点 E=%d" % lat_e) if lat_e is not None else "保序映射"


def cmd_compress(a):
    t0 = time.time()
    st = netcdf_io.compress_file(a.src, a.dst, level=a.level, chunk_t=a.chunk_t,
                                 threads=a.threads, quant_bits=a.quant_bits,
                                 mode=a.mode, entropy=a.entropy,
                                 file_chunk=a.file_chunk,
                                 verify=(a.quant_bits == 0))
    dt = time.time() - t0
    print("compress  %s -> %s" % (a.src, a.dst))
    print("  节律识别  : %s" % st["cadence"])
    for name, m, e in st["chosen"]:
        print("  表示/模型 : %-34s %-8s %s" % (name, m, _rep(e)))
    print("  熵编码    : %s" % a.entropy)
    print("  datasets  : %d   payloads: %d" % (st["entries"], st["payloads"]))
    print("  raw       : %s" % _fmt(st["raw_bytes"]))
    print("  source .nc: %s" % _fmt(st["src_bytes"]))
    print("  .gpz      : %s" % _fmt(st["gpz_bytes"]))
    print("  ratio     : %.2fx vs raw   %.2fx vs source .nc" % (st["ratio_vs_raw"], st["vs_src"]))
    print("  elapsed   : %.1fs  (%.1f MB/s raw)" % (dt, st["raw_bytes"] / MB / max(dt, 1e-9)))


def cmd_decompress(a):
    t0 = time.time()
    st = netcdf_io.decompress_file(a.src, a.dst)
    print("decompress %s -> %s" % (a.src, a.dst))
    print("  .gpz     : %s" % _fmt(st["gpz_bytes"]))
    print("  restored : %s" % _fmt(st["nc_bytes"]))
    print("  elapsed  : %.1fs" % (time.time() - t0))


def cmd_verify(a):
    st = netcdf_io.verify_file(a.src, a.gpz)
    for path, same in st["details"]:
        print("  %-40s %s" % (path, "OK" if same else "MISMATCH"))
    print("EXACT" if st["exact"] else "NOT EXACT")
    return 0 if st["exact"] else 1


def cmd_info(a):
    m = container.read_meta(a.gpz)
    total_raw = sum(p["raw_nbytes"] for e in m["entries"] for p in e["parts"])
    total_pay = sum(p["length"] for p in m["payload_index"])
    print("source    : %s (%s)" % (m.get("source_name"), _fmt(m.get("source_bytes", 0))))
    print("codec v%s : mode=%s level=%s quant_bits=%s entropy=%s"
          % (m.get("codec_version"), m.get("mode"), m.get("level"),
             m.get("quant_bits"), m.get("entropy", "zstd")))
    print("节律      : %s" % m.get("cadence_note"))
    print("datasets  : %d   raw=%s  payload=%s  ratio=%.2fx"
          % (len(m["entries"]), _fmt(total_raw), _fmt(total_pay), total_raw / max(total_pay, 1)))
    pi = 0
    for e in m["entries"]:
        pay = sum(p["length"] for p in m["payload_index"][pi:pi + len(e["parts"])])
        pi += len(e["parts"])
        raw = sum(p["raw_nbytes"] for p in e["parts"])
        print("  %-34s %-12s %-9s mode=%-8s raw=%9s payload=%9s %.2fx"
              % (e["path"], e["dtype"], _rep(e.get("lattice_E")), e.get("mode", "?"),
                 _fmt(raw), _fmt(pay), raw / max(pay, 1)))


def cmd_bench(a):
    import h5py

    with h5py.File(a.src, "r") as f:
        best = None
        for k in f.keys():
            o = f[k]
            if isinstance(o, h5py.Dataset) and o.ndim == 3 and (best is None or o.size > best.size):
                best = o
        full = best.shape
        arr = np.ascontiguousarray(best[:a.nt])
        name = best.name
    raw = arr.nbytes
    raw_full = int(np.prod(full)) * arr.dtype.itemsize
    lat_e = codec.detect_lattice_e(arr)
    print("%s  var=%s full=%s used=%s" % (os.path.basename(a.src), name, full, arr.shape))
    print("表示: %s" % _rep(lat_e))
    for ent in a.entropy:
        for mode in ("none", "spatial", "dt", "lag24"):
            t0 = time.time()
            try:
                blob = codec.encode_array(arr, level=a.level, chunk_t=a.chunk_t,
                                          threads=a.threads, mode=mode, entropy=ent)
            except Exception as ex:
                print("  %-5s %-8s skip (%s)" % (ent, mode, ex))
                continue
            dt = time.time() - t0
            print("  %-5s %-8s ratio=%5.2fx  projected_full=%8s  %.1fs"
                  % (ent, mode, raw / len(blob), _fmt(len(blob) / raw * raw_full), dt))


def main(argv=None):
    p = argparse.ArgumentParser(prog="gridpack", description="格点数据（ERA5 等）无损压缩工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help=".nc -> .gpz")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--level", type=int, default=12, help="熵编码等级（zstd: 1-22, zlib: 1-9）")
    c.add_argument("--entropy", default="zstd", choices=list(entropy.METHODS),
                   help="熵编码后端，默认 zstd")
    c.add_argument("--chunk-t", type=int, default=16, help="编码内部时间分块（0=不分块）")
    c.add_argument("--file-chunk", type=int, default=0, help="文件级时间分块（0=按内存自适应）")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--quant-bits", type=int, default=0, help="0=严格无损；>0 为有容差模式")
    c.add_argument("--mode", default="auto",
                   help="auto | none | spatial | dt | lagN（N 为时间滞后步数）")
    c.set_defaults(func=cmd_compress)

    d = sub.add_parser("decompress", help=".gpz -> .nc")
    d.add_argument("src")
    d.add_argument("dst")
    d.set_defaults(func=cmd_decompress)

    v = sub.add_parser("verify", help="比对 .nc 与 .gpz 是否逐字节一致")
    v.add_argument("src")
    v.add_argument("gpz")
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser("info", help="查看 .gpz 元数据")
    i.add_argument("gpz")
    i.set_defaults(func=cmd_info)

    b = sub.add_parser("bench", help="在 .nc 上测各种时间模型/熵编码的压缩率")
    b.add_argument("src")
    b.add_argument("--nt", type=int, default=48)
    b.add_argument("--level", type=int, default=12)
    b.add_argument("--entropy", nargs="+", default=["zstd"], choices=list(entropy.METHODS))
    b.add_argument("--chunk-t", type=int, default=16)
    b.add_argument("--threads", type=int, default=4)
    b.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    args.chunk_t = args.chunk_t or None
    args.file_chunk = args.file_chunk or None
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
