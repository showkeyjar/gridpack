"""把 entropy 参数与格点表示报告接进 codec.py / netcdf_io.py。"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "gridpack")


def patch(fname, pairs):
    p = os.path.join(PKG, fname)
    s = io.open(p, encoding="utf-8").read()
    for old, new in pairs:
        if old not in s:
            print("MISS in %s: %r" % (fname, old[:70]))
            sys.exit(1)
        s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", fname)


patch("codec.py", [
    ('__all__ = ["encode_array", "decode_array", "encode_bytes", "decode_bytes",\n'
     '           "CODEC_VERSION", "MODES", "candidate_modes"]',
     '__all__ = ["encode_array", "decode_array", "encode_bytes", "decode_bytes",\n'
     '           "CODEC_VERSION", "MODES", "candidate_modes", "detect_lattice_e"]'),
    ('def encode_array(a: np.ndarray, level: int = 12,',
     'def detect_lattice_e(a: np.ndarray):\n'
     '    """返回数组所用格点指数 E；不适用时返回 None（供报告与基准使用）。"""\n'
     '    return _lattice.detect_lattice(a)\n'
     '\n'
     '\n'
     'def encode_array(a: np.ndarray, level: int = 12,'),
])

patch("netcdf_io.py", [
    ('                  threads: int = 1, quant_bits: int = 0, mode: str = "auto",\n'
     '                  file_chunk: int | None = None, verify: bool = True) -> dict:',
     '                  threads: int = 1, quant_bits: int = 0, mode: str = "auto",\n'
     '                  entropy: str = "zstd",\n'
     '                  file_chunk: int | None = None, verify: bool = True) -> dict:'),
    ('            if len(modes) == 1:\n'
     '                use_mode = modes[0]\n'
     '            else:\n'
     '                s0, e0 = slices[0]\n'
     '                probe = np.ascontiguousarray(d[()] if d.ndim == 0 else d[s0:e0])\n'
     '                sizes = {}',
     '            s0, e0 = slices[0]\n'
     '            probe = np.ascontiguousarray(d[()] if d.ndim == 0 else d[s0:e0])\n'
     '            if len(modes) == 1:\n'
     '                use_mode = modes[0]\n'
     '            else:\n'
     '                sizes = {}'),
    ('                                                           threads=threads, quant_bits=quant_bits,\n'
     '                                                           mode=mm))',
     '                                                           threads=threads, quant_bits=quant_bits,\n'
     '                                                           mode=mm, entropy=entropy))'),
    ('            chosen.append((name, use_mode))',
     '            lat_e = codec.detect_lattice_e(probe) if quant_bits == 0 else None\n'
     '            chosen.append((name, use_mode, lat_e))'),
    ('                                          chunk_t=chunk_t, threads=threads,\n'
     '                                          quant_bits=quant_bits, mode=use_mode)',
     '                                          chunk_t=chunk_t, threads=threads,\n'
     '                                          quant_bits=quant_bits, mode=use_mode,\n'
     '                                          entropy=entropy)'),
    ('                "mode": use_mode,\n                "file_chunk": fc,',
     '                "mode": use_mode,\n                "lattice_E": lat_e,\n                "file_chunk": fc,'),
    ('        "quant_bits": quant_bits, "mode": mode,',
     '        "quant_bits": quant_bits, "mode": mode, "entropy": entropy,'),
])
print("OK")
