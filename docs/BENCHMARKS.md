# Benchmarks

All numbers below are produced by the code in this repository and can be regenerated with:

```bash
python scripts/bench_era5.py <file.nc> --nt 12 --out report.md
```

## Test data

| | |
|---|---|
| Source | ERA5 reanalysis, 10 m eastward wind |
| Variable | `eastward_wind_at_10_metres` |
| Shape | `(120, 721, 1440)` float32 (a 120-time-step subset of a monthly file) |
| Raw float bytes | 47.53 MB |
| Source `.nc` size | 475.28 MB (this subset is stored uncompressed) |
| Detected lattice exponent | `E = -4` (values are multiples of 0.0625 m/s) |

## 1. Representation × time model

Entropy backend fixed at `zstd`, level 12.

| Representation | Time model | Compressed | Ratio | Time |
|---|---|---|---|---|
| lattice E=-4 | none | 8.35 MB | 5.69× | 1.4 s |
| lattice E=-4 | spatial | 5.05 MB | 9.40× | 1.7 s |
| **lattice E=-4** | **dt** | **4.59 MB** | **10.36×** | 1.7 s |
| order map | none | 9.05 MB | 5.25× | 1.2 s |
| order map | spatial | 10.83 MB | 4.39× | 2.0 s |
| order map | dt | 11.07 MB | 4.29× | 2.0 s |

Observations:

- The lattice representation roughly **doubles** the achievable ratio (10.36× vs 5.25×).
- Under the plain order map, prediction **hurts** (5.25× → 4.29×). The order-preserving bit
  map has a ~2^31 step at the zero crossing, so a field that changes sign yields huge
  residuals; the byte-plane-only variant avoids that. With integer-lattice residuals the
  problem disappears and `dt` becomes the best model.
- `spatial` vs `dt`: the time difference is what buys the extra ~10 % on a 10 m wind field.

## 2. Entropy backend

Same byte-plane buffer, first 200 000 bytes (representation = lattice, model = `dt`).
Smaller is better.

| Backend | Compressed | vs raw | Time | Notes |
|---|---|---|---|---|
| zstd (level 12) | 74 363 B | 2.69× | 0.03 s | production default |
| zlib (level 12 → 9) | 75 024 B | 2.67× | 0.22 s | stdlib fallback |
| rc0 | 68 216 B | 2.93× | 0.45 s | adaptive range coder, context = bit index |
| **rc1** | **65 015 B** | **3.08×** | 0.48 s | adaptive range coder, context = prev byte × bit index |
| huff | 73 150 B | 2.73× | 0.03 s | static order-0 canonical Huffman |

- `rc1` is **12.6 % smaller than zstd** and 4.7 % smaller than `rc0` on this data, which
  confirms that byte-plane data still carries usable inter-byte context.
- The cost is ~15× the runtime in pure Python, so `rc1` is an offline/archival option, not a
  default. `rc0`/`rc1`/`huff` are intentionally limited to small blocks.

## 3. Whole-file result

| | |
|---|---|
| Source `.nc` | 475.28 MB |
| `.gpz` | 45.83 MB |
| Ratio vs raw float bytes | **10.37×** |
| Ratio vs source `.nc` | 10.37× |
| Wall clock | 47.6 s |
| Byte-for-byte verification | **pass** |

Per-variable decisions made automatically by `--mode auto`:

| Dataset | Representation | Mode |
|---|---|---|
| `eastward_wind_at_10_metres` | lattice E=-4 | `dt` |
| `lat` | lattice E=-2 | `spatial` |
| `lon` | lattice E=-2 | `spatial` |
| `time0` | lattice E=4 | `spatial` |

## 4. How to read the ratio

The headline ratio depends on what you compare against:

- **vs raw float bytes** — the honest measure of how much information the codec removed.
  This is what the tables above report.
- **vs the source `.nc`** — depends entirely on how that file was written. NetCDF files that
  are already zlib-compressed and quantised leave less headroom, so the gain over them is
  smaller. This subset `.nc` was stored uncompressed, so the two numbers coincide here.

Across a set of real ERA5 monthly files, quantised months (lattice exponent present,
low-entropy mantissas) land around **10×** versus raw, while full-precision unquantised
months land around **4×**.

## 5. Throughput and memory

- ~10 MB/s of raw data per process with `zstd`; roughly linear in cores when compressing
  several files in parallel.
- Peak memory is governed by `--file-chunk` (default targets ~96 MB per block), not by the
  file size. `--chunk-t` bounds the inner codec block.
- Raising `--level` from 12 to 19 costs about 18× the time for roughly 3 % smaller output;
  level 12 is the recommended default.
