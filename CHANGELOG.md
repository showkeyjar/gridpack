# Changelog

## 0.3.1

Fixes found by a full-scale test over the whole local ERA5 archive
(834 files / 356.6 GB) plus non-ERA5 NetCDF files. See
[`docs/FULLSCALE_TEST.md`](docs/FULLSCALE_TEST.md).

- **Critical: silent precision loss in lattice detection.** The exponent was read as
  `u >> 23`, which folds the sign bit into the exponent and inflates it by 256 for every
  negative value. `E_max` then came out too large, `v / 2**E` stopped being an integer, and
  `np.rint` silently rounded — a lossy result from a "lossless" codec. ERA5 files never
  showed it because `E_max` is a minimum taken over all values and some smaller positive
  value always supplied the correct bound. Fixed by masking the exponent field
  (`(u >> 23) & 0xFF`, `(u >> 52) & 0x7FF` for float64), and by adding `verify_lattice()`,
  which substitutes the values back through `k * 2**E` per block and refuses the lattice
  representation if any value does not round-trip exactly.
- **Crash on 0-dimensional datasets.** `compress_file` indexed `d.shape[0]` when sizing the
  file chunk, which raises `IndexError` for scalar variables such as `crs` in
  `radar.nc`; the part length for 0-D datasets was also recorded as 0, so decoding would
  have produced an empty array. Both fixed.
- **Memory:** `detect_lattice` peaked at ~17.5x the block size (1662 MB for a 95 MB block),
  enough to fail with `MemoryError` on the largest files under memory pressure. It now
  processes in 4 M-value blocks, uses int32 on the float32 path, and `codec` falls back to
  the order-preserving map on `MemoryError` instead of failing the whole file.

Full-scale test outcome after the fixes: 834/834 ERA5 files verified byte-exact;
whole-file ratios 1.71x–28.83x versus the source `.nc` (41 %–97 % smaller).

## 0.3.0

Merge of the two independent implementations (`gridpack` and a lattice-aware codec).

- **New: exact lattice representation** (`gridpack/lattice.py`). Detects the largest binary
  exponent `E` shared by every value of a field, rewrites floats as integers `k = v / 2**E`,
  and records `-0.0` positions as a sparse bitmask to stay bit-exact. On real ERA5 10 m wind
  this roughly doubles the compression ratio.
- **New: pluggable entropy backends** (`gridpack/entropy.py`): `zstd` (default), `zlib`,
  and pure-Python `rc0` / `rc1` adaptive binary range coders plus a static order-0 Huffman
  coder. Selectable from the CLI (`--entropy`) and the Python API.
- **New block layout** (codec version 4): a tag byte now distinguishes raw / predicted /
  byte-planes / lattice+byte-planes / lattice+predicted, and each block records its entropy
  backend id.
- `compress_file` records the detected lattice exponent per dataset, shown by
  `gridpack info` and `compress`.
- New benchmark scripts: `scripts/bench.py` (model × backend sweep) and
  `scripts/bench_era5.py` (full reproducible report).
- Tests extended to cover lattice round-trip, negative zero, all entropy backends, chunked
  encoding and a NetCDF end-to-end cycle.

## 0.2.0

- Time-cadence detection (hourly / daily / monthly) with candidate lags.
- Per-variable automatic model selection.
- Per-block SHA-256 verification.
- Full NetCDF metadata reconstruction on decompress.
- Adaptive file chunking targeting ~96 MB per block.
