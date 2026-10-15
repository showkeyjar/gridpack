# Changelog

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
