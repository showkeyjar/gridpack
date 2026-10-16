# Changelog

## 0.5.3

First release built and published from CI. `.github/workflows/publish.yml` runs on a
`v*` tag: it checks the tag against `pyproject.toml`, builds the sdist and wheel,
`twine check`s them, runs the test suite, and publishes through PyPI Trusted
Publishing (OIDC, no stored API token).

No library change from 0.5.2. This release exists to validate the pipeline, and to
ship a CHANGELOG whose top heading matches the published version -- the 0.5.2 sdist
still carried a `## 0.5.0` heading, because the heading was renamed after that
distribution had already been built.

## 0.5.2

First published release. 0.5.0 and 0.5.1 were built but never published -- see
the note at the end of this section.

Performance release. No format change and no API change: `CODEC_VERSION` is untouched, so
`.gpz` files written by 0.4.0 still read.

- **`detect_lattice` 2.4x faster; whole-file compression 1.56x faster.** Lattice detection
  dominated compression time and, within it, `_trailing_zeros` dominated detection: it ran
  a shift-and-mask loop over the entire field for every doubling step. The <=32-bit path is
  now a two-level 16-bit lookup table. Measured on a 483.1 MB ERA5 2 m temperature file
  (3 089.8 MB of raw `float32`): `_block_exp` 13.96 s -> 4.96 s (2.81x), `detect_lattice`
  19.6 s -> 8.2 s, `compress_file` end to end 148.6 s -> 95.3 s. `detect_lattice` fell from
  13 % to 8.6 % of the wall clock.
- The output is bit-identical to 0.4.0 — same 296.1 MB container, same 1.631x ratio,
  `verify_file` byte-exact — so this is a pure speed-up.
- The lookup table was checked against the original implementation exhaustively over
  `1 .. 2**20` and on 2 x 10**5 random 32-bit values; the detected exponent on real ERA5
  data is unchanged.
- The float64 path (`nbits = 53`) keeps the original implementation.
- **Packaging:** the wheel and sdist are published to PyPI. The build is a
  hand-assembled zip/tarball (the sandbox cannot run `bdist_wheel`), so the
  dist-info is copied from the 0.4.0 wheel with `Version` rewritten and the
  `RECORD` recomputed.

## 0.4.0

xarray backend + pip packaging.

- New `gridpack/xarray_backend.py`: `GpzBackendEntrypoint`, the `ds.gpz` accessor, and the
  `open_xarray` / `from_xarray` helpers.
- `pyproject.toml` registers `[project.entry-points."xarray.backends"]`, so
  `xr.open_dataset(path, engine="gpz")` works right after `pip install gridpack`;
  `guess_can_open` matches the `.gpz` suffix so the engine can also be auto-selected.
- Dimension names are resolved from the container's dimension-scale links; when a link is
  missing, an axis length that uniquely matches a dimension-scale variable is used, and
  `dimN` is the last resort (no more `dim_0` collisions between `lat` and `lon`).
- The write path strips netCDF-internal attributes (`_NCProperties`, `CLASS`,
  `_Netcdf4Coordinates`, ...); any further name the backend rejects is dropped and the
  write retried.
- New optional extras `xarray` and `all`, plus `[project.urls]`.
- New `tests/test_xarray.py` (entry point, bit-exact read, coordinate restoration,
  accessor write-back, suffix auto-detection).
- Measured: the 120-step ERA5 subset (4 variables) reads back bit-exactly through
  `engine="gpz"`, and `ds.gpz.to_file()` re-reads bit-exactly at 10.30x.

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
