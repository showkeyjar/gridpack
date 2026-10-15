# gridpack

**Lossless compression for gridded scientific data (ERA5 and friends).**

`gridpack` shrinks NetCDF/HDF5 gridded fields — `(time × lat × lon)` float arrays — without
changing a single bit. `decode(encode(x)) == x` holds byte-for-byte, including `-0.0`,
`NaN` and `±Inf`. It also exposes an array-level codec so other projects can use it as a
drop-in compression layer.

On real ERA5 10 m wind fields it reaches **10.4× vs the raw float bytes** while staying
bit-exact, and stores every block with a SHA-256 so silent corruption is impossible.

---

## Why it works

Most gridded products are not arbitrary floats. ERA5 fields, for example, sit on an exact
binary lattice: 10 m wind is quantised to `2**-4` (0.0625 m/s), total precipitation to
`2**-14`, 2 m temperature to `2**-4`. `gridpack` detects that lattice **exactly** and
rewrites the field as integers `k = v / 2**E`. Small integers compress far better than
32-bit float bit patterns.

Detection is not a heuristic. For a non-zero float32 value `v = sig · 2**(exp-150)` with a
24-bit significand `sig`, `v` is a multiple of `2**E` iff `tz(sig) + exp - 150 >= E`, so the
largest usable exponent is `E_max = min(tz(sig) + exp - 150)`. The code verifies the
rewrite is exact before committing to it, and falls back to an order-preserving bit map
otherwise.

| Stage | What it does | Why |
|---|---|---|
| 1. Representation | exact lattice detection → integers, else order-preserving bit map | shrinks the alphabet before any modelling |
| 2. Prediction | 1st-order differencing along time (`lag`), latitude and longitude (separable 3-D Lorenzo) | turns smooth fields into small residuals |
| 3. Zig-zag | signed residual → unsigned small integer | clears the high byte planes |
| 4. Byte-plane split | de-interleave bytes into planes | gives the entropy coder near-stationary input |
| 5. Entropy coding | `zstd` (default), `zlib`, `rc0`/`rc1` adaptive range coder, `huff` | last few percent |

`-0.0` cannot be represented as an integer, so its positions are stored separately as a
sparse bitmask — that is what keeps the lattice path bit-exact.

---

## Install

```bash
pip install gridpack                 # core: numpy + h5py (zstandard optional, recommended)
pip install "gridpack[xarray]"       # + xarray backend support
```

Or from source:

```bash
git clone https://github.com/showkeyjar/gridpack && cd gridpack
pip install -e .
```

Installing the package also registers the xarray backend entry point, so
`engine="gpz"` works immediately -- see [xarray](#xarray).

---

## Command line

```bash
python -m gridpack compress   in.nc  out.gpz [options]
python -m gridpack decompress in.gpz out.nc
python -m gridpack verify     in.nc  out.gpz      # byte-for-byte comparison
python -m gridpack info       out.gpz
python -m gridpack bench      in.nc --nt 48
```

Example:

```bash
python -m gridpack compress era5_10u.nc era5_10u.gpz --level 12 --entropy zstd --mode auto
python -m gridpack verify   era5_10u.nc era5_10u.gpz
# EXACT
```

---

## Parameters

### `compress`

| Parameter | Default | Meaning |
|---|---|---|
| `--level` | `12` | Entropy-coder level. zstd accepts 1–22, zlib 1–9. Level 12 is a good speed/size point; 19 is ~3 % smaller but ~18× slower. |
| `--entropy` | `zstd` | Backend: `zstd`, `zlib`, `rc0`, `rc1`, `huff`. `rc*`/`huff` are pure-Python research backends — best ratio, far slower; use them on small blocks only. |
| `--mode` | `auto` | Time model: `none` (byte planes only), `spatial` (lat+lon differencing), `dt` (time Δt + spatial), `lagN` (time ΔN + spatial), `auto` (tries the candidates per variable and keeps the smallest). |
| `--chunk-t` | `16` | Inner time chunk for the codec. `0` disables internal chunking. |
| `--file-chunk` | `0` | Time chunk used when slicing the file. `0` picks a size targeting ~96 MB per block so the time chain is not cut unnecessarily. |
| `--threads` | `4` | zstd worker threads. |
| `--quant-bits` | `0` | `0` = strict lossless. `>0` drops the low `quant-bits` of the ordered integer — **lossy**, opt-in only. |

`--mode auto` is per-variable: the codec probes the first block with every candidate and
locks in the winner for that dataset. Different variables routinely pick different models.

### Python API

```python
import gridpack

# file level
stats = gridpack.compress_file(src, dst, level=12, chunk_t=16, threads=4,
                               quant_bits=0, mode="auto", entropy="zstd",
                               file_chunk=None, verify=True)
gridpack.verify_file(src, dst)          # -> {"exact": True, "details": [...]}
gridpack.decompress_file(dst, out_nc)

# array level
blob = gridpack.encode_array(arr, level=12, chunk_t=16, threads=1,
                             quant_bits=0, mode="dt", entropy="zstd")
arr2 = gridpack.decode_array(blob, arr.shape, arr.dtype)

# what representation will be used?
gridpack.detect_lattice_e(arr)          # -> -4, or None if no lattice
```

`chunk_t=None` encodes the array as a single block. With `chunk_t=N` the array is split
along axis 0 and each chunk is stored independently, so `decode_array` can rebuild without
the whole payload in memory.

### xarray

Installing the package registers an xarray backend entry point, so a `.gpz` file opens
like any other dataset:

```python
import xarray as xr
import gridpack                                  # registers backend + accessor

ds = xr.open_dataset("era5_10u.gpz", engine="gpz")
ds = xr.open_dataset("era5_10u.gpz", engine="gpz",
                     chunks={"time0": 24})       # dask-backed, lazy chunks
```

`engine="gpz"` is also selected automatically by `xr.open_dataset("...gpz")`, because the
backend's `guess_can_open` matches the `.gpz` suffix. Dimensions and coordinates are rebuilt
from the dimension-scale links stored in the container, and every value is bit-identical to
the source `.nc`.

xarray has no generic *write* backend interface, so writing goes through the accessor:

```python
ds.gpz.to_file("out.gpz", level=12, threads=4)    # -> compress_file() stats dict
ds.gpz.info("out.gpz")                            # -> ratio, per-variable model, ...

gridpack.from_xarray(ds, "out.gpz")               # same, without the accessor
gridpack.open_xarray("out.gpz")                   # same as xr.open_dataset(..., engine="gpz")
```

`to_file` writes a temporary netCDF (needs `h5netcdf` or `netCDF4`) and compresses that, then
removes it. netCDF-internal attributes (`_NCProperties`, `CLASS`, `_Netcdf4Coordinates`, ...)
are stripped automatically -- if a backend rejects any further reserved name, it is dropped
and the write is retried.

Verified on the bundled ERA5 subset (120 time steps, 4 variables): all four variables
round-trip bit-exactly through `engine="gpz"`, and a `ds.isel(time0=slice(0, 6)).gpz.to_file()`
write-back re-reads bit-exactly at **10.30x**.

```bash
python tests/test_xarray.py
```

---

## Benchmarks

Real ERA5, 10 m eastward wind (`eastward_wind_at_10_metres`, `float32`, 120 time steps,
47.53 MB raw). Full reproducible output in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md);
regenerate with `python scripts/bench_era5.py <file.nc>`.

### Representation × time model (entropy = zstd level 12)

| Representation | Time model | Compressed | Ratio |
|---|---|---|---|
| lattice E=-4 | none | 8.35 MB | 5.69× |
| lattice E=-4 | spatial | 5.05 MB | 9.40× |
| **lattice E=-4** | **dt** | **4.59 MB** | **10.36×** |
| order map | none | 9.05 MB | 5.25× |
| order map | spatial | 10.83 MB | 4.39× |
| order map | dt | 11.07 MB | 4.29× |

Two things worth noting. The lattice representation roughly **doubles** the ratio
(10.36× vs 5.25×). And with the plain order map, adding prediction makes things *worse*
(5.25× → 4.29×): the order-preserving bit map has a large step at the zero crossing, so a
field that changes sign produces enormous residuals. Integer-lattice residuals do not have
that problem.

### Entropy backend (first 200 000 bytes of the same byte-plane buffer)

| Backend | Compressed | vs raw | Time |
|---|---|---|---|
| zstd | 74 363 B | 2.69× | 0.03 s |
| zlib | 75 024 B | 2.67× | 0.22 s |
| rc0 | 68 216 B | 2.93× | 0.45 s |
| **rc1** | **65 015 B** | **3.08×** | 0.48 s |
| huff | 73 150 B | 2.73× | 0.03 s |

The adaptive context range coder (`rc1`, context = previous byte × bit index) beats zstd by
**12.6 %** on this data, and `rc0` by 4.7 %. That is a real gain but it costs ~15× the time
in pure Python, so `zstd` stays the production default and `rc*` are there for offline
re-encoding or archival runs where CPU time is cheap.

### Whole file

| | |
|---|---|
| source `.nc` | 475.28 MB |
| `.gpz` | 45.83 MB |
| ratio vs raw float bytes | **10.37×** |
| wall clock | 47.6 s (≈10 MB/s of raw data) |
| byte-for-byte verify | **pass** |
| chosen models | wind → lattice E=-4 / `dt`; `lat` → E=-2 / `spatial`; `lon` → E=-2 / `spatial`; `time0` → E=4 / `spatial` |

Ratio depends heavily on how the source was written. Fields already quantised (most modern
ERA5 monthly files) land at ~10×; unquantised, full-precision months land at ~4×; a `.nc`
that is itself zlib-compressed by netCDF will show a smaller gain versus that `.nc`, so
compare against the raw float bytes to get a like-for-like number.

---

## Correctness

`tests/test_roundtrip.py` covers every path and asserts byte equality, not approximate
equality:

```bash
python tests/test_roundtrip.py                 # unit + array level
GRIDPACK_TEST_NC=file.nc python tests/test_roundtrip.py   # + NetCDF end-to-end
```

It checks the order map on `±0.0 / ±Inf / NaN / subnormals`, lattice detection and
round-trip for float32 and float64, negative-zero preservation, all five entropy backends
(including single-symbol and empty inputs), all time models on random / lattice / constant /
NaN-bearing / integer / 1-D arrays, chunked encoding, and a full NetCDF compress → verify →
decompress cycle.

Container guarantees:

- every block carries a SHA-256 of its raw bytes; a mismatch raises instead of returning
  wrong data,
- the restored `.nc` keeps variable names, shapes, dtypes, dimension scales, chunking,
  compression settings, `_FillValue` and all attributes,
- peak memory is bounded by the file chunk size, not by the file size.

---

## Container format

`.gpz` (`MAGIC = b"GPZ1"`): JSON metadata header + a payload index + the compressed blocks.
Metadata records the source name/size, codec version, and per-dataset shape, dtype,
chunking, attributes, dimension links, chosen mode, and the detected lattice exponent.

Each block starts with a tag byte (`0` raw, `1` predicted, `2` byte-planes only,
`3` lattice + byte-planes, `4` lattice + predicted), an entropy-backend id, and — for the
lattice tags — the exponent and the length of the negative-zero bitmap.

---

## Limitations

- The lattice path only engages when **every** value shares one exponent and the resulting
  integers fit comfortably in int32; otherwise the codec falls back to the order map.
- `.gpz` needs this library to read. It is not a NetCDF file. If you need `.nc`
  compatibility, decompress back.
- `rc0`/`rc1`/`huff` are pure Python and unsuitable for large files.
- Throughput is ~10 MB/s of raw data per process. Parallelise across files.
- Only `quant_bits=0` is lossless; anything else deliberately discards information.

---

## 中文说明

`gridpack` 是面向 ERA5 这类**格点科学数据**的严格无损压缩与读写工具包：解码后与原始
数据**逐字节相同**（含 `-0.0`、`NaN`、`±Inf`），实测在真实 ERA5 10m 风场上达到
**10.37 倍**压缩（相对原始 float 字节），并支持按需读取。

**核心思路**：ERA5 各要素的值并不是任意浮点数，而是**精确落在同一个二进制格点 `2**E`
上**（10m 风量化到 `2**-4`，总降水 `2**-14`，2m 气温 `2**-4`）。工具会**精确判定**这个
格点并把浮点无损改写成整数 `k = v / 2**E`，再做时间/空间一阶差分、zig-zag、字节平面重排
与熵编码。负零单独用稀疏位图记录，保证位级无损。

**关键参数**：`--mode`（`none`/`spatial`/`dt`/`lagN`/`auto`，按变量自动选）、
`--entropy`（`zstd`/`zlib`/`rc0`/`rc1`/`huff`）、`--level`（zstd 1–22，默认 12）、
`--chunk-t`（内部时间分块，默认 16）、`--file-chunk`（文件级分块，0 = 自适应约 96 MB）、
`--threads`、`--quant-bits`（0 = 严格无损，>0 为有容差模式）。

**实测**（真实 ERA5 10m 风场，120 时次，47.53 MB 原始）：格点整数 + `dt` 达 **10.36×**，
而保序映射最好只有 5.25×——格点表示带来近一倍收益；熵编码上 `rc1` 比 `zstd` 再小
**12.6%**，但纯 Python 慢约 15 倍，故生产默认仍用 `zstd`。整文件 475.28 MB → 45.83 MB，
**10.37×**，耗时 47.6 s，逐字节校验通过。

**xarray 支持**：`pip install gridpack` 后 xarray 后端即注册完成，`.gpz` 可直接
`xr.open_dataset("x.gpz", engine="gpz")` 打开（`.gpz` 后缀也会被自动识别）；维度与坐标
从容器里记录的维标关联还原，数值与原始 `.nc` 逐位一致。写回用
`ds.gpz.to_file("out.gpz", level=12, threads=4)`，也可用 `gridpack.from_xarray(ds, path)`；
查看某个 `.gpz` 的压缩率与逐变量选中的模型用 `ds.gpz.info(path)`。实测 120 时次 ERA5 子集
的 4 个变量经 `engine="gpz"` 读出全部逐位通过，写回再读回同样逐位通过（10.30×）。

---

## License

MIT — see [LICENSE](LICENSE).
