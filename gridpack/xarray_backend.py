"""gridpack.xarray_backend -- xarray 后端插件。

安装后（entry point ``xarray.backends`` 已注册）可以直接把 ``.gpz`` 当普通数据集打开::

    import xarray as xr
    import gridpack                      # 注册后端与访问器

    ds = xr.open_dataset("era5_10u.gpz", engine="gpz")
    print(ds)                            # 维度、坐标、属性都还原

写回（xarray 官方没有通用的写入后端接口，所以用访问器）::

    ds.gpz.to_file("era5_10u.gpz", level=12, threads=4)

也可以绕过 xarray 直接用顶层函数::

    gridpack.open_xarray("x.gpz")
    gridpack.from_xarray(ds, "x.gpz")

维度名解析顺序：

1. 容器里记录的维标关联（``dim_links``，来自 HDF5 dimension scale）；
2. 该轴长度与某个维标变量长度**唯一**匹配时，用那个维标的名字；
3. 仍无法确定时退回 ``dimN``（N 为长度，保证同名同长、不冲突）。

说明：

* 读路径完全走 ``container``/``codec``，逐块解码后拼成 numpy 数组再交给 xarray，
  因此读取后的 ``.gpz`` 与原始 ``.nc`` 在数值上逐位一致。
* 写路径先把 Dataset 落到临时 netCDF（需要 ``h5netcdf`` 或 ``netCDF4``），
  再用 ``netcdf_io.compress_file`` 压缩；临时文件用完即删。
* 时间坐标等 CF 解码交给 ``xr.decode_cf`` 处理，失败则退回未解码的 Dataset。
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np

from . import codec, container, netcdf_io

try:
    import xarray as xr
    from xarray.backends import BackendEntrypoint
    HAVE_XARRAY = True
except Exception:                                            # pragma: no cover
    xr = None
    BackendEntrypoint = object
    HAVE_XARRAY = False

# netCDF/HDF5 内部属性：从 .nc 读出来时会带上，写回时必须剔除
_KNOWN_RESERVED = {"_NCProperties", "_SuperblockVersion", "_IsNetcdf4", "_Format",
                   "CLASS", "DIMENSION_LIST", "NAME", "REFERENCE_LIST",
                   "_Netcdf4Coordinates", "_Netcdf4Dimid", "_nc3_strict"}


def _is_reserved(k, extra=()):
    return (k in _KNOWN_RESERVED or k in extra or k.startswith("_NC")
            or k.startswith("_Netcdf") or k.startswith("_nc"))


def _strip_attrs(ds, extra=()):
    """浅拷贝并剔除内部属性（数据集级 + 变量级 + encoding）。

    Dataset.copy(deep=False) 会给每个变量复制出新的 attrs 字典，因此原地删除不会
    影响调用方传入的 Dataset。
    """
    ds = ds.copy(deep=False)
    for k in [k for k in ds.attrs if _is_reserved(k, extra)]:
        ds.attrs.pop(k, None)
    for var in ds.variables.values():
        for k in [k for k in var.attrs if _is_reserved(k, extra)]:
            var.attrs.pop(k, None)
        enc = getattr(var, "encoding", None)
        if enc:
            for k in [k for k in enc if _is_reserved(k, extra)]:
                enc.pop(k, None)
    return ds


def _write_nc(ds, nc):
    """写临时 netCDF；若 h5netcdf 因保留属性名拒绝，则摘掉该属性后重试。

    比硬编码黑名单更稳：不同 netCDF 库写出的内部属性名不尽相同，这里按后端实际
    报出的名字逐个剔除，最多重试 40 轮。
    """
    extra = set()
    for _ in range(40):
        clean = _strip_attrs(ds, extra)
        try:
            try:
                clean.to_netcdf(nc, engine="h5netcdf")
            except (ImportError, ValueError):
                clean.to_netcdf(nc)                          # 退回 xarray 默认引擎
            return
        except AttributeError as ex:
            m = re.search(r"reserved name '([^']+)'", str(ex))
            if not m or m.group(1) in extra:
                raise
            extra.add(m.group(1))
    raise RuntimeError("无法写出临时 netCDF：需剔除的内部属性过多")


__all__ = ["GpzBackendEntrypoint", "open_dataset", "to_gpz", "GpzAccessor",
           "HAVE_XARRAY", "available_engines"]


def _read_arrays(path):
    """返回 (meta, {path: (entry, ndarray)})。"""
    out = {}
    with container.open_reader(path) as rd:
        meta = rd.meta
        idx = 0
        for e in meta["entries"]:
            dt = np.dtype(e["dtype"])
            shape = tuple(e["shape"])
            parts = []
            for p in e["parts"]:
                pshape = netcdf_io._part_shape(e, p["n"])
                raw = codec.decode_bytes(rd.payload(idx), pshape, dt)
                idx += 1
                parts.append(np.frombuffer(raw, dtype=dt).reshape(pshape))
            arr = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            out[e["path"]] = (e, np.ascontiguousarray(arr).reshape(shape))
    return meta, out


def _resolve_dims(arrays):
    """给每个变量定维度名，返回 {path: (dims, ...)}。"""
    scales = [(p.lstrip("/"), a.shape[0])
              for p, (e, a) in arrays.items() if e.get("is_scale") and a.ndim == 1]
    dims_of = {}
    for p, (e, arr) in arrays.items():
        dims = [None] * arr.ndim
        for link in (e.get("dim_links") or []):
            try:
                axis, spath = int(link[0]), str(link[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= axis < arr.ndim:
                dims[axis] = spath.lstrip("/")
        for i, ln in enumerate(arr.shape):
            if dims[i] is None:
                cand = [n for n, L in scales if L == ln]
                if len(cand) == 1:
                    dims[i] = cand[0]          # 长度唯一匹配维标
                else:
                    dims[i] = "dim%d" % ln     # 同名同长，不会互相冲突
        dims_of[p] = tuple(dims)
    return dims_of


def _build_dataset(meta, arrays):
    dims_of = _resolve_dims(arrays)
    coords = {}
    data_vars = {}
    for p, (e, arr) in arrays.items():
        name = p.lstrip("/")
        dims = dims_of[p]
        attrs = {k: container.decode_attr(v) for k, v in (e.get("attrs") or {}).items()}
        if e.get("fillvalue") is not None:
            attrs["_FillValue"] = container.decode_attr(e["fillvalue"])
        var = xr.Variable(dims, arr, attrs=attrs)
        if e.get("is_scale") and arr.ndim == 1 and dims[0] == name:
            coords[name] = var
        else:
            data_vars[name] = var
    root_attrs = {k: container.decode_attr(v)
                  for k, v in (meta.get("root_attrs") or {}).items()}
    return xr.Dataset(data_vars, coords=coords, attrs=root_attrs)


def open_dataset(filename_or_obj, *, drop_variables=None, decode_times=True,
                 decode_cf=True, chunks=None, **kwargs):
    """把 .gpz 读成 xarray.Dataset（严格还原原始数值）。"""
    if not HAVE_XARRAY:                                      # pragma: no cover
        raise ImportError("需要 xarray：pip install xarray")
    path = os.fspath(filename_or_obj)
    meta, arrays = _read_arrays(path)
    ds = _build_dataset(meta, arrays)
    if decode_cf:
        try:
            ds = xr.decode_cf(ds, decode_times=decode_times)
        except Exception:
            pass                                             # 属性不标准时保留原样
    if drop_variables is not None:
        if isinstance(drop_variables, str):
            drop_variables = [drop_variables]
        ds = ds.drop_vars(drop_variables)
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds


def to_gpz(ds, path, *, level=12, chunk_t=None, threads=4, quant_bits=0,
           mode="auto", entropy="zstd", file_chunk=None, verify=True, tmpdir=None):
    """把 xarray.Dataset 无损压缩写成 .gpz。返回 compress_file 的统计字典。"""
    tmp = tempfile.mkdtemp(prefix="gridpack_xr_", dir=tmpdir)
    nc = os.path.join(tmp, "tmp.nc")
    try:
        _write_nc(ds, nc)
        return netcdf_io.compress_file(nc, path, level=level, chunk_t=chunk_t,
                                       threads=threads, quant_bits=quant_bits,
                                       mode=mode, entropy=entropy,
                                       file_chunk=file_chunk, verify=verify)
    finally:
        try:
            if os.path.exists(nc):
                os.remove(nc)
            os.rmdir(tmp)
        except OSError:
            pass


class GpzBackendEntrypoint(BackendEntrypoint):
    """xarray ``engine="gpz"`` 后端。"""

    description = "gridpack 无损压缩容器（.gpz）读取后端"
    url = "https://github.com/showkeyjar/gridpack"
    open_dataset_parameters = ("filename_or_obj", "drop_variables", "decode_times",
                               "decode_cf", "chunks")

    def open_dataset(self, filename_or_obj, *, drop_variables=None, decode_times=True,
                     decode_cf=True, chunks=None, **kwargs):
        return open_dataset(filename_or_obj, drop_variables=drop_variables,
                            decode_times=decode_times, decode_cf=decode_cf, chunks=chunks)

    def guess_can_open(self, filename_or_obj):
        try:
            return os.fspath(filename_or_obj).lower().endswith(".gpz")
        except TypeError:
            return False


if HAVE_XARRAY:

    @xr.register_dataset_accessor("gpz")
    class GpzAccessor:
        """``ds.gpz.to_file(path)`` / ``ds.gpz.info(path)``。"""

        def __init__(self, xarray_obj):
            self._obj = xarray_obj

        def to_file(self, path, **kwargs):
            return to_gpz(self._obj, path, **kwargs)

        def info(self, path):
            m = container.read_meta(path)
            total_raw = sum(p["raw_nbytes"] for e in m["entries"] for p in e["parts"])
            total_pay = sum(p["length"] for p in m["payload_index"])
            return {"source": m.get("source_name"), "codec_version": m.get("codec_version"),
                    "entropy": m.get("entropy", "zstd"), "datasets": len(m["entries"]),
                    "raw_bytes": total_raw, "gpz_bytes": total_pay,
                    "ratio": total_raw / max(total_pay, 1),
                    "per_dataset": [[e["path"], e.get("mode"), e.get("lattice_E")]
                                    for e in m["entries"]]}
else:                                                        # pragma: no cover
    class GpzAccessor:                                       # 占位，避免 import 失败
        pass


def available_engines():
    """列出 xarray 当前可用的后端，确认 gpz 是否注册成功。"""
    if not HAVE_XARRAY:                                      # pragma: no cover
        return []
    try:
        import xarray.backends as xb
        return sorted(xb.list_engines())
    except Exception:                                        # pragma: no cover
        return []
