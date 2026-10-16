"""gridpack -- 格点数据（ERA5 等）无损压缩读写工具包。

严格无损：``decode(encode(x)) == x``，解码后与原始数据**逐字节相同**（含 -0.0、NaN、Inf）。

典型用法::

    import gridpack

    # 文件级：.nc -> .gpz，再还原
    stats = gridpack.compress_file(r"D:/data/era5.nc", r"D:/data/era5.nc.gpz")
    gridpack.verify_file(r"D:/data/era5.nc", r"D:/data/era5.nc.gpz")
    gridpack.decompress_file(r"D:/data/era5.nc.gpz", r"D:/data/era5_restored.nc")

    # 数组级：任意 numpy 数组
    blob = gridpack.encode_array(arr)          # arr -> bytes
    arr2 = gridpack.decode_array(blob, arr.shape, arr.dtype)

    # 指定熵编码后端
    blob = gridpack.encode_array(arr, mode="dt", entropy="zlib")

装了 xarray 之后还可以当后端用::

    import xarray as xr
    ds = xr.open_dataset("era5_10u.gpz", engine="gpz")
    ds.gpz.to_file("out.gpz", level=12)
"""

from . import codec, container, entropy, lattice, netcdf_io
from .codec import (CODEC_VERSION, MODES, candidate_modes, decode_array, decode_bytes,
                    detect_lattice_e, encode_array, encode_bytes)
from .entropy import METHODS as ENTROPY_METHODS
from .lattice import detect_lattice, verify_lattice
from .netcdf_io import compress_file, decompress_file, detect_cadence, verify_file

try:                                   # 注册 xarray 后端与 ds.gpz 访问器；未装 xarray 时跳过
    from . import xarray_backend                        # noqa: F401
    from .xarray_backend import open_dataset as open_xarray
    from .xarray_backend import to_gpz as from_xarray
    HAVE_XARRAY = True
except Exception:                                       # pragma: no cover
    xarray_backend = None
    open_xarray = None
    from_xarray = None
    HAVE_XARRAY = False

__version__ = "0.5.0"

__all__ = [
    "codec", "container", "entropy", "lattice", "netcdf_io",
    "encode_array", "decode_array", "encode_bytes", "decode_bytes",
    "compress_file", "decompress_file", "verify_file", "detect_cadence",
    "detect_lattice", "detect_lattice_e", "verify_lattice",
    "open_xarray", "from_xarray", "HAVE_XARRAY",
    "CODEC_VERSION", "MODES", "candidate_modes", "ENTROPY_METHODS", "__version__",
]
