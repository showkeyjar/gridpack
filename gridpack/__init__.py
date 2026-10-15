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
"""

from . import codec, container, entropy, lattice, netcdf_io
from .codec import (CODEC_VERSION, MODES, candidate_modes, decode_array, decode_bytes,
                    detect_lattice_e, encode_array, encode_bytes)
from .entropy import METHODS as ENTROPY_METHODS
from .lattice import detect_lattice
from .netcdf_io import compress_file, decompress_file, detect_cadence, verify_file

__version__ = "0.3.0"

__all__ = [
    "codec", "container", "entropy", "lattice", "netcdf_io",
    "encode_array", "decode_array", "encode_bytes", "decode_bytes",
    "compress_file", "decompress_file", "verify_file", "detect_cadence",
    "detect_lattice", "detect_lattice_e",
    "CODEC_VERSION", "MODES", "candidate_modes", "ENTROPY_METHODS", "__version__",
]
