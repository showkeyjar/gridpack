"""xarray 后端测试。

覆盖：
1. entry point 注册（``xr.open_dataset(path, engine="gpz")`` 可用）；
2. ``engine="gpz"`` 读出的每个变量与原始 ``.nc`` **逐位一致**；
3. 维度/坐标还原；
4. ``ds.gpz.to_file()`` 写出后再读回**逐位一致**；
5. ``guess_can_open`` 只认 ``.gpz``。

不依赖外部数据，全部现场构造。
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gridpack                                                    # noqa: E402

try:
    import xarray as xr
    from gridpack.xarray_backend import HAVE_XARRAY, available_engines
except ImportError:                                                # pragma: no cover
    HAVE_XARRAY = False

FAILED = []


def check(name, cond, extra=""):
    print("%-46s %s %s" % (name, "PASS" if cond else "FAIL", extra))
    if not cond:
        FAILED.append(name)


def build(path):
    nt, nlat, nlon = 24, 41, 61
    t = np.arange(nt, dtype="float64")
    lat = np.linspace(-10.0, 10.0, nlat).astype("float32")
    lon = np.linspace(0.0, 20.0, nlon).astype("float32")
    # 平滑场，量化到 2**-4 格点：既真实（可压缩）又精确落在格点上
    base = (np.sin(t[:, None, None] / 3.0)
            + np.cos(lat[None, :, None] / 2.0)
            + np.sin(lon[None, None, :] / 4.0))
    k = np.rint(base * 64.0).astype("float32")
    wind = (k * np.float32(2.0 ** -4)).astype("float32")
    ds = xr.Dataset(
        {"eastward_wind_at_10_metres": (("time0", "lat", "lon"), wind)},
        coords={"time0": (("time0",), t), "lat": (("lat",), lat),
                "lon": (("lon",), lon)},
        attrs={"title": "synthetic", "Conventions": "CF-1.6"},
    )
    ds.to_netcdf(path, engine="h5netcdf")
    return ds


def main():
    if not HAVE_XARRAY:
        print("xarray 未安装，跳过")
        return 0

    tmp = tempfile.mkdtemp(prefix="gridpack_xrtest_")
    try:
        src = os.path.join(tmp, "src.nc")
        gpz = os.path.join(tmp, "src.gpz")
        gpz2 = os.path.join(tmp, "small.gpz")
        build(src)

        engines = available_engines()
        check("xarray backend 'gpz' registered", "gpz" in engines, str(engines))

        from gridpack.xarray_backend import GpzBackendEntrypoint
        ep = GpzBackendEntrypoint()
        check("guess_can_open('.gpz')", ep.guess_can_open(gpz) is True)
        check("guess_can_open('.nc') == False", ep.guess_can_open(src) is False)

        st = gridpack.compress_file(src, gpz, level=12, threads=2, verify=True)
        check("compress_file", st["ratio_vs_raw"] > 1.0,
              "%.2fx" % st["ratio_vs_raw"])

        ref = xr.open_dataset(src, engine="h5netcdf", decode_cf=False)
        got = xr.open_dataset(gpz, engine="gpz", decode_cf=False)
        ok = True
        for v in ref.variables:
            a, b = np.asarray(ref[v].values), np.asarray(got[v].values)
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                ok = False
                print("    MISMATCH", v, a.dtype, b.dtype, a.shape, b.shape)
        check("engine='gpz' read is bit-exact (%d vars)" % len(ref.variables), ok)
        check("dataset attrs preserved", got.attrs.get("title") == "synthetic")

        dec = xr.open_dataset(gpz, engine="gpz")                   # decode_cf=True
        check("decode_cf=True keeps coords",
              set(dec.coords) >= {"lat", "lon", "time0"} and
              "eastward_wind_at_10_metres" in dec.data_vars,
              str(dict(dec.sizes)))

        try:
            auto = xr.open_dataset(gpz, decode_cf=False)           # 不指定 engine
            check("auto engine by suffix", "time0" in auto.sizes)
        except Exception as ex:
            check("auto engine by suffix", False, "%s: %s" % (type(ex).__name__, ex))

        small = got.isel(time0=slice(0, 3))
        info = small.gpz.to_file(gpz2, level=12, threads=2)
        back = xr.open_dataset(gpz2, engine="gpz", decode_cf=False)
        ok2 = True
        for v in small.variables:
            a, b = np.asarray(small[v].values), np.asarray(back[v].values)
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                ok2 = False
                print("    MISMATCH", v)
        check("ds.gpz.to_file write-back bit-exact", ok2,
              "%.2fx" % info["ratio_vs_raw"])

        meta = small.gpz.info(gpz2)
        check("ds.gpz.info", meta["datasets"] == 4 and meta["ratio"] > 1.0,
              "ratio %.2f, per_dataset=%s" % (meta["ratio"], meta["per_dataset"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print("FAILED:", FAILED)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
