"""gridpack.netcdf_io -- NetCDF4/HDF5 文件级压缩与还原。

- 每个数据集（含维标 dimension scale）逐个无损压缩进 .gpz 容器，元数据一并保存。
- 从时间坐标自动识别节律（逐小时/逐天/逐月），据此给出候选时间滞后。
- mode='auto' 时先在小样本上试各种时间模型，选最小的那个再整份编码。
- 分块长度按「每块约 96 MB」自适应，尽量不切断时间链。
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from . import codec, container

_SKIP_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST"}
_SCALE_OWN_ATTRS = {"CLASS", "NAME"}
TARGET_BLOCK_BYTES = 96 << 20
MIN_CHUNK = 8

# 以「小时」表示的候选滞后
LAG_HOURS = (1, 24, 168, 720, 8760)


def _plain_attrs(obj) -> dict:
    out = {}
    for k, v in obj.attrs.items():
        if k in _SKIP_ATTRS:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind == "O":
            continue
        out[k] = container.encode_attr(v)
    return out


def _scale_links(d) -> list:
    links = []
    for i in range(d.ndim):
        try:
            for k in d.dims[i].keys():
                links.append([i, d.dims[i][k].name])
        except Exception:
            pass
    return links


def _is_scale(d) -> bool:
    c = d.attrs.get("CLASS")
    if isinstance(c, bytes):
        c = c.decode("utf-8", "ignore")
    return c == "DIMENSION_SCALE"


def _walk(f) -> list:
    names = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(obj.name)

    f.visititems(visit)
    return names


def _part_shape(entry, n):
    return (n,) + tuple(entry["shape"][1:])


def detect_cadence(f):
    """返回 (step_hours, 候选滞后步数列表, 说明)。"""
    for name in _walk(f):
        d = f[name]
        if d.ndim != 1 or d.shape[0] < 3:
            continue
        units = d.attrs.get("units")
        if isinstance(units, bytes):
            units = units.decode("utf-8", "ignore")
        nm = d.name.lower()
        if not (isinstance(units, str) and "since" in units.lower()) and "time" not in nm:
            continue
        v = np.asarray(d[()], dtype="float64")
        if v.size < 3:
            continue
        step = float(np.median(np.diff(v)))
        if not np.isfinite(step) or step <= 0:
            continue
        unit = (units or "").lower()
        if "second" in unit:
            step_h = step / 3600.0
        elif "hour" in unit:
            step_h = step
        elif "day" in unit:
            step_h = step * 24.0
        else:
            step_h = step / 3600.0
        if step_h <= 0:
            continue
        lags = []
        for h in LAG_HOURS:
            L = int(round(h / step_h))
            if L >= 2 and L < d.shape[0]:
                lags.append(L)
        seen = []
        for L in lags:
            if L not in seen:
                seen.append(L)
        note = "%s step=%.4gh (%.4gh/step)" % (name, step_h, step_h)
        return step_h, seen, note
    return None, [], "未找到时间坐标"


def _adaptive_chunk(ny, nx, itemsize):
    per = max(ny * nx * itemsize, 1)
    return max(MIN_CHUNK, int(TARGET_BLOCK_BYTES // per))


def compress_file(src: str, dst: str, level: int = 12, chunk_t: int | None = None,
                  threads: int = 1, quant_bits: int = 0, mode: str = "auto",
                  entropy: str = "zstd",
                  file_chunk: int | None = None, verify: bool = True) -> dict:
    """把 .nc 压缩为 .gpz。返回统计信息 dict。"""
    entries = []
    payloads = []
    src_size = os.path.getsize(src)
    chosen = []

    with h5py.File(src, "r") as f:
        root_attrs = _plain_attrs(f)
        step_h, lags, note = detect_cadence(f)
        modes = codec.candidate_modes(lags) if mode == "auto" else [mode]

        for name in _walk(f):
            d = f[name]
            fc = file_chunk
            if fc is None:
                fc = _adaptive_chunk(d.shape[-2], d.shape[-1], d.dtype.itemsize) if d.ndim >= 2 \
                    else max(d.shape[0], 1)
            if d.ndim == 0:
                slices = [(0, 1)]
            else:
                step = max(fc if d.shape[0] > fc else d.shape[0], 1)
                slices = [(s, min(s + step, d.shape[0])) for s in range(0, d.shape[0], step)]

            # --- 选时间模型：在首个分块上试所有候选 ---
            s0, e0 = slices[0]
            probe = np.ascontiguousarray(d[()] if d.ndim == 0 else d[s0:e0])
            if len(modes) == 1:
                use_mode = modes[0]
            else:
                sizes = {}
                for mm in modes:
                    try:
                        sizes[mm] = len(codec.encode_bytes(probe.tobytes(), probe.shape,
                                                           probe.dtype, level=level, chunk_t=chunk_t,
                                                           threads=threads, quant_bits=quant_bits,
                                                           mode=mm, entropy=entropy))
                    except Exception:
                        pass
                use_mode = min(sizes, key=sizes.get) if sizes else "dt"
            lat_e = codec.detect_lattice_e(probe) if quant_bits == 0 else None
            chosen.append((name, use_mode, lat_e))

            parts = []
            for s, e in slices:
                arr = np.ascontiguousarray(d[()] if d.ndim == 0 else d[s:e])
                raw = arr.tobytes()
                blob = codec.encode_bytes(raw, arr.shape, arr.dtype, level=level,
                                          chunk_t=chunk_t, threads=threads,
                                          quant_bits=quant_bits, mode=use_mode,
                                          entropy=entropy)
                if verify and quant_bits == 0:
                    if codec.decode_bytes(blob, arr.shape, arr.dtype) != raw:
                        raise RuntimeError("round-trip mismatch on %s[%d:%d]" % (name, s, e))
                parts.append({"n": int(arr.shape[0]) if arr.ndim else 0,
                              "raw_nbytes": len(raw),
                              "sha256": container.sha256_bytes(raw)})
                payloads.append(blob)
            entries.append({
                "path": name,
                "shape": list(d.shape),
                "dtype": d.dtype.str,
                "chunks": list(d.chunks) if d.chunks else None,
                "compression": d.compression,
                "compression_opts": (list(d.compression_opts)
                                     if isinstance(d.compression_opts, tuple) else d.compression_opts),
                "shuffle": bool(d.shuffle),
                "fletcher32": bool(d.fletcher32),
                "fillvalue": None if d.fillvalue is None else container.encode_attr(d.fillvalue),
                "is_scale": _is_scale(d),
                "scale_name": (d.attrs.get("NAME").decode("utf-8", "ignore")
                               if isinstance(d.attrs.get("NAME"), bytes) else d.attrs.get("NAME")),
                "dim_links": _scale_links(d),
                "attrs": _plain_attrs(d),
                "mode": use_mode,
                "lattice_E": lat_e,
                "file_chunk": fc,
                "parts": parts,
            })

    meta = {
        "version": container.VERSION,
        "codec_version": codec.CODEC_VERSION,
        "source_name": os.path.basename(src),
        "source_bytes": src_size,
        "level": level, "chunk_t": chunk_t, "threads": threads,
        "quant_bits": quant_bits, "mode": mode, "entropy": entropy,
        "cadence_hours": step_h, "candidate_lags": lags, "cadence_note": note,
        "root_attrs": root_attrs,
        "entries": entries,
    }
    container.pack(dst, meta, payloads)

    raw_total = sum(p["raw_nbytes"] for e in entries for p in e["parts"])
    gpz_size = os.path.getsize(dst)
    return {"raw_bytes": raw_total, "src_bytes": src_size, "gpz_bytes": gpz_size,
            "ratio_vs_raw": raw_total / gpz_size, "vs_src": src_size / gpz_size,
            "entries": len(entries), "payloads": len(payloads),
            "cadence": note, "chosen": chosen}


def decompress_file(src: str, dst: str) -> dict:
    """把 .gpz 还原为 .nc（按原结构重建）。"""
    with container.open_reader(src) as rd:
        meta = rd.meta
        idx = 0
        with h5py.File(dst, "w") as g:
            for k, v in meta.get("root_attrs", {}).items():
                g.attrs[k] = container.decode_attr(v)
            created = {}
            for e in meta["entries"]:
                kw = {}
                if e.get("chunks"):
                    kw["chunks"] = tuple(e["chunks"])
                if e.get("compression"):
                    kw["compression"] = e["compression"]
                    if e.get("compression_opts") is not None:
                        kw["compression_opts"] = e["compression_opts"]
                    kw["shuffle"] = e.get("shuffle", False)
                if e.get("fletcher32"):
                    kw["fletcher32"] = True
                dt = np.dtype(e["dtype"])
                ds = g.create_dataset(e["path"], shape=tuple(e["shape"]), dtype=dt, **kw)
                pos = 0
                for p in e["parts"]:
                    raw = codec.decode_bytes(rd.payload(idx), _part_shape(e, p["n"]), dt)
                    idx += 1
                    if container.sha256_bytes(raw) != p["sha256"]:
                        raise RuntimeError("checksum mismatch on %s" % e["path"])
                    if ds.ndim == 0:
                        ds[()] = np.frombuffer(raw, dtype=dt)
                    else:
                        ds[pos:pos + p["n"]] = np.frombuffer(raw, dtype=dt).reshape(
                            _part_shape(e, p["n"]))
                        pos += p["n"]
                if e.get("fillvalue") is not None:
                    ds.attrs["_FillValue"] = container.decode_attr(e["fillvalue"])
                if e.get("is_scale"):
                    ds.make_scale(e.get("scale_name") or e["path"].lstrip("/"))
                created[e["path"]] = ds

            for e in meta["entries"]:
                ds = created[e["path"]]
                for axis, spath in e.get("dim_links", []):
                    try:
                        ds.dims[axis].attach_scale(g[spath])
                    except Exception:
                        pass
                for k, v in e.get("attrs", {}).items():
                    if e.get("is_scale") and k in _SCALE_OWN_ATTRS:
                        continue
                    ds.attrs[k] = container.decode_attr(v)

    return {"gpz_bytes": os.path.getsize(src), "nc_bytes": os.path.getsize(dst)}


def verify_file(src_nc: str, gpz: str) -> dict:
    """逐分块比对 .nc 与 .gpz（解码后），确认逐字节一致。"""
    ok = True
    details = []
    with container.open_reader(gpz) as rd:
        meta = rd.meta
        with h5py.File(src_nc, "r") as f:
            idx = 0
            for e in meta["entries"]:
                same = True
                pos = 0
                d = f[e["path"]]
                dt = np.dtype(e["dtype"])
                for p in e["parts"]:
                    raw = codec.decode_bytes(rd.payload(idx), _part_shape(e, p["n"]), dt)
                    idx += 1
                    ref = np.ascontiguousarray(d[()] if d.ndim == 0 else d[pos:pos + p["n"]])
                    pos += p["n"]
                    if raw != ref.tobytes():
                        same = False
                ok = ok and same
                details.append((e["path"], same))
    return {"exact": ok, "details": details}
