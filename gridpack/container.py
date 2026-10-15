"""gridpack.container -- .gpz 容器读写。

文件结构：
    magic(4) | header_len(uint64 LE) | header(json utf-8) | payload 拼接区

header 里保存元数据与每个 payload 的偏移/长度/校验，因此解码只依赖本文件。
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

MAGIC = b"GPZ1"
VERSION = 1


# --------------------------------------------------------------------------
# 属性序列化（attr 值 -> JSON 可存形式）
# --------------------------------------------------------------------------
def encode_attr(value: Any) -> dict:
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return {"skip": True}
        return {"t": "arr", "d": value.dtype.str, "s": list(value.shape),
                "v": value.tobytes().hex()}
    if isinstance(value, (bytes, np.bytes_)):
        return {"t": "bytes", "v": bytes(value).hex()}
    if isinstance(value, (str, np.str_)):
        return {"t": "str", "v": str(value)}
    if isinstance(value, (bool, np.bool_)):
        return {"t": "bool", "v": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"t": "int", "v": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"t": "float", "v": float(value)}
    if isinstance(value, (list, tuple)):
        return {"t": "list", "v": [encode_attr(x) for x in value]}
    return {"t": "repr", "v": repr(value)}


def decode_attr(obj: dict):
    import numpy as np

    t = obj.get("t")
    if t == "arr":
        a = np.frombuffer(bytes.fromhex(obj["v"]), dtype=np.dtype(obj["d"]))
        return a.reshape(obj["s"])
    if t == "bytes":
        return np.bytes_(bytes.fromhex(obj["v"]))
    if t == "str":
        return obj["v"]
    if t == "bool":
        return bool(obj["v"])
    if t == "int":
        return int(obj["v"])
    if t == "float":
        return float(obj["v"])
    if t == "list":
        return [decode_attr(x) for x in obj["v"]]
    return obj.get("v")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------
# 容器读写
# --------------------------------------------------------------------------
def pack(path: str, meta: dict, payloads: list) -> None:
    """写出容器。payload 顺序即 meta['payload_index'] 的顺序。"""
    idx = []
    off = 0
    for p in payloads:
        idx.append({"offset": off, "length": len(p)})
        off += len(p)
    meta["payload_index"] = idx
    head = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for p in payloads:
            f.write(p)


def _read_header(f):
    if f.read(4) != MAGIC:
        raise ValueError("not a gridpack container")
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n).decode("utf-8")), n


def read_meta(path: str) -> dict:
    with open(path, "rb") as f:
        meta, _ = _read_header(f)
    return meta


def read_payload(path: str, meta: dict, index: int) -> bytes:
    with open(path, "rb") as f:
        _, n = _read_header(f)
        e = meta["payload_index"][index]
        f.seek(e["offset"], 1)
        return f.read(e["length"])


def open_reader(path: str) -> "_Reader":
    return _Reader(path)


class _Reader:
    """顺序读取容器内各 payload，避免一次性载入整个文件。"""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self.meta, n = _read_header(self._f)
        self._base = 4 + 8 + n

    def payload(self, index: int) -> bytes:
        e = self.meta["payload_index"][index]
        self._f.seek(self._base + e["offset"])
        return self._f.read(e["length"])

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
