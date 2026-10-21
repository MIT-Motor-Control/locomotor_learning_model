"""Versioned, non-pickle archives for simulation results and resumable state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
MODEL_VERSION = "locad-python-1"


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Object arrays are not supported in simulation archives")
        key = f"array_{len(arrays):05d}"
        arrays[key] = value
        return {"array": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Archive dictionary keys must be strings")
        return {"dict": {key: _encode(item, arrays) for key, item in value.items()}}
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, np.ndarray) for item in value)
        and all(item.shape == value[0].shape and item.dtype == value[0].dtype for item in value)
    ):
        return {"array_list": _encode(np.stack(value), arrays)}
    if isinstance(value, (tuple, list)):
        return {
            "tuple" if isinstance(value, tuple) else "list": [
                _encode(item, arrays) for item in value
            ]
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported archive value: {type(value).__name__}")


def _decode(value: Any, archive: Any) -> Any:
    if not isinstance(value, dict):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("Invalid archive metadata")
    if len(value) != 1:
        raise ValueError("Invalid archive metadata tag")
    tag, payload = next(iter(value.items()))
    if tag == "array" and isinstance(payload, str):
        result = archive[payload]
        if result.dtype.hasobject:
            raise ValueError("Object arrays are not supported")
        return result.copy()
    if tag == "dict" and isinstance(payload, dict):
        return {key: _decode(item, archive) for key, item in payload.items()}
    if tag in ("list", "tuple") and isinstance(payload, list):
        result = [_decode(item, archive) for item in payload]
        return tuple(result) if tag == "tuple" else result
    if tag == "array_list":
        return [item.copy() for item in _decode(payload, archive)]
    if tag == "float" and payload in ("nan", "inf", "-inf"):
        return float(payload)
    raise ValueError("Invalid archive metadata tag")


def save_archive(path: str | Path, kind: str, payload: dict[str, Any]) -> None:
    """Atomically replace an archive; only plain JSON and non-object NumPy arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    tree = _encode(payload, arrays)
    header = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "kind": kind,
        "payload": tree,
    }
    arrays["metadata_json"] = np.array(json.dumps(header, allow_nan=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_archive(path: str | Path, kind: str) -> dict[str, Any]:
    """Read an archive without executing pickle or importing serialized types."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata_json"].item()))
            if header.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported simulation archive schema version")
            if header.get("model_version") != MODEL_VERSION:
                raise ValueError("Simulation archive model version is incompatible")
            if header.get("kind") != kind:
                raise ValueError(f"Expected a {kind} archive")
            result = _decode(header["payload"], archive)
            if not isinstance(result, dict):
                raise ValueError("Archive payload must be a dictionary")  # noqa: TRY004
            return result
    except (KeyError, TypeError, AttributeError, OSError, EOFError) as error:
        raise ValueError(f"Invalid simulation archive: {error}") from error


def configuration_digest(configuration: dict[str, Any]) -> str:
    """Hash configuration content, including array dtype/shape and exact bytes."""
    arrays: dict[str, np.ndarray] = {}
    tree = _encode(configuration, arrays)
    digest = hashlib.sha256(json.dumps(tree, sort_keys=True, allow_nan=False).encode())
    for key, array in arrays.items():
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write a small human-readable manifest atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
