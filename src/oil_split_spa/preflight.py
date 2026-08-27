"""Fail-closed validation of a checked vegetable-oil HDF5 input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import h5py
import numpy as np
import re

from .errors import InputContractError
from .errors import ReceiptContractError
from .receipt import (
    AXIS_NAMES,
    MATRIX_AXIS_ROLES,
    PUBLIC_OIL_TYPES,
    PUBLIC_PARENT_IDS,
    PUBLIC_PROFILES,
    REQUIRED_GROUPS,
    ReceiptDocument,
    file_sha256,
    parse_rebuild_receipt,
)


_ROOT_ATTRS = frozenset(
    {
        "schema_version",
        "release_version",
        "year",
        "axis_manifest",
        "source_fingerprints",
        "source_receipt",
        "backend_identity",
        "profile",
    }
)
_GROUP_ATTRS = frozenset(
    {
        "shape",
        "dtype",
        "row_axis_name",
        "col_axis_name",
        "row_axis_hash",
        "col_axis_hash",
        "value_checksum",
    }
)
_GROUP_DATASETS = frozenset({"values", "row_labels", "col_labels"})
_X_DATASETS = frozenset({"values", "row_labels", "col_labels", "columns_values"})
_URL = re.compile(r"https?://", re.IGNORECASE)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_EMBEDDED_SCHEME = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_BARE_FILE = re.compile(r"^[^\s/\\]+\.(?:csv|h5|hdf5|json|mat|npz|parquet|txt|xlsx|zip)$", re.IGNORECASE)
_EMBEDDED_FILE = re.compile(
    r"(?<![A-Za-z0-9_/-])[^\s/\\]+\.(?:csv|h5|hdf5|json|mat|npz|parquet|txt|xlsx|zip)(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?token|"
    r"authorization|private[_-]?key|sig|signature|key|auth|client[_-]?secret)\s*[=:]",
    re.IGNORECASE,
)
_HOST_USER = re.compile(r"(?:^|[^A-Za-z0-9_])(?:host|hostname|machine|user|username)\s*[=:]", re.IGNORECASE)


def _fail(message: str) -> InputContractError:
    return InputContractError(message)


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(value: Any, *, field: str) -> str:
    value = _decode(value)
    if type(value) is not str:
        raise _fail(f"{field} is not text")
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _axis_label(value: Any, *, field: str) -> str:
    """Validate one portable public axis label."""

    value = _decode(value)
    if type(value) is not str or not value or value != value.strip():
        raise _fail(f"{field} contains an invalid label")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _fail(f"{field} contains an invalid label")
    if _CREDENTIAL.search(value) or _HOST_USER.search(value) or "@" in value:
        raise _fail(f"{field} contains unsafe label text")
    if _BARE_FILE.fullmatch(value) or _EMBEDDED_FILE.search(value):
        raise _fail(f"{field} contains local-file text")
    if "/" in value or "\\" in value or ".." in value or _URL.search(value):
        raise _fail(f"{field} contains path-like text")
    normalized = value.replace("::", "")
    if _SCHEME.match(normalized) or _EMBEDDED_SCHEME.search(normalized):
        raise _fail(f"{field} contains a locator scheme")
    return value


def _portable_source_text(value: Any, *, field: str, allow_basename: bool = False) -> str:
    """Validate one source-receipt text field without retaining local locators."""

    value = _decode(value)
    if type(value) is not str or not value or value != value.strip():
        raise _fail(f"{field} is not portable text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _fail(f"{field} contains control characters")
    if "@" in value or "/" in value or "\\" in value or ".." in value:
        raise _fail(f"{field} contains a path or credential token")
    if _URL.search(value) or _CREDENTIAL.search(value) or _HOST_USER.search(value):
        raise _fail(f"{field} contains an unsafe locator")
    if _SCHEME.match(value) or _EMBEDDED_SCHEME.search(value):
        raise _fail(f"{field} contains a locator scheme")
    if allow_basename:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
            raise _fail(f"{field} must be a safe basename")
    elif _BARE_FILE.fullmatch(value) or _EMBEDDED_FILE.search(value):
        raise _fail(f"{field} contains local-file text")
    return value


def _axis_hash(labels: list[Any]) -> str:
    return hashlib.sha256(_stable_json(labels).encode("utf-8")).hexdigest()


def _decode_labels(dataset: h5py.Dataset, *, field: str, expected_length: int) -> tuple[str, ...]:
    if dataset.ndim != 1 or dataset.shape != (expected_length,):
        raise _fail(f"{field} has an invalid label shape")
    try:
        values = dataset[()]
    except (OSError, TypeError, ValueError) as exc:
        raise _fail(f"{field} cannot be read") from exc
    labels: list[str] = []
    for value in np.asarray(values).tolist():
        decoded = _decode(value)
        labels.append(_axis_label(decoded, field=field))
    if len(set(labels)) != len(labels):
        raise _fail(f"{field} contains duplicate labels")
    return tuple(labels)


def _read_axis_manifest(handle: h5py.File) -> dict[str, tuple[str, ...]]:
    raw = handle.attrs.get("axis_manifest")
    if raw is None:
        raise _fail("HDF5 axis manifest is missing")
    try:
        manifest = json.loads(_text(raw, field="axis_manifest"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail("HDF5 axis manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != set(AXIS_NAMES):
        raise _fail("HDF5 axis manifest has the wrong axes")
    axes: dict[str, tuple[str, ...]] = {}
    for axis_name in AXIS_NAMES:
        entry = manifest[axis_name]
        if not isinstance(entry, dict) or set(entry) != {"name", "length", "labels", "hash"}:
            raise _fail(f"axis manifest entry {axis_name} is malformed")
        if entry["name"] != axis_name or type(entry["length"]) is not int or entry["length"] < 1:
            raise _fail(f"axis manifest entry {axis_name} is invalid")
        labels = entry["labels"]
        if not isinstance(labels, list) or len(labels) != entry["length"]:
            raise _fail(f"axis manifest labels for {axis_name} are invalid")
        labels = [_axis_label(label, field=f"axis manifest {axis_name}") for label in labels]
        if len(set(labels)) != len(labels) or entry["hash"] != _axis_hash(labels):
            raise _fail(f"axis manifest hash for {axis_name} is invalid")
        axes[axis_name] = tuple(labels)
    if axes["x_columns"] != ("Total Output",):
        raise _fail("x_columns axis must contain exactly Total Output")
    return axes


def _read_root(handle: h5py.File) -> tuple[dict[str, Any], dict[str, str]]:
    root_attrs = set(handle.attrs.keys())
    if not root_attrs.issubset(_ROOT_ATTRS | {"release_map_bindings"}) or not _ROOT_ATTRS.issubset(root_attrs):
        raise _fail("HDF5 root attributes are incomplete or unknown")
    schema = _text(handle.attrs["schema_version"], field="schema_version")
    if schema != "base-h5-1":
        raise _fail("HDF5 schema version is unsupported")
    release = _text(handle.attrs["release_version"], field="release_version")
    year = _decode(handle.attrs["year"])
    if type(year) is not int or year != 2022:
        raise _fail("HDF5 year must be 2022")
    backend = _text(handle.attrs["backend_identity"], field="backend_identity")
    profile = _text(handle.attrs["profile"], field="profile")
    if backend not in {"numpy", "cupy-managed"} or profile not in PUBLIC_PROFILES:
        raise _fail("HDF5 backend or profile is unknown")
    try:
        source_values = json.loads(_text(handle.attrs["source_fingerprints"], field="source_fingerprints"))
        source_receipt = json.loads(_text(handle.attrs["source_receipt"], field="source_receipt"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail("HDF5 source metadata is not valid JSON") from exc
    if not isinstance(source_values, dict) or not isinstance(source_receipt, dict):
        raise _fail("HDF5 source metadata must be objects")
    _validate_source_receipt(source_receipt)
    fingerprints: dict[str, str] = {}
    for key, value in source_values.items():
        if type(key) is not str or not key or not isinstance(value, str) or len(value) != 64:
            raise _fail("HDF5 source fingerprints are malformed")
        try:
            int(value, 16)
        except ValueError as exc:
            raise _fail("HDF5 source fingerprints are malformed") from exc
        fingerprints[key] = value.lower()
    raw_bindings = handle.attrs.get("release_map_bindings")
    if raw_bindings is None:
        bindings = None
    else:
        try:
            from .receipt import _validate_release_map_bindings

            bindings = _validate_release_map_bindings(
                json.loads(_text(raw_bindings, field="release_map_bindings")),
                release_version=release,
            )
        except (TypeError, ValueError, json.JSONDecodeError, ReceiptContractError) as exc:
            raise _fail("HDF5 release map bindings are invalid") from exc
    if profile == "production" and bindings is None:
        raise _fail("production HDF5 is missing release map bindings")
    return {
        "schema_version": schema,
        "release_version": release,
        "year": year,
        "backend_identity": backend,
        "profile": profile,
        "release_map_bindings": bindings,
    }, fingerprints


def _validate_source_receipt(value: Mapping[str, Any]) -> None:
    """Validate the portable raw-input receipt stored on the HDF5 root."""

    if not value:
        return
    allowed_groups = {"Z", "Y", "Q"}
    if set(value) - allowed_groups:
        raise _fail("HDF5 source receipt contains unknown groups")
    required_fields = {"source_id", "source_variable", "source_file", "source_shape", "final_shape", "dtype", "orientation"}
    expected_variables = {"Z": {"T", "Z"}, "Y": {"Y"}, "Q": {"Q"}}
    for group_name, item in value.items():
        if not isinstance(item, dict) or set(item) != required_fields:
            raise _fail("HDF5 source receipt entry is malformed")
        for field_name in ("source_id", "source_variable", "dtype", "orientation"):
            _portable_source_text(item[field_name], field=f"source_receipt.{group_name}.{field_name}")
        if item["source_variable"] not in expected_variables[group_name] or item["orientation"] not in {"as_is", "transpose"}:
            raise _fail("HDF5 source receipt axis metadata is invalid")
        _portable_source_text(item["source_file"], field=f"source_receipt.{group_name}.source_file", allow_basename=True)
        for field_name in ("source_shape", "final_shape"):
            shape = item[field_name]
            if not isinstance(shape, list) or len(shape) != 2 or any(type(dim) is not int or dim < 0 for dim in shape):
                raise _fail("HDF5 source receipt shape is invalid")
        expected_shape = item["source_shape"] if item["orientation"] == "as_is" else [item["source_shape"][1], item["source_shape"][0]]
        if item["final_shape"] != expected_shape:
            raise _fail("HDF5 source receipt orientation is contradictory")


def _stream_values(dataset: h5py.Dataset, *, name: str, expected_shape: tuple[int, int]) -> str:
    if tuple(dataset.shape) != expected_shape or dataset.ndim != 2:
        raise _fail(f"{name} values shape is invalid")
    if not np.issubdtype(dataset.dtype, np.number):
        raise _fail(f"{name} values are not numeric")
    digest = hashlib.sha256()
    digest.update(str(dataset.dtype).encode("utf-8"))
    digest.update(_stable_json(list(expected_shape)).encode("utf-8"))
    rows, columns = expected_shape
    max_bytes = 8 * 1024 * 1024
    bytes_per_row = max(1, columns * dataset.dtype.itemsize)
    row_block = max(1, min(rows, max_bytes // bytes_per_row))
    for start in range(0, rows, row_block):
        stop = min(rows, start + row_block)
        try:
            chunk = np.asarray(dataset[start:stop, :])
        except (OSError, TypeError, ValueError) as exc:
            raise _fail(f"{name} values cannot be read") from exc
        if not np.all(np.isfinite(chunk)):
            raise _fail(f"{name} values contain non-finite data")
        contiguous = np.ascontiguousarray(chunk)
        digest.update(contiguous.view(np.uint8))
        del chunk, contiguous
    return digest.hexdigest()


def _read_group(
    data_group: h5py.Group,
    name: str,
    axes: Mapping[str, tuple[str, ...]],
    receipt_shapes: Mapping[str, tuple[int, int]],
) -> tuple[tuple[int, int], str, str]:
    group = data_group[name]
    if not isinstance(group, h5py.Group):
        raise _fail(f"matrix group {name} is invalid")
    expected_datasets = _X_DATASETS if name == "x" else _GROUP_DATASETS
    if set(group.keys()) != expected_datasets or set(group.attrs.keys()) != _GROUP_ATTRS:
        raise _fail(f"matrix group {name} has unexpected datasets or attributes")
    if any(set(group[dataset_name].attrs.keys()) for dataset_name in expected_datasets):
        raise _fail(f"matrix group {name} datasets have unexpected attributes")
    row_axis, col_axis = MATRIX_AXIS_ROLES[name]
    if _text(group.attrs["row_axis_name"], field=f"{name}.row_axis_name") != row_axis or _text(group.attrs["col_axis_name"], field=f"{name}.col_axis_name") != col_axis:
        raise _fail(f"matrix group {name} axis roles are invalid")
    if _text(group.attrs["row_axis_hash"], field=f"{name}.row_axis_hash") != _axis_hash(list(axes[row_axis])) or _text(group.attrs["col_axis_hash"], field=f"{name}.col_axis_hash") != _axis_hash(list(axes[col_axis])):
        raise _fail(f"matrix group {name} axis hashes are invalid")
    row_labels = _decode_labels(group["row_labels"], field=f"{name}.row_labels", expected_length=len(axes[row_axis]))
    col_labels = _decode_labels(group["col_labels"], field=f"{name}.col_labels", expected_length=len(axes[col_axis]))
    if row_labels != axes[row_axis] or col_labels != axes[col_axis]:
        raise _fail(f"matrix group {name} labels do not match the axis manifest")
    if name == "x":
        x_labels = _decode_labels(group["columns_values"], field="x.columns_values", expected_length=1)
        if x_labels != ("Total Output",):
            raise _fail("x columns are not exactly Total Output")
    expected = (len(axes[row_axis]), len(axes[col_axis]))
    stored_shape = np.asarray(group.attrs["shape"])
    if stored_shape.ndim != 1 or stored_shape.shape != (2,) or any(type(_decode(item)) is not int for item in stored_shape.tolist()) or tuple(int(item) for item in stored_shape.tolist()) != expected:
        raise _fail(f"matrix group {name} shape metadata is invalid")
    values = group["values"]
    if _text(group.attrs["dtype"], field=f"{name}.dtype") != str(values.dtype):
        raise _fail(f"matrix group {name} dtype metadata is invalid")
    checksum = _stream_values(values, name=name, expected_shape=expected)
    if _text(group.attrs["value_checksum"], field=f"{name}.value_checksum").lower() != checksum:
        raise _fail(f"matrix group {name} value checksum is invalid")
    if tuple(receipt_shapes[name]) != expected:
        raise _fail(f"matrix group {name} shape differs from receipt")
    return expected, _text(group.attrs["dtype"], field=f"{name}.dtype"), checksum


def _normalise_stage(value: str) -> str:
    text = value.lower().replace("&", "and")
    text = "".join(character if character.isalnum() else "-" for character in text)
    text = "-".join(part for part in text.split("-") if part)
    return {
        "cultivation-of-oil-seeds": "cultivation-oil-seeds",
        "processing-vegetable-oils-and-fats": "processing-vegetable-oils-fats",
    }.get(text, text)


def _validate_oil_labels(labels: tuple[str, ...]) -> None:
    # Expanded public sectors are represented as ``country::parent::oil``;
    # the original parent name is intentionally retained by the builder.  Do
    # not depend on that local parent text: bind only the public child order
    # and require two distinct target-parent spans.
    groups: dict[str, list[str]] = {}
    for label in labels:
        parts = label.split("::")
        if len(parts) < 3:
            continue
        oil = parts[-1]
        key = "::".join(parts[:-1])
        groups.setdefault(key, []).append(oil)

    candidate_count = 0
    for oils in groups.values():
        # A target span has exactly five children.  If a span has a known oil
        # label, or happens to have five children, it is a declared oil span
        # and must match the canonical public order exactly.  This catches a
        # renamed, duplicated, or missing oil even when all known names are
        # absent from one mutated span.
        if any(oil in PUBLIC_OIL_TYPES for oil in oils) or len(oils) == len(PUBLIC_OIL_TYPES):
            if tuple(oils) != PUBLIC_OIL_TYPES:
                raise _fail("oil-sector labels must contain the five public oil types in order")
            candidate_count += 1
    if candidate_count < 2:
        raise _fail("HDF5 sector axis is missing one of the two oil parent spans")


@dataclass(frozen=True)
class ValidatedInput:
    """Public preflight result; paths remain runtime-only fields."""

    h5_path: Path
    receipt_path: Path
    receipt_sha256: str
    output_sha256: str
    release_version: str
    year: int
    profile: str
    backend_identity: str
    shapes: dict[str, tuple[int, int]]
    axes: dict[str, tuple[str, ...]]
    source_fingerprints: dict[str, str]
    release_map_bindings: dict[str, Any] | None = None

    @property
    def sector_labels(self) -> tuple[str, ...]:
        return self.axes["sector_country"]

    @property
    def final_demand_labels(self) -> tuple[str, ...]:
        return self.axes["final_demand"]

    @property
    def stressor_labels(self) -> tuple[str, ...]:
        return self.axes["stressor"]

    @property
    def country_labels(self) -> tuple[str, ...]:
        seen: list[str] = []
        for label in self.sector_labels:
            country = label.split("::", 1)[0]
            if country not in seen:
                seen.append(country)
        return tuple(seen)

    def to_dict(self) -> dict[str, object]:
        """Return path-free metadata suitable for public result JSON."""

        result: dict[str, object] = {
            "output_sha256": self.output_sha256,
            "receipt_sha256": self.receipt_sha256,
            "release_version": self.release_version,
            "year": self.year,
            "profile": self.profile,
            "backend_identity": self.backend_identity,
            "shapes": {name: list(shape) for name, shape in self.shapes.items()},
            "axes": {name: list(labels) for name, labels in self.axes.items()},
            "source_fingerprints": dict(self.source_fingerprints),
        }
        if self.release_map_bindings is not None:
            result["release_map_bindings"] = dict(self.release_map_bindings)
        return result


def _production_gate(output_shapes: Mapping[str, tuple[int, int]], axes: Mapping[str, tuple[str, ...]]) -> None:
    n = 32319
    expected = {
        "Z": (n, n),
        "A": (n, n),
        "B": (n, n),
        "L": (n, n),
        "G": (n, n),
        "Y": (n, 189),
        "x": (n, 1),
        "F": (24, n),
        "S": (24, n),
        "Q": (24, n),
    }
    if output_shapes != expected or len(axes["sector_country"]) != n or len(axes["final_demand"]) != 189 or len(axes["stressor"]) != 24 or len(axes["x_columns"]) != 1:
        raise _fail("production profile dimensions do not match the published layout")


def _fixture_gate(output_shapes: Mapping[str, tuple[int, int]], axes: Mapping[str, tuple[str, ...]]) -> None:
    n = len(axes["sector_country"])
    expected = {
        "Z": (n, n),
        "A": (n, n),
        "B": (n, n),
        "L": (n, n),
        "G": (n, n),
        "Y": (n, len(axes["final_demand"])),
        "x": (n, 1),
        "F": (len(axes["stressor"]), n),
        "S": (len(axes["stressor"]), n),
        "Q": (len(axes["stressor"]), n),
    }
    if output_shapes != expected:
        raise _fail("test-fixture matrix shapes do not match the declared axes")


def preflight_input(
    h5_path: Path,
    receipt_path: Path,
    expected_release: str,
    *,
    allow_test_fixture: bool = False,
) -> ValidatedInput:
    """Validate a receipt and HDF5 pair before any analysis code sees it."""

    try:
        h5_path = Path(h5_path)
        receipt_path = Path(receipt_path)
    except (TypeError, ValueError) as exc:
        raise _fail("selected HDF5 and receipt paths are invalid") from exc
    if type(allow_test_fixture) is not bool:
        raise _fail("allow_test_fixture must be a native boolean")
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
        receipt: ReceiptDocument = parse_rebuild_receipt(receipt_payload)
    except InputContractError:
        raise
    except Exception as exc:
        raise _fail("rebuild receipt is invalid") from exc
    if type(expected_release) is not str or not expected_release or expected_release != receipt["release_version"]:
        raise _fail("receipt release version does not match the expected release")
    output = receipt.output_identity
    if output.schema_version != "base-h5-1" or output.year != 2022:
        raise _fail("receipt output schema or year is unsupported")
    if output.profile == "test-fixture" and not allow_test_fixture:
        raise _fail("test-fixture input requires explicit opt-in")
    if output.artifact_name != h5_path.name:
        raise _fail("selected HDF5 name does not match the receipt artifact")
    if not output.source_fingerprints or not receipt["source_records"]:
        raise _fail("receipt must contain source identities")
    try:
        with h5py.File(h5_path, "r") as handle:
            roots, root_sources = _read_root(handle)
            if roots["release_version"] != receipt["release_version"] or roots["year"] != output.year or roots["backend_identity"] != receipt["backend_identity"] or roots["profile"] != output.profile:
                raise _fail("HDF5 root identity does not match the receipt")
            if root_sources != output.source_fingerprints:
                raise _fail("HDF5 source fingerprints do not match the receipt")
            if roots.get("release_map_bindings") != output.release_map_bindings:
                raise _fail("HDF5 release map bindings do not match the receipt")
            try:
                actual_hash = file_sha256(h5_path)
            except (OSError, TypeError, ValueError) as exc:
                raise _fail("selected HDF5 cannot be hashed") from exc
            if actual_hash != output.hdf5_sha256:
                raise _fail("HDF5 SHA-256 does not match the receipt")
            axes = _read_axis_manifest(handle)
            if set(handle.keys()) != {"mrio_data"} or not isinstance(handle["mrio_data"], h5py.Group):
                raise _fail("HDF5 top-level groups are invalid")
            data_group = handle["mrio_data"]
            if set(data_group.attrs.keys()):
                raise _fail("HDF5 mrio_data group has unexpected attributes")
            if set(data_group.keys()) != set(REQUIRED_GROUPS):
                raise _fail("HDF5 must contain exactly the ten matrix groups")
            actual_shapes: dict[str, tuple[int, int]] = {}
            for name in REQUIRED_GROUPS:
                actual_shapes[name], _, _ = _read_group(data_group, name, axes, output.shapes)
            if {name: len(axes[name]) for name in AXIS_NAMES} != output.axis_lengths:
                raise _fail("HDF5 axis lengths do not match the receipt")
            if {name: _axis_hash(list(axes[name])) for name in AXIS_NAMES} != output.axis_hashes:
                raise _fail("HDF5 axis hashes do not match the receipt")
            _validate_oil_labels(axes["sector_country"])
            if output.profile == "production":
                _production_gate(actual_shapes, axes)
            else:
                _fixture_gate(actual_shapes, axes)
    except InputContractError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _fail("selected HDF5 does not satisfy the input contract") from exc
    except Exception as exc:
        # Any unexpected parser/HDF5 boundary failure is still a contract
        # rejection; callers must never observe implementation exceptions.
        raise _fail("selected HDF5 does not satisfy the input contract") from exc
    return ValidatedInput(
        h5_path=h5_path,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        output_sha256=actual_hash,
        release_version=receipt["release_version"],
        year=output.year,
        profile=output.profile,
        backend_identity=output.backend_identity,
        shapes=actual_shapes,
        axes=axes,
        source_fingerprints=dict(output.source_fingerprints),
        release_map_bindings=output.release_map_bindings,
    )


__all__ = ["ValidatedInput", "preflight_input"]
