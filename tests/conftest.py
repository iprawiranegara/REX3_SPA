"""Synthetic fixture helpers for the standalone application package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def public_oil_types() -> tuple[str, ...]:
    return ("palm", "soybean", "sunflower", "rapeseed", "other")


def _axis_hash(labels: list[str]) -> str:
    payload = json.dumps(labels, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value_checksum(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


@pytest.fixture
def fixture_pair(tmp_path: Path, public_oil_types: tuple[str, ...]) -> tuple[Path, Path]:
    """Write a portable test-fixture HDF5 and matching strict receipt."""

    n = 12
    sector_labels = tuple(
        [f"A::Cultivation of oil seeds::{oil}" for oil in public_oil_types]
        + [f"A::Processing vegetable oils and fats::{oil}" for oil in public_oil_types]
        + ["A::Untouched sector 1", "A::Untouched sector 2"]
    )
    axes = {
        "sector_country": list(sector_labels),
        "final_demand": ["A::Households", "A::Government"],
        "stressor": ["climate-change", "land-use"],
        "x_columns": ["Total Output"],
    }
    arrays = {
        "Z": np.eye(n, dtype=np.float64) * 0.1,
        "Y": np.column_stack(
            [
                np.ones(n, dtype=np.float64),
                np.ones(n, dtype=np.float64) * 2.0,
            ]
        ),
        "x": np.ones((n, 1), dtype=np.float64),
        "A": np.eye(n, dtype=np.float64) * 0.05,
        "B": np.eye(n, dtype=np.float64) * 0.05,
        "L": np.eye(n, dtype=np.float64),
        "G": np.eye(n, dtype=np.float64),
        "F": np.vstack(
            [
                np.ones(n, dtype=np.float64) * 0.2,
                np.ones(n, dtype=np.float64) * 0.4,
            ]
        ),
        "S": np.vstack(
            [
                np.ones(n, dtype=np.float64) * 0.1,
                np.ones(n, dtype=np.float64) * 0.3,
            ]
        ),
        "Q": np.vstack(
            [
                np.ones(n, dtype=np.float64) * 0.2,
                np.ones(n, dtype=np.float64) * 0.5,
            ]
        ),
    }
    roles = {
        "Z": ("sector_country", "sector_country"),
        "A": ("sector_country", "sector_country"),
        "B": ("sector_country", "sector_country"),
        "L": ("sector_country", "sector_country"),
        "G": ("sector_country", "sector_country"),
        "Y": ("sector_country", "final_demand"),
        "x": ("sector_country", "x_columns"),
        "F": ("stressor", "sector_country"),
        "S": ("stressor", "sector_country"),
        "Q": ("stressor", "sector_country"),
    }
    axis_manifest = {
        name: {"name": name, "length": len(labels), "labels": labels, "hash": _axis_hash(labels)}
        for name, labels in axes.items()
    }
    source_fingerprints = {"synthetic-source": "a" * 64}
    h5_path = tmp_path / "vegetable-oil-fixture.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["schema_version"] = "base-h5-1"
        handle.attrs["release_version"] = "0.1.0"
        handle.attrs["year"] = 2022
        handle.attrs["axis_manifest"] = json.dumps(axis_manifest, sort_keys=True, separators=(",", ":"))
        handle.attrs["source_fingerprints"] = json.dumps(source_fingerprints, sort_keys=True, separators=(",", ":"))
        source_receipt = {
            "Z": {
                "source_id": "synthetic-base",
                "source_variable": "T",
                "source_file": "base.mat",
                "source_shape": [n, n],
                "final_shape": [n, n],
                "dtype": "float64",
                "orientation": "as_is",
            },
            "Y": {
                "source_id": "synthetic-base",
                "source_variable": "Y",
                "source_file": "base.mat",
                "source_shape": [n, 1],
                "final_shape": [n, 1],
                "dtype": "float64",
                "orientation": "as_is",
            },
            "Q": {
                "source_id": "synthetic-base",
                "source_variable": "Q",
                "source_file": "base.mat",
                "source_shape": [1, n],
                "final_shape": [1, n],
                "dtype": "float64",
                "orientation": "as_is",
            },
        }
        handle.attrs["source_receipt"] = json.dumps(source_receipt, sort_keys=True, separators=(",", ":"))
        handle.attrs["backend_identity"] = "numpy"
        handle.attrs["profile"] = "test-fixture"
        data_group = handle.create_group("mrio_data")
        for name, values in arrays.items():
            group = data_group.create_group(name)
            row_axis, col_axis = roles[name]
            group.create_dataset("values", data=values)
            group.create_dataset("row_labels", data=np.asarray(axes[row_axis], dtype=object), dtype=h5py.string_dtype("utf-8"))
            group.create_dataset("col_labels", data=np.asarray(axes[col_axis], dtype=object), dtype=h5py.string_dtype("utf-8"))
            if name == "x":
                group.create_dataset("columns_values", data=np.asarray(axes["x_columns"], dtype=object), dtype=h5py.string_dtype("utf-8"))
            group.attrs["shape"] = np.asarray(values.shape, dtype=np.int64)
            group.attrs["dtype"] = str(values.dtype)
            group.attrs["row_axis_name"] = row_axis
            group.attrs["col_axis_name"] = col_axis
            group.attrs["row_axis_hash"] = _axis_hash(axes[row_axis])
            group.attrs["col_axis_hash"] = _axis_hash(axes[col_axis])
            group.attrs["value_checksum"] = _value_checksum(values)

    digest = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    output_shapes = {name: list(values.shape) for name, values in arrays.items()}
    receipt = {
        "receipt_schema": "vegetable-oil-rebuild-receipt-1",
        "release_version": "0.1.0",
        "command_name": "build",
        "configuration_fingerprint": "b" * 64,
        "source_records": [
            {
                "source_id": "synthetic-source",
                "publisher": "Synthetic publisher",
                "acquisition_url": "https://example.org/synthetic",
                "expected_format": "hdf5",
                "attribution": "Synthetic fixture",
                "reuse_condition": "Use for tests only",
                "version": "fixture-1",
                "retrieved_at": "2026-08-25T00:00:00Z",
                "sha256": "a" * 64,
            }
        ],
        "output": {
            "artifact_name": h5_path.name,
            "hdf5_sha256": digest,
            "schema_version": "base-h5-1",
            "release_version": "0.1.0",
            "year": 2022,
            "required_groups": ["Z", "Y", "x", "A", "B", "L", "G", "F", "S", "Q"],
            "shapes": output_shapes,
            "total_output_column": "Total Output",
            "axis_lengths": {name: len(labels) for name, labels in axes.items()},
            "axis_hashes": {name: _axis_hash(labels) for name, labels in axes.items()},
            "source_fingerprints": source_fingerprints,
            "backend_identity": "numpy",
            "profile": "test-fixture",
        },
        "checks": [
            {
                "name": "fixture-structure",
                "status": "passed",
                "passed": True,
                "required": True,
                "reason": "fixture accepted",
                "message": "fixture accepted",
                "evidence": "synthetic fixture",
                "observed": True,
                "expected": True,
                "tolerance": None,
            }
        ],
        "optional_checks": [
            {
                "name": "optional-comparison",
                "status": "not_run",
                "passed": False,
                "required": False,
                "reason": "no permitted comparator",
                "message": "no permitted comparator",
                "evidence": "no permitted comparator",
                "observed": None,
                "expected": None,
                "tolerance": None,
            }
        ],
        "backend_identity": "numpy",
        "software_identity": "oil-split-spa-fixture-0.1.0",
        "fallbacks": [],
    }
    receipt_path = tmp_path / "vegetable-oil-fixture-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return h5_path, receipt_path
