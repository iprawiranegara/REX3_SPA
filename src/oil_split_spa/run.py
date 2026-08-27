"""Application orchestration and portable run artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

import h5py
import numpy as np

from .config import AnalysisConfig, validate_analysis_config
from .errors import InputContractError, OilSplitError
from .preflight import ValidatedInput, preflight_input
from .receipt import file_sha256, read_rebuild_receipt
from .spa import StructuralPath, StructuralPathResult, forward_structural_paths


APPLICATION_VERSION = "0.1.0"
RUN_RECEIPT_SCHEMA = "oil-split-spa-application-run-receipt-1"
SUMMARY_SCHEMA = "oil-split-spa-summary-1"
PATH_ROWS_SCHEMA = "oil-split-spa-path-rows-1"
_SAFE_LEAF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RunArtifact:
    """A written artifact and its portable digest."""

    path: Path
    artifact_name: str
    sha256: str

    @property
    def name(self) -> str:
        return self.artifact_name

    def to_dict(self) -> dict[str, str]:
        return {"artifact_name": self.artifact_name, "sha256": self.sha256}


@dataclass(frozen=True)
class RunResult:
    """Stable in-memory description of a completed application run."""

    summary: dict[str, object]
    path_rows: tuple[dict[str, object], ...]
    receipt: dict[str, object]
    output_directory: Path
    summary_path: Path
    path_rows_path: Path
    receipt_path: Path
    structural_results: tuple[StructuralPathResult, ...]

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        return self.path_rows

    @property
    def paths_path(self) -> Path:
        return self.path_rows_path

    @property
    def run_receipt_path(self) -> Path:
        return self.receipt_path

    @property
    def artifacts(self) -> tuple[Path, ...]:
        return (self.summary_path, self.path_rows_path, self.receipt_path)

    @property
    def total_impact(self) -> float:
        return float(self.summary["total_impact"])

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "path_rows": list(self.path_rows),
            "receipt": self.receipt,
        }


def _canonical_bytes(payload: Mapping[str, object] | Sequence[object]) -> bytes:
    try:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OilSplitError("run output contains unsupported non-portable data") from exc
    return text.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_leaf(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or Path(value).name != value or not _SAFE_LEAF.fullmatch(value):
        raise InputContractError(f"{field} is not a portable artifact name")
    return value


def _validate_input_unchanged(validated: ValidatedInput) -> None:
    if not isinstance(validated, ValidatedInput):
        raise InputContractError("run_analysis requires a ValidatedInput")
    try:
        current_receipt_hash = file_sha256(validated.receipt_path)
    except (OSError, TypeError, ValueError) as exc:
        raise InputContractError("rebuild receipt cannot be read") from exc
    if current_receipt_hash != validated.receipt_sha256:
        raise InputContractError("rebuild receipt content changed after validation")
    try:
        receipt = read_rebuild_receipt(validated.receipt_path)
    except Exception as exc:
        if isinstance(exc, OilSplitError):
            raise
        raise InputContractError("rebuild receipt cannot be read") from exc
    output = receipt.output_identity
    if output.release_version != validated.release_version or output.year != validated.year or output.profile != validated.profile or output.backend_identity != validated.backend_identity:
        raise InputContractError("rebuild receipt identity differs from validated input")
    if output.artifact_name != validated.h5_path.name or output.hdf5_sha256 != validated.output_sha256:
        raise InputContractError("rebuild receipt fingerprint differs from validated input")
    try:
        current_hash = hashlib.sha256(validated.h5_path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        raise InputContractError("selected HDF5 cannot be read") from exc
    if current_hash != validated.output_sha256:
        raise InputContractError("selected HDF5 fingerprint changed after validation")


def _indices(labels: Sequence[str], selected: Sequence[str], *, field: str) -> tuple[int, ...]:
    positions = {label: index for index, label in enumerate(labels)}
    try:
        return tuple(positions[item] for item in selected)
    except KeyError as exc:
        raise InputContractError(f"{field} selection is not present in the validated input") from exc


def _read_analysis_arrays(
    validated: ValidatedInput, config: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    try:
        with h5py.File(validated.h5_path, "r") as handle:
            group = handle["mrio_data"]
            B = np.asarray(group["B"]["values"][()], dtype=np.float64)
            Y = np.asarray(group["Y"]["values"][()], dtype=np.float64)
            S = np.asarray(group["S"]["values"][()], dtype=np.float64)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise InputContractError("validated HDF5 arrays cannot be read") from exc
    sector_labels = tuple(validated.axes["sector_country"])
    final_labels = tuple(validated.axes["final_demand"])
    stressor_labels = tuple(validated.axes["stressor"])
    final_positions = _indices(final_labels, config.final_demand_labels, field="final_demand_labels")
    stressor_positions = _indices(stressor_labels, config.impact_indicators, field="impact_indicators")
    if B.shape != (len(sector_labels), len(sector_labels)) or Y.shape != (len(sector_labels), len(final_labels)) or S.shape != (len(stressor_labels), len(sector_labels)):
        raise InputContractError("validated HDF5 analysis arrays have unexpected shapes")
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(Y)) or not np.all(np.isfinite(S)):
        raise InputContractError("validated HDF5 analysis arrays are not finite")
    demand = np.sum(Y[:, final_positions], axis=1, dtype=np.float64)
    extensions = S[list(stressor_positions), :]
    return B, demand, extensions, tuple(config.impact_indicators)


def _readback_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OilSplitError("output artifact cannot be opened without following links") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OilSplitError("output artifact is not a private regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _artifact(path: Path, payload: bytes) -> RunArtifact:
    """Atomically write one private regular artifact and hash its readback."""

    try:
        parent_metadata = path.parent.stat()
    except OSError as exc:
        raise OilSplitError("output artifact directory cannot be inspected") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise OilSplitError("output artifact parent is not a directory")
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise OilSplitError("output artifact cannot be inspected") from exc
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise OilSplitError("output artifact leaf must not be a symlink")
        if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
            raise OilSplitError("output artifact leaf must be a private regular file")

    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        actual = _readback_bytes(path)
        if actual != payload:
            raise OilSplitError("output artifact readback differs from the requested bytes")
        digest = _sha256_bytes(actual)
    except OilSplitError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise OilSplitError("output artifact cannot be written safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return RunArtifact(path=path, artifact_name=_safe_leaf(path.name, field="output artifact"), sha256=digest)


def _path_row(indicator: str, row: StructuralPath) -> dict[str, object]:
    return {"impact_indicator": indicator, "depth": row.depth, "impact": row.impact}


def run_analysis(validated: ValidatedInput, config: AnalysisConfig) -> RunResult:
    """Run deterministic SPA after rechecking the selected input boundary."""

    _validate_input_unchanged(validated)
    checked = validate_analysis_config(config, validated_input=validated)
    B, demand, extensions, indicators = _read_analysis_arrays(validated, checked)

    structural_results: list[StructuralPathResult] = []
    path_rows: list[dict[str, object]] = []
    indicator_summaries: list[dict[str, object]] = []
    total = 0.0
    for indicator, extension in zip(indicators, extensions, strict=True):
        result = forward_structural_paths(B, demand, extension, cutoff=checked.cutoff, max_depth=checked.max_depth)
        structural_results.append(result)
        path_rows.extend(_path_row(indicator, row) for row in result.rows)
        indicator_summaries.append(
            {
                "impact_indicator": indicator,
                "path_count": len(result.rows),
                "total_impact": result.total_impact,
            }
        )
        total += result.total_impact

    output_directory = Path(checked.output_directory)
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OilSplitError("output directory cannot be created") from exc

    summary: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "year": checked.year,
        "configuration_fingerprint": checked.configuration_fingerprint,
        "input_artifact_name": _safe_leaf(validated.h5_path.name, field="input artifact"),
        "input_sha256": validated.output_sha256,
        "path_count": len(path_rows),
        "total_impact": 0.0 if total == 0.0 else total,
        "indicators": indicator_summaries,
    }
    paths_payload: list[dict[str, object]] = path_rows
    summary_bytes = _canonical_bytes(summary)
    path_rows_bytes = _canonical_bytes({"schema": PATH_ROWS_SCHEMA, "rows": paths_payload})
    summary_artifact = _artifact(output_directory / "summary.json", summary_bytes)
    path_rows_artifact = _artifact(output_directory / "path-rows.json", path_rows_bytes)

    receipt: dict[str, object] = {
        "receipt_schema": RUN_RECEIPT_SCHEMA,
        "application": {"name": "oil-split-spa", "version": APPLICATION_VERSION},
        "input": {
            "artifact_name": _safe_leaf(validated.h5_path.name, field="input artifact"),
            "receipt_artifact_name": _safe_leaf(validated.receipt_path.name, field="input receipt artifact"),
            "hdf5_sha256": validated.output_sha256,
            "receipt_sha256": validated.receipt_sha256,
            "builder_release_version": validated.release_version,
            "year": validated.year,
            "profile": validated.profile,
        },
        "analysis": {
            "configuration_fingerprint": checked.configuration_fingerprint,
            "path_count": len(path_rows),
            "total_impact": 0.0 if total == 0.0 else total,
        },
        "outputs": {
            "summary": summary_artifact.to_dict(),
            "path_rows": path_rows_artifact.to_dict(),
        },
    }
    receipt_artifact = _artifact(output_directory / "application-run-receipt.json", _canonical_bytes(receipt))
    return RunResult(
        summary=summary,
        path_rows=tuple(path_rows),
        receipt=receipt,
        output_directory=output_directory,
        summary_path=summary_artifact.path,
        path_rows_path=path_rows_artifact.path,
        receipt_path=receipt_artifact.path,
        structural_results=tuple(structural_results),
    )


__all__ = [
    "APPLICATION_VERSION",
    "PATH_ROWS_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "SUMMARY_SCHEMA",
    "RunArtifact",
    "RunResult",
    "run_analysis",
]
