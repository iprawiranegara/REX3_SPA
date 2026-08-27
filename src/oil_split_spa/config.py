"""Strict, path-free analysis configuration for the public application."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

import numpy as np

from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - imported only for static type checkers
    from .preflight import ValidatedInput


_FIELDS = frozenset(
    {
        "year",
        "final_demand_labels",
        "impact_indicators",
        "cutoff",
        "max_depth",
        "output_directory",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "final_demand_labels",
        "impact_indicators",
        "cutoff",
        "max_depth",
        "output_directory",
    }
)
_REMOVED_FIELDS = frozenset({"origins", "destinations", "stages", "oil_types"})
_CREDENTIAL = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?token|"
    r"authorization|private[_-]?key|client[_-]?secret)\s*[=:]",
    re.IGNORECASE,
)
_HOST_USER = re.compile(r"(?:^|[^A-Za-z0-9_])(?:host|hostname|machine|user|username)\s*[=:]", re.IGNORECASE)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_EMBEDDED_SCHEME = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_URL = re.compile(r"https?://", re.IGNORECASE)
_BARE_FILE = re.compile(r"^[^\s/\\]+\.(?:csv|h5|hdf5|json|mat|npz|parquet|txt|xlsx|zip)$", re.IGNORECASE)
_EMBEDDED_FILE = re.compile(
    r"(?<![A-Za-z0-9_/-])[^\s/\\]+\.(?:csv|h5|hdf5|json|mat|npz|parquet|txt|xlsx|zip)(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)


def _safe_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ConfigurationError(f"{field} contains control characters")
    if _CREDENTIAL.search(value) or _HOST_USER.search(value):
        raise ConfigurationError(f"{field} contains credential or host/user text")
    # ``::`` is the public axis-label separator.  Ordinary prose such as
    # ``Public publisher: release`` remains valid, while locator-like forms
    # (including an underscore-hidden scheme token) are rejected.
    normalized = value.replace("::", "")
    if "@" in value or _BARE_FILE.fullmatch(value) or _EMBEDDED_FILE.search(value) or "\\" in value or "/" in value or _SCHEME.match(normalized) or _EMBEDDED_SCHEME.search(normalized):
        raise ConfigurationError(f"{field} must be a public identifier")
    if _URL.search(value) or value in {".", ".."} or ".." in value:
        raise ConfigurationError(f"{field} must be a public identifier")
    if len(value) > 512:
        raise ConfigurationError(f"{field} is too long")
    return value


def _selection(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (tuple, list)):
        raise ConfigurationError(f"{field} must be an ordered list or tuple of strings")
    if not value:
        raise ConfigurationError(f"{field} must not be empty")
    result = tuple(_safe_text(item, field=field) for item in value)
    if len(set(result)) != len(result):
        raise ConfigurationError(f"{field} must not contain duplicate values")
    return result


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigurationError(f"{field} must be numeric")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ConfigurationError(f"{field} must be finite and non-negative")
    return number


def _max_depth(value: Any) -> int:
    # ``type(...) is int`` is deliberate: NumPy integer scalars are not part
    # of the public configuration contract.
    if type(value) is not int or value <= 0:
        raise ConfigurationError("max_depth must be a positive native integer")
    return value


def _year(value: Any) -> int:
    if type(value) is not int or value != 2022:
        raise ConfigurationError("year must be the integer 2022")
    return value


def _path(value: Any) -> Path:
    if isinstance(value, (str, bytes, bytearray)):
        # Runtime paths are intentionally accepted and never serialized.
        try:
            return Path(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("output_directory must be a valid runtime path") from exc
    if isinstance(value, Path):
        return value
    try:
        return Path(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("output_directory must be a valid runtime path") from exc


def _canonical_payload(config: "AnalysisConfig") -> dict[str, object]:
    return {
        "year": config.year,
        "final_demand_labels": list(config.final_demand_labels),
        "impact_indicators": list(config.impact_indicators),
        "cutoff": config.cutoff,
        "max_depth": config.max_depth,
    }


@dataclass(frozen=True)
class AnalysisConfig:
    """Validated public analysis selections and runtime output location."""

    final_demand_labels: tuple[str, ...]
    impact_indicators: tuple[str, ...]
    cutoff: float
    max_depth: int
    output_directory: Path
    year: int = 2022

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_demand_labels", _selection(self.final_demand_labels, field="final_demand_labels"))
        object.__setattr__(self, "impact_indicators", _selection(self.impact_indicators, field="impact_indicators"))
        object.__setattr__(self, "cutoff", _number(self.cutoff, field="cutoff"))
        object.__setattr__(self, "max_depth", _max_depth(self.max_depth))
        object.__setattr__(self, "year", _year(self.year))
        object.__setattr__(self, "output_directory", _path(self.output_directory))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        validated_input: "ValidatedInput | None" = None,
    ) -> "AnalysisConfig":
        if not isinstance(value, Mapping):
            raise ConfigurationError("analysis configuration must be an object")
        if any(type(key) is not str for key in value):
            raise ConfigurationError("analysis configuration keys must be strings")
        unknown = set(value) - _FIELDS
        if unknown:
            removed = sorted(unknown & _REMOVED_FIELDS)
            if removed:
                raise ConfigurationError(
                    "analysis configuration contains removed fields that are not supported: "
                    f"{removed}"
                )
            raise ConfigurationError(f"analysis configuration has unknown fields: {sorted(unknown)}")
        missing = _REQUIRED_FIELDS - set(value)
        if missing:
            raise ConfigurationError(f"analysis configuration is missing fields: {sorted(missing)}")
        payload = dict(value)
        payload.setdefault("year", 2022)
        config = cls(**payload)
        if validated_input is not None:
            config.validate_against(validated_input)
        return config

    def validate_against(self, validated_input: "ValidatedInput") -> "AnalysisConfig":
        """Check selections against the axes exposed by a validated input."""

        if validated_input is None:
            raise ConfigurationError("validated_input is required for axis membership checks")

        def axis_values(name: str, fallback: Sequence[str] = ()) -> tuple[str, ...]:
            axes = getattr(validated_input, "axes", None)
            if isinstance(axes, Mapping) and name in axes:
                candidate = axes[name]
                if hasattr(candidate, "labels"):
                    candidate = candidate.labels
                return tuple(str(item) for item in candidate)
            candidate = getattr(validated_input, f"{name}_labels", fallback)
            return tuple(str(item) for item in candidate)

        final_labels = set(axis_values("final_demand"))
        unknown_final = set(self.final_demand_labels) - final_labels
        if unknown_final:
            raise ConfigurationError(f"final_demand_labels contains unknown labels: {sorted(unknown_final)}")
        stressors = set(axis_values("stressor"))
        unknown_impacts = set(self.impact_indicators) - stressors
        if unknown_impacts:
            raise ConfigurationError(f"impact_indicators contains unknown labels: {sorted(unknown_impacts)}")
        return self

    @property
    def canonical_payload(self) -> dict[str, object]:
        return _canonical_payload(self)

    @property
    def configuration_fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self.configuration_fingerprint

    def to_dict(self) -> dict[str, object]:
        """Return path-free public configuration data."""

        return dict(self.canonical_payload)


def validate_analysis_config(
    config: AnalysisConfig | Mapping[str, Any],
    *,
    validated_input: "ValidatedInput | None" = None,
) -> AnalysisConfig:
    """Validate a configuration instance or strict mapping."""

    checked = config if isinstance(config, AnalysisConfig) else AnalysisConfig.from_mapping(config)
    if validated_input is not None:
        checked.validate_against(validated_input)
    return checked


__all__ = [
    "AnalysisConfig",
    "validate_analysis_config",
]
