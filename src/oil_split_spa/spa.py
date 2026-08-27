"""Deterministic forward structural-path analysis primitives."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


class _CanonicalJSON(str):
    """String payload that can also be called for small API variations."""

    def __call__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class StructuralPath:
    """One depth-aggregated structural path impact."""

    depth: int
    impact: float

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth <= 0:
            raise ValueError("path depth must be a positive native integer")
        value = float(self.impact)
        if not np.isfinite(value):
            raise ValueError("path impact must be finite")
        object.__setattr__(self, "impact", 0.0 if value == 0.0 else value)

    @property
    def value(self) -> float:
        """Alias used by callers that call the scalar a path value."""

        return self.impact

    @property
    def contribution(self) -> float:
        return self.impact

    @property
    def path_impact(self) -> float:
        return self.impact

    def to_dict(self) -> dict[str, object]:
        return {"depth": self.depth, "impact": self.impact}


# A descriptive alias keeps both common names available without duplicating
# the public record type.
PathRecord = StructuralPath


@dataclass(frozen=True)
class StructuralPathResult:
    """Stable output from :func:`forward_structural_paths`."""

    rows: tuple[StructuralPath, ...]
    total_impact: float

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if any(not isinstance(row, StructuralPath) for row in rows):
            raise TypeError("rows must contain StructuralPath values")
        object.__setattr__(self, "rows", rows)
        value = float(self.total_impact)
        if not np.isfinite(value):
            raise ValueError("total impact must be finite")
        object.__setattr__(self, "total_impact", 0.0 if value == 0.0 else value)

    @property
    def path_rows(self) -> tuple[StructuralPath, ...]:
        return self.rows

    @property
    def total(self) -> float:
        return self.total_impact

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "total_impact": self.total_impact,
        }

    @property
    def canonical_json(self) -> _CanonicalJSON:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return _CanonicalJSON(encoded)

    @property
    def canonical_bytes(self) -> bytes:
        return self.canonical_json.encode("utf-8")

    @property
    def canonical_json_bytes(self) -> bytes:
        return self.canonical_bytes

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    def to_json(self) -> str:
        return str(self.canonical_json)


def _numeric_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.ndim != 1 or np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric vector")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _numeric_matrix(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric square matrix") from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1] or np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric square matrix")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric square matrix") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _cutoff(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("cutoff must be finite and non-negative")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError("cutoff must be finite and non-negative")
    return number


def _depth(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("max_depth must be a positive native integer")
    return value


def forward_structural_paths(
    B: np.ndarray,
    final_demand: np.ndarray,
    stressor: np.ndarray,
    *,
    cutoff: float,
    max_depth: int,
) -> StructuralPathResult:
    """Propagate ``v_next = B @ v_current`` and collect stable depth rows.

    A row is emitted only when its absolute impact is strictly above the
    cutoff.  Once a depth falls to the cutoff (or has a zero propagated
    vector), later depths are not explored.  This gives deterministic finite
    output for both acyclic and convergent matrices while preserving the
    exact forward recurrence requested by the public API.
    """

    matrix = _numeric_matrix(B, name="B")
    demand = _numeric_vector(final_demand, name="final_demand")
    extension = _numeric_vector(stressor, name="stressor")
    if matrix.shape[0] != demand.size or demand.size != extension.size:
        raise ValueError("B, final_demand, and stressor dimensions must agree")
    threshold = _cutoff(cutoff)
    depth_limit = _depth(max_depth)

    current = demand.copy()
    rows: list[StructuralPath] = []
    total = 0.0
    for depth in range(1, depth_limit + 1):
        if not np.any(current):
            break
        next_vector = matrix @ current
        if not np.all(np.isfinite(next_vector)):
            raise ValueError("forward propagation produced non-finite values")
        if not np.any(next_vector):
            break
        impact = float(np.dot(extension, next_vector))
        if not np.isfinite(impact):
            raise ValueError("forward propagation produced a non-finite impact")
        if abs(impact) <= threshold:
            break
        row = StructuralPath(depth=depth, impact=impact)
        rows.append(row)
        total += row.impact
        current = next_vector
    return StructuralPathResult(rows=tuple(rows), total_impact=total)


__all__ = [
    "PathRecord",
    "StructuralPath",
    "StructuralPathResult",
    "forward_structural_paths",
]
