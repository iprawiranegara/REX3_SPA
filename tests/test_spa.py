"""Deterministic structural-path execution tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from oil_split_spa.spa import forward_structural_paths


def test_forward_structural_paths_golden_case() -> None:
    B = np.array([[0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]])
    y = np.array([0.0, 0.0, 1.0])
    s = np.array([2.0, 1.0, 0.0])

    result = forward_structural_paths(B, y, s, cutoff=0.01, max_depth=3)

    assert result.total_impact == pytest.approx(0.42)
    assert [row.depth for row in result.rows] == [1, 2]
    assert [row.impact for row in result.rows] == pytest.approx([0.30, 0.12])


def test_forward_structural_paths_canonical_json_is_byte_stable() -> None:
    B = np.array([[0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]])
    y = np.array([0.0, 0.0, 1.0])
    s = np.array([2.0, 1.0, 0.0])

    first = forward_structural_paths(B, y, s, cutoff=0.01, max_depth=3)
    second = forward_structural_paths(B, y, s, cutoff=0.01, max_depth=3)

    assert first.canonical_json == second.canonical_json
    assert json.loads(first.canonical_json)["total_impact"] == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("B", "y", "s", "cutoff", "max_depth"),
    [
        (np.ones((2, 3)), np.ones(3), np.ones(3), 0.01, 3),
        (np.ones((2, 2)), np.ones(3), np.ones(2), 0.01, 3),
        (np.ones((2, 2)), np.ones(2), np.ones(2), -0.01, 3),
        (np.ones((2, 2)), np.ones(2), np.ones(2), 0.01, 0),
        (np.ones((2, 2)), np.array([np.nan, 0.0]), np.ones(2), 0.01, 3),
    ],
)
def test_forward_structural_paths_rejects_invalid_inputs(
    B: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    cutoff: float,
    max_depth: int,
) -> None:
    with pytest.raises(ValueError):
        forward_structural_paths(B, y, s, cutoff=cutoff, max_depth=max_depth)


def test_forward_structural_paths_stops_at_cutoff_and_zero() -> None:
    B = np.array([[0.0, 0.001], [0.0, 0.0]])
    y = np.array([0.0, 1.0])
    s = np.array([1.0, 0.0])

    result = forward_structural_paths(B, y, s, cutoff=0.01, max_depth=10)

    assert result.rows == ()
    assert result.total_impact == 0.0
