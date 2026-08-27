"""Strict analysis-configuration contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from oil_split_spa.config import AnalysisConfig
from oil_split_spa.errors import ConfigurationError


@dataclass
class _Validated:
    axes: dict[str, tuple[str, ...]]


@pytest.fixture
def validated() -> _Validated:
    return _Validated(
        axes={
            "sector_country": (
                "A::Cultivation of oil seeds::palm",
                "A::Cultivation of oil seeds::soybean",
                "B::Processing vegetable oils and fats::palm",
            ),
            "final_demand": ("A::households", "B::exports"),
            "stressor": ("climate-change", "land-use"),
        }
    )


def _payload(tmp_path: Path) -> dict[str, object]:
    return {
        "final_demand_labels": ("A::households",),
        "impact_indicators": ("climate-change",),
        "cutoff": 0.01,
        "max_depth": 5,
        "output_directory": tmp_path,
    }


def test_valid_configuration_and_path_free_fingerprint(tmp_path: Path, validated: _Validated) -> None:
    config = AnalysisConfig.from_mapping(_payload(tmp_path), validated_input=validated)
    assert config.year == 2022
    assert config.configuration_fingerprint
    assert set(config.canonical_payload) == {
        "year",
        "final_demand_labels",
        "impact_indicators",
        "cutoff",
        "max_depth",
    }
    assert "output_directory" not in config.to_dict()
    assert str(tmp_path) not in repr(config.to_dict())

    other = AnalysisConfig.from_mapping({**_payload(tmp_path), "output_directory": tmp_path / "different"}, validated_input=validated)
    assert other.configuration_fingerprint == config.configuration_fingerprint


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_demand_labels", ("",)),
        ("impact_indicators", ("host=bad",)),
        ("impact_indicators", ("secret.csv",)),
        ("cutoff", -1.0),
        ("cutoff", np.nan),
        ("cutoff", np.inf),
        ("max_depth", 0),
        ("max_depth", True),
        ("max_depth", np.int64(3)),
        ("year", 2021),
    ],
)
def test_invalid_configuration_values_raise(tmp_path: Path, field: str, value: object) -> None:
    payload = _payload(tmp_path)
    payload[field] = value
    with pytest.raises(ConfigurationError):
        AnalysisConfig.from_mapping(payload)


def test_unknown_fields_and_selection_types_raise(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["unexpected"] = "nope"
    with pytest.raises(ConfigurationError):
        AnalysisConfig.from_mapping(payload)

    payload = _payload(tmp_path)
    payload[1] = "malformed-key"  # type: ignore[index]
    payload["unexpected"] = "nope"
    with pytest.raises(ConfigurationError):
        AnalysisConfig.from_mapping(payload)
    payload = _payload(tmp_path)
    payload["final_demand_labels"] = "A::households"
    with pytest.raises(ConfigurationError):
        AnalysisConfig.from_mapping(payload)


@pytest.mark.parametrize("field", ["origins", "destinations", "stages", "oil_types"])
def test_removed_selection_fields_fail_with_a_clear_public_error(tmp_path: Path, field: str) -> None:
    payload = _payload(tmp_path)
    payload[field] = ["inert-selection"]

    with pytest.raises(ConfigurationError, match=field):
        AnalysisConfig.from_mapping(payload)


def test_membership_is_checked_against_validated_axes(tmp_path: Path, validated: _Validated) -> None:
    payload = _payload(tmp_path)
    payload["final_demand_labels"] = ("Z::not-an-axis",)
    with pytest.raises(ConfigurationError):
        AnalysisConfig.from_mapping(payload, validated_input=validated)
    payload = _payload(tmp_path)
    payload["impact_indicators"] = ("Z::not-an-axis",)
    with pytest.raises(ConfigurationError):
        AnalysisConfig.from_mapping(payload, validated_input=validated)
