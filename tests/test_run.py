"""Application run orchestration and portable artifact tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
import os
from pathlib import Path

import pytest

from oil_split_spa.config import AnalysisConfig
from oil_split_spa.errors import InputContractError, OilSplitError
from oil_split_spa.preflight import preflight_input
from oil_split_spa.run import run_analysis


def _config(output_directory: Path) -> AnalysisConfig:
    return AnalysisConfig(
        final_demand_labels=("A::Households",),
        impact_indicators=("climate-change",),
        cutoff=0.01,
        max_depth=3,
        output_directory=output_directory,
    )


def _selected_config(output_directory: Path, *, final_demand: str, indicator: str) -> AnalysisConfig:
    return AnalysisConfig(
        final_demand_labels=(final_demand,),
        impact_indicators=(indicator,),
        cutoff=0.01,
        max_depth=3,
        output_directory=output_directory,
    )


def test_run_analysis_writes_portable_deterministic_outputs(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    config = _config(tmp_path / "out")

    first = run_analysis(validated, config)
    first_bytes = {path.name: path.read_bytes() for path in first.artifacts}

    second = run_analysis(validated, config)
    second_bytes = {path.name: path.read_bytes() for path in second.artifacts}

    assert first_bytes == second_bytes
    receipt = json.loads(first.receipt_path.read_text(encoding="utf-8"))
    assert str(tmp_path) not in first.receipt_path.read_text(encoding="utf-8")
    assert receipt["input"]["hdf5_sha256"] == hashlib.sha256(h5_path.read_bytes()).hexdigest()
    assert set(receipt["outputs"]) == {"summary", "path_rows"}
    for value in receipt["outputs"].values():
        assert Path(value["artifact_name"]).name == value["artifact_name"]
        assert value["sha256"]


def test_run_analysis_rejects_mutated_hdf5_after_preflight(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    h5_path.write_bytes(h5_path.read_bytes() + b"changed")

    try:
        run_analysis(validated, _config(tmp_path / "out"))
    except Exception as exc:
        assert "input" in str(exc).lower() or "fingerprint" in str(exc).lower()
    else:  # pragma: no cover - assertion branch
        raise AssertionError("mutated HDF5 was accepted")


def test_run_analysis_rejects_mutated_receipt_after_preflight(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload["output"]["hdf5_sha256"] = "c" * 64
    receipt_path.write_text(json.dumps(receipt_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    try:
        run_analysis(validated, _config(tmp_path / "out"))
    except Exception as exc:
        assert "receipt" in str(exc).lower() or "input" in str(exc).lower()
    else:  # pragma: no cover - assertion branch
        raise AssertionError("mutated receipt was accepted")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("software_identity", "changed-software"),
        lambda payload: payload.__setitem__("configuration_fingerprint", "d" * 64),
        lambda payload: payload["source_records"][0].__setitem__("attribution", "changed attribution"),
        lambda payload: payload["checks"][0].__setitem__("message", "changed required-check message"),
        lambda payload: payload.__setitem__(
            "fallbacks",
            [{"name": "synthetic-fallback", "status": "recorded", "reason": "changed fallback"}],
        ),
    ],
    ids=("software", "configuration", "source-attribution", "check-message", "fallback"),
)
def test_run_analysis_rejects_any_verified_receipt_metadata_mutation(
    fixture_pair: tuple[Path, Path], tmp_path: Path, mutation
) -> None:  # type: ignore[no-untyped-def]
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(payload)
    receipt_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(InputContractError):
        run_analysis(validated, _config(tmp_path / "out"))


@pytest.mark.parametrize("leaf", ["summary.json", "path-rows.json", "application-run-receipt.json"])
def test_run_analysis_never_writes_through_output_symlink(
    fixture_pair: tuple[Path, Path], tmp_path: Path, leaf: str
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = tmp_path / "sentinel-symlink"
    sentinel.write_bytes(b"sentinel")
    (output / leaf).symlink_to(sentinel)

    with pytest.raises((InputContractError, OilSplitError)):
        run_analysis(validated, _config(output))
    assert sentinel.read_bytes() == b"sentinel"


@pytest.mark.parametrize("leaf", ["summary.json", "path-rows.json", "application-run-receipt.json"])
def test_run_analysis_never_writes_through_output_hardlink(
    fixture_pair: tuple[Path, Path], tmp_path: Path, leaf: str
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = tmp_path / "sentinel-hardlink"
    sentinel.write_bytes(b"sentinel")
    os.link(sentinel, output / leaf)

    with pytest.raises((InputContractError, OilSplitError)):
        run_analysis(validated, _config(output))
    assert sentinel.read_bytes() == b"sentinel"


def test_configuration_mutation_changes_canonical_run_receipt(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    first = run_analysis(validated, _config(tmp_path / "first"))
    second_config = replace(_config(tmp_path / "second"), cutoff=0.001)
    second = run_analysis(validated, second_config)

    assert first.receipt["analysis"]["configuration_fingerprint"] != second.receipt["analysis"]["configuration_fingerprint"]
    assert first.receipt_path.read_bytes() != second.receipt_path.read_bytes()


def test_final_demand_selection_changes_analysis_inputs_and_result(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    households = run_analysis(
        validated,
        _selected_config(tmp_path / "households", final_demand="A::Households", indicator="climate-change"),
    )
    government = run_analysis(
        validated,
        _selected_config(tmp_path / "government", final_demand="A::Government", indicator="climate-change"),
    )

    assert households.total_impact != government.total_impact
    assert households.receipt["analysis"]["configuration_fingerprint"] != government.receipt["analysis"]["configuration_fingerprint"]


def test_impact_indicator_selection_changes_analysis_inputs_and_result(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    climate = run_analysis(
        validated,
        _selected_config(tmp_path / "climate", final_demand="A::Households", indicator="climate-change"),
    )
    land = run_analysis(
        validated,
        _selected_config(tmp_path / "land", final_demand="A::Households", indicator="land-use"),
    )

    assert climate.total_impact != land.total_impact
    assert climate.receipt["analysis"]["configuration_fingerprint"] != land.receipt["analysis"]["configuration_fingerprint"]


def test_output_artifact_mutation_is_detectable_from_receipt(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    result = run_analysis(validated, _config(tmp_path / "out"))
    result.summary_path.write_bytes(result.summary_path.read_bytes() + b"changed")
    payload = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    recorded = payload["outputs"]["summary"]["sha256"]
    actual = hashlib.sha256(result.summary_path.read_bytes()).hexdigest()
    assert recorded != actual
