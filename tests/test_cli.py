"""CLI contract tests for the standalone application."""

from __future__ import annotations

import json
from pathlib import Path

from oil_split_spa.cli import main


def test_cli_help_is_available(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--help"]) == 0
    assert "oil-split-spa" in capsys.readouterr().out


def test_cli_run_writes_outputs(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "final_demand_labels": ["A::Households"],
                "impact_indicators": ["climate-change"],
                "cutoff": 0.01,
                "max_depth": 3,
                "output_directory": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run",
                "--database",
                str(h5_path),
                "--receipt",
                str(receipt_path),
                "--release-version",
                "0.1.0",
                "--config",
                str(config_path),
                "--allow-test-fixture",
            ]
        )
        == 0
    )
    assert (tmp_path / "out" / "application-run-receipt.json").exists()


def test_cli_fails_closed_for_missing_input(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["run", "--database", str(tmp_path / "missing.h5")]) != 0
    assert capsys.readouterr().err
