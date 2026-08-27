"""Command-line entry point for ``oil-split-spa``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .config import AnalysisConfig
from .errors import OilSplitError
from .preflight import preflight_input
from .run import APPLICATION_VERSION, run_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oil-split-spa",
        description="Run deterministic REX3 structural path analysis for vegetable oil.",
    )
    parser.add_argument("--version", action="version", version=f"oil-split-spa {APPLICATION_VERSION}")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="validate the selected input and run analysis")
    run.add_argument("--database", "--hdf5", "--input", dest="database", required=True, help="selected HDF5 database")
    run.add_argument("--receipt", required=True, help="matching rebuild receipt")
    run.add_argument("--release-version", required=True, help="expected builder release version")
    run.add_argument("--config", required=True, help="JSON analysis configuration")
    run.add_argument("--output-directory", help="override the runtime output directory")
    run.add_argument("--allow-test-fixture", action="store_true", help="allow the explicit synthetic test fixture profile")
    return parser


def _normalise_argv(argv: Sequence[str] | None) -> list[str] | None:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] not in {"run", "--help", "-h", "--version"}:
        values.insert(0, "run")
    return values


def _load_config(path: Path, output_override: str | None) -> AnalysisConfig | dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise OilSplitError("analysis configuration cannot be read") from exc
    if output_override is not None:
        if not isinstance(payload, dict):
            raise OilSplitError("analysis configuration must be an object")
        payload = dict(payload)
        payload["output_directory"] = output_override
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(_normalise_argv(argv))
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if args.command is None:
            parser.print_help()
            return 0
        config_payload = _load_config(Path(args.config), args.output_directory)
        validated = preflight_input(
            Path(args.database),
            Path(args.receipt),
            args.release_version,
            allow_test_fixture=args.allow_test_fixture,
        )
        config = AnalysisConfig.from_mapping(config_payload, validated_input=validated)
        result = run_analysis(validated, config)
        print(json.dumps({"receipt": result.receipt_path.name, "total_impact": result.total_impact}, sort_keys=True, separators=(",", ":")))
        return 0
    except (OilSplitError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"oil-split-spa: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
