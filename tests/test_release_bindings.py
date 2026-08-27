"""Synthetic consumer-side release-map binding checks."""

from __future__ import annotations

import json
import hashlib

import h5py
import pytest

from oil_split_spa.errors import InputContractError, ReceiptContractError
from oil_split_spa.receipt import parse_rebuild_receipt
from oil_split_spa.preflight import preflight_input


def _binding() -> dict[str, object]:
    return {
        "schema": "release-map-bindings-1",
        "release_version": "0.1.0",
        "country_order_sha256": "a" * 64,
        "country_unavailable_indices": [77, 103],
        "artifacts": {
            "country-concordance.json": {
                "sha256": "b" * 64,
                "version": "0.1.0",
                "status": "published-country-concordance",
                "coverage": "incomplete-identifiers-declared",
                "validation": {"status": "passed", "validator": "validate_country_concordance"},
            },
            "item-map.json": {
                "sha256": "c" * 64,
                "version": "0.1.0",
                "status": "published-item-map",
                "coverage": "declared-scope-only",
                "validation": {"status": "passed", "validator": "validate_item_map"},
            },
            "vegetable-oil-scope.json": {
                "sha256": "d" * 64,
                "version": "0.1.0",
                "status": "published scope; source coverage is required before a production build",
                "coverage": "Fail closed when any declared country, parent sector, or public oil type is missing.",
                "validation": {"status": "passed", "validator": "validate_vegetable_oil_scope"},
            },
        },
        "validation": {"status": "passed", "evidence": "all required release maps validated"},
    }


def test_app_parser_requires_bindings_for_production(fixture_pair) -> None:
    _, receipt_path = fixture_pair
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["output"]["profile"] = "production"
    with pytest.raises(ReceiptContractError):
        parse_rebuild_receipt(payload)


def test_app_preflight_compares_receipt_and_hdf5_bindings_before_matrix_reads(
    fixture_pair,
    monkeypatch,
) -> None:
    h5_path, receipt_path = fixture_pair
    binding = _binding()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["output"]["release_map_bindings"] = binding
    with h5py.File(h5_path, "r+") as handle:
        mutated = json.loads(json.dumps(binding))
        mutated["artifacts"]["item-map.json"]["sha256"] = "e" * 64
        handle.attrs["release_map_bindings"] = json.dumps(mutated, sort_keys=True, separators=(",", ":"))
    payload["output"]["hdf5_sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def _unexpected_matrix_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("matrix data was read before binding comparison")

    monkeypatch.setattr("oil_split_spa.preflight._read_group", _unexpected_matrix_read)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_app_preflight_binding_mismatch_skips_checksum_and_matrix_reads(
    fixture_pair,
    monkeypatch,
) -> None:
    h5_path, receipt_path = fixture_pair
    binding = _binding()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["output"]["release_map_bindings"] = binding
    with h5py.File(h5_path, "r+") as handle:
        mutated = json.loads(json.dumps(binding))
        mutated["artifacts"]["item-map.json"]["sha256"] = "e" * 64
        handle.attrs["release_map_bindings"] = json.dumps(mutated, sort_keys=True, separators=(",", ":"))
    receipt_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "oil_split_spa.preflight.file_sha256",
        lambda *_args, **_kwargs: pytest.fail("full-file checksum ran before binding comparison"),
    )
    monkeypatch.setattr(
        "oil_split_spa.preflight._read_group",
        lambda *_args, **_kwargs: pytest.fail("matrix-group read ran before binding comparison"),
    )
    with pytest.raises(InputContractError, match="bindings"):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_app_parser_requires_production_map_validation_checks(fixture_pair) -> None:
    _, receipt_path = fixture_pair
    binding = _binding()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["output"]["profile"] = "production"
    payload["output"]["release_map_bindings"] = binding
    payload["checks"] = [check for check in payload["checks"] if check["name"] != "release-map-country"]
    with pytest.raises(ReceiptContractError, match="release-map"):
        parse_rebuild_receipt(payload)
