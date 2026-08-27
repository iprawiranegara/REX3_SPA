"""Input preflight contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from oil_split_spa.errors import InputContractError, ReceiptContractError
from oil_split_spa.preflight import preflight_input
from oil_split_spa.receipt import parse_rebuild_receipt, read_rebuild_receipt


def _payload(receipt_path: Path) -> dict[str, object]:
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _write_receipt(receipt_path: Path, payload: dict[str, object]) -> None:
    receipt_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _refresh_hash(h5_path: Path, receipt_path: Path) -> dict[str, object]:
    payload = _payload(receipt_path)
    payload["output"]["hdf5_sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    return payload


def _checksum(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def test_valid_fixture_requires_explicit_opt_in(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    validated = preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    assert validated.output_sha256
    assert str(h5_path) not in json.dumps(validated.to_dict(), sort_keys=True)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0")


def test_canonical_receipt_has_no_top_level_profile(fixture_pair: tuple[Path, Path]) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    normalized = parse_rebuild_receipt(payload)
    assert "profile" not in normalized
    assert normalized["output"]["profile"] == "test-fixture"  # type: ignore[index]


@pytest.mark.parametrize("source_id", ["usda-psd", "un-comtrade"])
def test_receipt_rejects_external_reference_source_ids(
    fixture_pair: tuple[Path, Path], source_id: str
) -> None:
    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    source_record = payload["source_records"][0]  # type: ignore[index]
    source_record["source_id"] = source_id  # type: ignore[index]
    payload["output"]["source_fingerprints"] = {source_id: source_record["sha256"]}  # type: ignore[index]
    _write_receipt(receipt_path, payload)

    with pytest.raises(ReceiptContractError, match=source_id):
        read_rebuild_receipt(receipt_path)
    with pytest.raises(InputContractError, match="rebuild receipt"):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


@pytest.mark.parametrize("source_id", ["usda-psd", "un-comtrade"])
def test_receipt_rejects_external_reference_fallback_source_ids(
    fixture_pair: tuple[Path, Path], source_id: str
) -> None:
    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["fallbacks"] = [
        {
            "name": "external-reference",
            "status": "fallback_used",
            "reason": "synthetic test record",
            "source": source_id,
        }
    ]
    _write_receipt(receipt_path, payload)

    with pytest.raises(ReceiptContractError, match=source_id):
        read_rebuild_receipt(receipt_path)
    with pytest.raises(InputContractError, match="rebuild receipt"):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("source_id", "foo(ssh:bar)"),
        ("dtype", "foo_ssh:host"),
        ("dtype", "float/foo"),
        ("source_file", "bad\x00name.mat"),
        ("source_file", "bad\x7fname.mat"),
        ("source_id", "ssh://host/source"),
    ],
)
def test_source_receipt_rejects_nonportable_fields(
    fixture_pair: tuple[Path, Path], field: str, bad_value: str
) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        source_receipt = json.loads(handle.attrs["source_receipt"])
        source_receipt["Z"][field] = bad_value
        handle.attrs["source_receipt"] = json.dumps(source_receipt, sort_keys=True, separators=(",", ":"))
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_source_receipt_rejects_runtime_absolute_source_path(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        source_receipt = json.loads(handle.attrs["source_receipt"])
        source_receipt["Z"]["source_id"] = str(tmp_path / "raw.mat")
        handle.attrs["source_receipt"] = json.dumps(source_receipt, sort_keys=True, separators=(",", ":"))
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_source_receipt_accepts_safe_basename_and_rejects_locator_punctuation(
    fixture_pair: tuple[Path, Path]
) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        source_receipt = json.loads(handle.attrs["source_receipt"])
        source_receipt["Z"]["source_file"] = "safe-base.mat"
        handle.attrs["source_receipt"] = json.dumps(source_receipt, sort_keys=True, separators=(",", ":"))
    _refresh_hash(h5_path, receipt_path)
    assert preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True).output_sha256
    with h5py.File(h5_path, "r+") as handle:
        source_receipt = json.loads(handle.attrs["source_receipt"])
        source_receipt["Z"]["source_id"] = "ordinary: prose"
        handle.attrs["source_receipt"] = json.dumps(source_receipt, sort_keys=True, separators=(",", ":"))
    _refresh_hash(h5_path, receipt_path)
    assert preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True).output_sha256


@pytest.mark.parametrize("artifact_name", [
    "./vegetable-oil-fixture.h5",
    "nested/vegetable-oil-fixture.h5",
    "nested//vegetable-oil-fixture.h5",
    "../vegetable-oil-fixture.h5",
])
def test_artifact_name_must_be_an_exact_lexical_leaf(
    fixture_pair: tuple[Path, Path], artifact_name: str
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["output"]["artifact_name"] = artifact_name  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_artifact_name_rejects_runtime_absolute_path(
    fixture_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["output"]["artifact_name"] = str(tmp_path / "vegetable-oil-fixture.h5")  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


@pytest.mark.parametrize("url", [
    "https://example.org:bad/data",
    "https://example.org/data\x00",
    "https://example.org/data\x7f",
    "https://user:pass@example.org/data",
    "https://example.org/data?token=bad",
    "https://example.org/data?foo=%20",
    "https://example.org/data?token%3Dabc",
])
def test_acquisition_url_rejects_malformed_or_unsafe_forms(
    fixture_pair: tuple[Path, Path], url: str
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["acquisition_url"] = url  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_public_https_url_and_non_scheme_colon_prose_are_valid(
    fixture_pair: tuple[Path, Path]
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["acquisition_url"] = "https://example.org:443/public-record?download=1"  # type: ignore[index]
    payload["source_records"][0]["publisher"] = "Public publisher: release"  # type: ignore[index]
    payload["source_records"][0]["attribution"] = "Citation https://example.org:443/public-record?download=1"  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    assert read_rebuild_receipt(receipt_path)["source_records"][0]["publisher"] == "Public publisher: release"  # type: ignore[index]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org:443/public-record?download=1",
        "https://[2001:db8::1]:8443/public-record?download=1",
        "https://example.org/public-record?note=a:b",
    ],
)
def test_acquisition_url_accepts_valid_port_ipv6_and_query_colon(
    fixture_pair: tuple[Path, Path], url: str
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["acquisition_url"] = url  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    assert read_rebuild_receipt(receipt_path)["source_records"][0]["acquisition_url"] == url  # type: ignore[index]


@pytest.mark.parametrize(
    ("query", "accepted"),
    [
        ("foo=bar.token=bad", True),
        ("foo=bar:token=bad", True),
        ("foo=bar-token=bad", True),
        ("foo=bar#token=bad", True),
        ("token-ok", False),
        ("xapi_key=ok", False),
        ("user_nameplate", False),
        ("my_key=ok", False),
        ("token;foo=bar", False),
    ],
)
def test_acquisition_url_uses_parameter_oriented_query_policy(
    fixture_pair: tuple[Path, Path], query: str, accepted: bool
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    url = f"https://example.org/public?{query}"
    payload["source_records"][0]["acquisition_url"] = url  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    if accepted:
        assert read_rebuild_receipt(receipt_path)["source_records"][0]["acquisition_url"] == url  # type: ignore[index]
    else:
        with pytest.raises(ReceiptContractError):
            read_rebuild_receipt(receipt_path)


@pytest.mark.parametrize(
    ("query", "accepted"),
    [
        ("foo=bar.token=bad", True),
        ("foo=bar:token=bad", True),
        ("foo=bar-token=bad", True),
        ("token-ok", False),
        ("xapi_key=ok", False),
        ("user_nameplate", False),
        ("my_key=ok", False),
        ("token;foo=bar", False),
        ("foo=%20", False),
        ("token%3Dabc", False),
    ],
)
def test_embedded_https_citations_use_parameter_oriented_query_policy(
    fixture_pair: tuple[Path, Path], query: str, accepted: bool
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    citation = f"Citation https://example.org/public?{query}"
    payload["source_records"][0]["attribution"] = citation  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    if accepted:
        assert read_rebuild_receipt(receipt_path)["source_records"][0]["attribution"] == citation  # type: ignore[index]
    else:
        with pytest.raises(ReceiptContractError):
            read_rebuild_receipt(receipt_path)


@pytest.mark.parametrize("citation", [
    "See https://example.org/public,ssh://host",
    "See https://example.org/public(ssh:host)",
    "See https://example.org/public[ssh:host]",
    "See https://example.org/public;ssh:host",
    "See https://example.org/public_ssh:host",
    "See https://example.org/public.ssh:host",
    "See https://example.org/public_sftp:host",
    "See https://example.org/public_ftp:host",
    "See https://example.org/public_file:host",
    "See https://example.org/public_urn:host",
    "See https://example.org/public_http:host",
    "See https://example.org/public_C:" + "/source",
    "See https://example.org/public://host",
])
def test_embedded_https_citations_reject_attached_non_https_locators(
    fixture_pair: tuple[Path, Path], citation: str
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["attribution"] = citation  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


@pytest.mark.parametrize("citation", [
    "See https://example.org:443/path",
    "See https://example.org/path:segment",
    "See https://example.org/?foo:bar",
    "See https://example.org/api.v1:segment",
    "See https://example.org/api-v1:segment",
    "See https://example.org/api_v1:segment",
    "See https://example.org/path?note=a:b",
])
def test_embedded_https_citations_allow_safe_port_path_and_query_colons(
    fixture_pair: tuple[Path, Path], citation: str
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["attribution"] = citation  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    assert read_rebuild_receipt(receipt_path)["source_records"][0]["attribution"] == citation  # type: ignore[index]


@pytest.mark.parametrize("citation", [
    "Citation https://example.org:bad/path",
    "Citation https://example.org:65536/path",
    "Citation https://example.org:/path",
    "Citation https://-bad.example/path",
    "Citation https://example.org./path",
    "Citation https://example.org\\path",
])
def test_embedded_public_https_citations_reject_malformed_hosts_and_ports(
    fixture_pair: tuple[Path, Path], citation: str
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["attribution"] = citation  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


@pytest.mark.parametrize(
    ("source_text", "accepted"),
    [
        ("foo_xbar:bar", True),
        ("_ssh:host", False),
    ],
)
def test_source_text_identifier_colon_boundary_matches_public_contract(
    fixture_pair: tuple[Path, Path], source_text: str, accepted: bool
) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["source_records"][0]["attribution"] = source_text  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    if accepted:
        assert read_rebuild_receipt(receipt_path)["source_records"][0]["attribution"] == source_text  # type: ignore[index]
    else:
        with pytest.raises(ReceiptContractError):
            read_rebuild_receipt(receipt_path)


def test_underscore_locator_bypass_is_rejected(fixture_pair: tuple[Path, Path]) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["software_identity"] = "foo_ssh:host"
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_public_parser_wraps_unhashable_values(fixture_pair: tuple[Path, Path]) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["backend_identity"] = []
    with pytest.raises(ReceiptContractError):
        parse_rebuild_receipt(payload)


def test_hdf5_hash_and_receipt_release_fail_closed(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["output"]["hdf5_sha256"] = "0" * 64  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
    payload = _payload(receipt_path)
    payload["output"]["hdf5_sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()  # type: ignore[index]
    payload["release_version"] = "0.2.0"
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


@pytest.mark.parametrize("field", ["checks", "optional_checks", "fallbacks"])
def test_receipt_missing_fields_are_rejected(fixture_pair: tuple[Path, Path], field: str) -> None:
    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload.pop(field)
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_receipt_unknown_and_concept_alias_fields_are_rejected(fixture_pair: tuple[Path, Path]) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["unexpected"] = True
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)
    payload = _payload(receipt_path)
    payload["profile"] = "test-fixture"
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)
    payload = _payload(receipt_path)
    payload["command"] = payload.pop("command_name")
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)
    payload = _payload(receipt_path)
    payload["input_sources"] = payload.pop("source_records")
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_required_failed_and_optional_not_run_reason_are_rejected(fixture_pair: tuple[Path, Path]) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    check = payload["checks"][0]
    check.update({"status": "failed", "passed": False})
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)
    payload = _payload(receipt_path)
    for key in ("reason", "message", "evidence"):
        payload["optional_checks"][0][key] = ""  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_invalid_backend_and_source_identity_forms_are_rejected(fixture_pair: tuple[Path, Path]) -> None:
    _, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["backend_identity"] = "unknown-backend"
    payload["output"]["backend_identity"] = "unknown-backend"  # type: ignore[index]
    payload["output"]["profile"] = "test-fixture"  # type: ignore[index]
    with pytest.raises(ReceiptContractError):
        _write_receipt(receipt_path, payload)
        read_rebuild_receipt(receipt_path)
    payload = _payload(receipt_path)
    payload["source_records"][0]["acquisition_url"] = "https://" + "user:secret" + "@example.org/data"  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_missing_group_dataset_attribute_and_changed_axis_are_rejected(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        del handle["mrio_data"]["Q"]
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_missing_matrix_attribute_is_rejected(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        del handle["mrio_data"]["Z"].attrs["value_checksum"]
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_axis_hash_order_and_total_output_mutations_fail(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        manifest = json.loads(handle.attrs["axis_manifest"])
        manifest["sector_country"]["labels"][0], manifest["sector_country"]["labels"][1] = manifest["sector_country"]["labels"][1], manifest["sector_country"]["labels"][0]
        handle.attrs["axis_manifest"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)

    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["output"]["total_output_column"] = "Output"  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_nonfinite_values_fail_even_when_receipt_hash_is_refreshed(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        values = handle["mrio_data"]["L"]["values"]
        values[0, 0] = np.nan
        values_group = handle["mrio_data"]["L"]
        values_group.attrs["value_checksum"] = _checksum(np.asarray(values[()]))
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_malformed_source_receipt_and_path_leak_fail(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        handle.attrs["source_receipt"] = json.dumps({"unsafe": "source"})
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)

    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["software_identity"] = "path=" + str(h5_path.parent / "secret")  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(ReceiptContractError):
        read_rebuild_receipt(receipt_path)


def test_negative_values_are_not_globally_invalid(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        values = handle["mrio_data"]["Z"]["values"]
        values[0, 0] = -2.0
        handle["mrio_data"]["Z"].attrs["value_checksum"] = _checksum(np.asarray(values[()]))
    _refresh_hash(h5_path, receipt_path)
    assert preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True).output_sha256


def test_wrong_year_profile_and_oil_labels_fail(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        handle.attrs["year"] = 2021
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)

    h5_path, receipt_path = fixture_pair
    payload = _payload(receipt_path)
    payload["output"]["profile"] = "production"  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)

    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        labels = handle["mrio_data"]["Z"]["row_labels"]
        labels[0] = "A::Cultivation of oil seeds::unknown"
    _refresh_hash(h5_path, receipt_path)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)


def test_production_profile_cannot_use_fixture_dimensions(fixture_pair: tuple[Path, Path]) -> None:
    h5_path, receipt_path = fixture_pair
    with h5py.File(h5_path, "r+") as handle:
        handle.attrs["profile"] = "production"
    payload = _payload(receipt_path)
    payload["output"]["profile"] = "production"  # type: ignore[index]
    payload["output"]["hdf5_sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()  # type: ignore[index]
    _write_receipt(receipt_path, payload)
    with pytest.raises(InputContractError):
        preflight_input(h5_path, receipt_path, "0.1.0", allow_test_fixture=True)
