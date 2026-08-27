"""Standalone strict parser for the vegetable-oil rebuild receipt.

The application repeats the builder's public wire contract locally.  Keeping
this parser independent means a consumer never imports builder implementation
code or trusts a builder-side object at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .errors import ReceiptContractError


RECEIPT_SCHEMA_VERSION = "vegetable-oil-rebuild-receipt-1"
REQUIRED_GROUPS: tuple[str, ...] = ("Z", "Y", "x", "A", "B", "L", "G", "F", "S", "Q")
AXIS_NAMES: tuple[str, ...] = ("sector_country", "final_demand", "stressor", "x_columns")
MATRIX_AXIS_ROLES: dict[str, tuple[str, str]] = {
    "Z": ("sector_country", "sector_country"),
    "A": ("sector_country", "sector_country"),
    "B": ("sector_country", "sector_country"),
    "L": ("sector_country", "sector_country"),
    "G": ("sector_country", "sector_country"),
    "Y": ("sector_country", "final_demand"),
    "x": ("sector_country", "x_columns"),
    "F": ("stressor", "sector_country"),
    "S": ("stressor", "sector_country"),
    "Q": ("stressor", "sector_country"),
}
PUBLIC_OIL_TYPES: tuple[str, ...] = ("palm", "soybean", "sunflower", "rapeseed", "other")
PUBLIC_PARENT_IDS: tuple[str, ...] = (
    "cultivation-oil-seeds",
    "processing-vegetable-oils-fats",
)
_UNSUPPORTED_RUNTIME_SOURCE_IDS = frozenset({"usda-psd", "un-comtrade"})
PUBLIC_BACKENDS = frozenset({"numpy", "cupy-managed"})
PUBLIC_PROFILES = frozenset({"production", "test-fixture"})
REQUIRED_RELEASE_MAP_CHECKS = frozenset(
    {"release-map-country", "release-map-item", "release-map-scope", "release-map-country-order"}
)
COMMANDS = frozenset({"build", "prepare-weights", "validate"})
CHECK_STATUSES = frozenset({"passed", "failed", "not_run"})
FALLBACK_STATUSES = frozenset({"fallback_used", "recorded", "not_used"})
BINDING_SCHEMA_VERSION = "release-map-bindings-1"
CANONICAL_RELEASE_MAP_FILENAMES: tuple[str, ...] = (
    "country-concordance.json",
    "item-map.json",
    "vegetable-oil-scope.json",
)
EXPECTED_UNAVAILABLE_COUNTRY_INDICES: tuple[int, ...] = (77, 103)

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_EMBEDDED_SCHEME = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_BARE_FILE = re.compile(r"^[^\s/\\]+\.(?:csv|h5|hdf5|json|mat|npz|parquet|txt|xlsx|zip)$", re.IGNORECASE)
_EMBEDDED_FILE = re.compile(
    r"(?<![A-Za-z0-9_/-])[^\s/\\]+\.(?:csv|h5|hdf5|json|mat|npz|parquet|txt|xlsx|zip)(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://", re.IGNORECASE)
_CREDENTIAL = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?token|authorization|private[_-]?key|"
    r"sig|signature|key|auth|client[_-]?secret)\s*[=:]",
    re.IGNORECASE,
)
_HOST_USER = re.compile(r"(?:^|[^A-Za-z0-9_])(?:host|hostname|machine|user|username)\s*[=:]", re.IGNORECASE)
_QUERY_CREDENTIAL_NAME = re.compile(
    r"(?:^|[._:-])(?:api[_-]?key|access[_-]?token|token|key|password|passwd|pwd|secret|"
    r"auth|authorization|client[_-]?secret|private[_-]?key|signature|sig|host|hostname|machine|"
    r"user|username)(?=[._:-]|$)",
    re.IGNORECASE,
)
_PUBLIC_HTTPS_URL = re.compile(r"(?<![A-Za-z0-9_])https://[^\s]+", re.IGNORECASE)
# The attached-token scan intentionally treats a period as a boundary so a
# greedy scheme match cannot swallow ``.ssh:host`` as one opaque token.
_ATTACHED_SCHEME = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9+]*):(?://|[^\s])")
_SOURCE_SCHEME = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])")
_UNDERSCORE_SOURCE_SCHEME = re.compile(
    r"(?<![A-Za-z0-9_])_[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\s])"
)
_KNOWN_NON_HTTPS_SCHEMES = frozenset(
    {
        "about",
        "data",
        "file",
        "ftp",
        "git",
        "http",
        "jdbc",
        "mailto",
        "mysql",
        "nfs",
        "postgres",
        "s3",
        "sftp",
        "smb",
        "ssh",
        "urn",
        "ws",
        "wss",
    }
)


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_text(value: Any, *, field: str, allow_url: bool = False) -> str:
    value = _decode(value)
    if type(value) is not str or not value or value != value.strip():
        raise ReceiptContractError(f"{field} must be a nonblank public string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ReceiptContractError(f"{field} contains control characters")
    if _CREDENTIAL.search(value) or _HOST_USER.search(value):
        raise ReceiptContractError(f"{field} contains credential or host/user text")
    if "@" in value or _BARE_FILE.fullmatch(value) or _EMBEDDED_FILE.search(value):
        raise ReceiptContractError(f"{field} contains local or credential-like text")
    if allow_url:
        return value
    if "\\" in value or "/" in value or _SCHEME.match(value) or _EMBEDDED_SCHEME.search(value):
        raise ReceiptContractError(f"{field} must be path-free")
    if value.startswith(("/", "./", "../", "~/")) or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ReceiptContractError(f"{field} must be path-free")
    if value in {".", ".."} or ".." in value or _URL.search(value):
        raise ReceiptContractError(f"{field} must be path-free")
    if len(value) > 512:
        raise ReceiptContractError(f"{field} is too long")
    return value


def _safe_source_text(value: Any, *, field: str) -> str:
    """Validate source attribution text while permitting public HTTPS links."""

    value = _decode(value)
    if type(value) is not str or not value or value != value.strip():
        raise ReceiptContractError(f"{field} must be a nonblank public string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ReceiptContractError(f"{field} contains control characters")
    if "@" in value:
        raise ReceiptContractError(f"{field} contains credential or host/user text")
    if _BARE_FILE.fullmatch(value) or _EMBEDDED_FILE.search(value):
        raise ReceiptContractError(f"{field} contains local-file text")
    if ".." in value or "\\" in value:
        raise ReceiptContractError(f"{field} must be path-free")
    if value.startswith(("/", "./", "../", "~/")) or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ReceiptContractError(f"{field} must be path-free")
    residual = _PUBLIC_HTTPS_URL.sub("public-url", value)
    if _CREDENTIAL.search(residual) or _HOST_USER.search(residual):
        raise ReceiptContractError(f"{field} contains credential or host/user text")
    if "/" in residual:
        raise ReceiptContractError(f"{field} must be path-free outside public HTTPS citations")
    for match in _PUBLIC_HTTPS_URL.finditer(value):
        token = match.group(0)
        if _has_attached_locator(token):
            raise ReceiptContractError(f"{field} contains an attached non-HTTPS locator")
        try:
            _url(token)
        except ReceiptContractError as exc:
            raise ReceiptContractError(f"{field} contains an invalid HTTPS citation") from exc
    # Any URI-like token that is not an HTTPS citation is a local/unsafe
    # locator in a portable source record.
    residual_scheme = re.sub(r"https://[^\s]+", "public-url", value, flags=re.IGNORECASE)
    # Treat underscores inside an identifier as ordinary prose (for example,
    # ``foo_xbar:bar``), while retaining a guard for underscore-prefixed
    # locator tokens such as ``_ssh:host``.
    if _SOURCE_SCHEME.search(residual_scheme) or _UNDERSCORE_SOURCE_SCHEME.search(residual_scheme):
        raise ReceiptContractError(f"{field} contains a non-HTTPS locator")
    return value


def _has_attached_locator(token: str) -> bool:
    """Reject URI tokens attached inside one greedy HTTPS citation span."""

    parsed = urlparse(token)
    authority_start = token.lower().find("://") + 3
    if authority_start < 3:
        return True
    # Do not interpret the ordinary host/port colon as an attached locator;
    # scan only the path, query, and fragment suffix after the HTTPS authority.
    authority_end = authority_start + len(parsed.netloc)
    suffix = token[authority_end:]
    for match in _ATTACHED_SCHEME.finditer(suffix):
        if match.start() == 0 and match.group(1).lower() == "https":
            continue
        scheme = match.group(1).lower()
        drive_like = len(scheme) == 1 and match.group(0)[1:3] in {":/", ":\\"}
        if "://" in match.group(0) or scheme in _KNOWN_NON_HTTPS_SCHEMES or drive_like:
            return True
    return False


def _query_contains_credential_name(query: str) -> bool:
    """Check raw parameter names without scanning query values."""

    for parameter in re.split(r"[&;]", query):
        name = parameter.split("=", 1)[0]
        if _QUERY_CREDENTIAL_NAME.search(name):
            return True
    return False


def _identifier(value: Any, *, field: str) -> str:
    text = _safe_text(value, field=field)
    if not _IDENTIFIER.fullmatch(text):
        raise ReceiptContractError(f"{field} is not a safe identifier")
    return text


def _runtime_source_id(value: Any, *, field: str) -> str:
    source_id = _identifier(value, field=field)
    if source_id in _UNSUPPORTED_RUNTIME_SOURCE_IDS:
        raise ReceiptContractError(
            f"{field} uses an unsupported external-reference source ID: {source_id}"
        )
    return source_id


def _digest(value: Any, *, field: str) -> str:
    value = _decode(value)
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ReceiptContractError(f"{field} must be a 64-character SHA-256 digest")
    return value.lower()


def _validate_release_map_bindings(
    value: Mapping[str, Any],
    *,
    release_version: str | None = None,
) -> dict[str, Any]:
    """Validate the builder's normalized, path-free release-map binding."""

    payload = _mapping(value, field="output.release_map_bindings")
    expected = {"schema", "release_version", "country_order_sha256", "country_unavailable_indices", "artifacts", "validation"}
    if set(payload) != expected or payload["schema"] != BINDING_SCHEMA_VERSION:
        raise ReceiptContractError("release_map_bindings is incomplete or unsupported")
    binding_release = _safe_text(payload["release_version"], field="release_map_bindings.release_version")
    if release_version is not None and binding_release != release_version:
        raise ReceiptContractError("release_map_bindings release version differs from output")
    order_hash = _digest(payload["country_order_sha256"], field="release_map_bindings.country_order_sha256")
    if payload["country_unavailable_indices"] != list(EXPECTED_UNAVAILABLE_COUNTRY_INDICES):
        raise ReceiptContractError("release_map_bindings unavailable-country declaration is invalid")
    artifacts = _mapping(payload["artifacts"], field="release_map_bindings.artifacts")
    if set(artifacts) != set(CANONICAL_RELEASE_MAP_FILENAMES):
        raise ReceiptContractError("release_map_bindings artifacts are incomplete")
    validators = {
        "country-concordance.json": "validate_country_concordance",
        "item-map.json": "validate_item_map",
        "vegetable-oil-scope.json": "validate_vegetable_oil_scope",
    }
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for filename in CANONICAL_RELEASE_MAP_FILENAMES:
        artifact = _mapping(artifacts[filename], field=f"release_map_bindings.artifacts.{filename}")
        if set(artifact) != {"sha256", "version", "status", "coverage", "validation"}:
            raise ReceiptContractError(f"release_map_bindings artifact {filename} is malformed")
        digest = _digest(artifact["sha256"], field=f"release_map_bindings.artifacts.{filename}.sha256")
        version = _safe_text(artifact["version"], field=f"release_map_bindings.artifacts.{filename}.version")
        if version != binding_release:
            raise ReceiptContractError(f"release_map_bindings artifact {filename} version differs from release")
        status = _safe_text(artifact["status"], field=f"release_map_bindings.artifacts.{filename}.status")
        coverage = _safe_text(artifact["coverage"], field=f"release_map_bindings.artifacts.{filename}.coverage")
        validation = _mapping(artifact["validation"], field=f"release_map_bindings.artifacts.{filename}.validation")
        if set(validation) != {"status", "validator"} or validation["status"] != "passed" or validation["validator"] != validators[filename]:
            raise ReceiptContractError(f"release_map_bindings artifact {filename} validation evidence is invalid")
        normalized_artifacts[filename] = {
            "sha256": digest,
            "version": version,
            "status": status,
            "coverage": coverage,
            "validation": {"status": "passed", "validator": validators[filename]},
        }
    evidence = _mapping(payload["validation"], field="release_map_bindings.validation")
    if set(evidence) != {"status", "evidence"} or evidence["status"] != "passed":
        raise ReceiptContractError("release_map_bindings validation evidence is invalid")
    return {
        "schema": BINDING_SCHEMA_VERSION,
        "release_version": binding_release,
        "country_order_sha256": order_hash,
        "country_unavailable_indices": list(EXPECTED_UNAVAILABLE_COUNTRY_INDICES),
        "artifacts": normalized_artifacts,
        "validation": {"status": "passed", "evidence": _safe_text(evidence["evidence"], field="release_map_bindings.validation.evidence")},
    }


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ReceiptContractError(f"{field} must be an object with string keys")
    return dict(value)


def _safe_scalar(value: Any, *, field: str) -> Any:
    value = _decode(value)
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ReceiptContractError(f"{field} must be finite")
        return value
    if type(value) is str:
        return _safe_text(value, field=field)
    raise ReceiptContractError(f"{field} contains unsupported diagnostic data")


def _url(value: Any) -> str:
    value = _decode(value)
    if type(value) is not str or not value or value != value.strip() or any(char.isspace() for char in value):
        raise ReceiptContractError("acquisition_url must be a nonblank HTTPS URL")
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or "\\" in value:
        raise ReceiptContractError("acquisition_url contains control or path-separator text")
    try:
        parsed = urlparse(value)
        if parsed.scheme.lower() != "https" or not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise ValueError
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError
        if parsed.netloc.endswith(":"):
            raise ValueError
        # Accessing ``port`` forces malformed and out-of-range port forms to
        # fail before the URL can cross the public contract boundary.
        parsed.port
    except (TypeError, ValueError) as exc:
        raise ReceiptContractError("acquisition_url must be a valid HTTPS URL") from exc
    if "@" in value or ".." in value:
        raise ReceiptContractError("acquisition_url contains unsafe text")
    if ":" in hostname:
        try:
            ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ReceiptContractError("acquisition_url contains a malformed host") from exc
    else:
        if not re.fullmatch(r"[A-Za-z0-9.-]+", hostname) or hostname.startswith(".") or hostname.endswith("."):
            raise ReceiptContractError("acquisition_url contains a malformed host")
        host_labels = hostname.split(".")
        if any(not label or label[0] == "-" or label[-1] == "-" for label in host_labels):
            raise ReceiptContractError("acquisition_url contains a malformed host")
    # Keep credential scanning on the authority/path only. Query values and
    # fragments are ordinary public locator text and are checked separately.
    queryless = parsed._replace(query="", fragment="").geturl()
    if _CREDENTIAL.search(queryless) or _HOST_USER.search(queryless):
        raise ReceiptContractError("acquisition_url contains credential or host/user text")
    if "%" in parsed.query:
        raise ReceiptContractError("acquisition_url contains percent escapes in HTTPS query text")
    if _query_contains_credential_name(parsed.query):
        raise ReceiptContractError("acquisition_url contains credential-like query text")
    return value


def _backend(value: Any, *, field: str) -> str:
    value = _decode(value)
    if type(value) is not str or value not in PUBLIC_BACKENDS:
        raise ReceiptContractError(f"{field} is unknown")
    return value


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    publisher: str
    acquisition_url: str
    expected_format: str
    attribution: str
    reuse_condition: str
    version: str
    retrieved_at: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _runtime_source_id(self.source_id, field="source_id"))
        for field_name in ("publisher", "expected_format", "version"):
            object.__setattr__(self, field_name, _safe_text(getattr(self, field_name), field=field_name))
        for field_name in ("attribution", "reuse_condition"):
            object.__setattr__(self, field_name, _safe_source_text(getattr(self, field_name), field=field_name))
        object.__setattr__(self, "acquisition_url", _url(self.acquisition_url))
        retrieved = _safe_text(self.retrieved_at, field="retrieved_at")
        try:
            parsed = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReceiptContractError("retrieved_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ReceiptContractError("retrieved_at must include a timezone")
        object.__setattr__(self, "retrieved_at", retrieved)
        object.__setattr__(self, "sha256", _digest(self.sha256, field="sha256"))

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "publisher": self.publisher,
            "acquisition_url": self.acquisition_url,
            "expected_format": self.expected_format,
            "attribution": self.attribution,
            "reuse_condition": self.reuse_condition,
            "version": self.version,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OutputIdentity:
    artifact_name: str
    hdf5_sha256: str
    schema_version: str
    release_version: str
    year: int
    required_groups: tuple[str, ...]
    shapes: dict[str, tuple[int, int]]
    total_output_column: str
    axis_lengths: dict[str, int]
    axis_hashes: dict[str, str]
    source_fingerprints: dict[str, str]
    backend_identity: str
    profile: str
    release_map_bindings: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OutputIdentity":
        payload = _mapping(value, field="output")
        fields = {
            "artifact_name",
            "hdf5_sha256",
            "schema_version",
            "release_version",
            "year",
            "required_groups",
            "shapes",
            "total_output_column",
            "axis_lengths",
            "axis_hashes",
            "source_fingerprints",
            "backend_identity",
            "profile",
        }
        if set(payload) - fields - {"release_map_bindings"} or fields - set(payload):
            raise ReceiptContractError("output fields are missing or unknown")
        return cls(**payload)

    def __post_init__(self) -> None:
        artifact = _decode(self.artifact_name)
        if type(artifact) is not str or not artifact or artifact != artifact.strip() or any(ord(char) < 32 or ord(char) == 127 for char in artifact):
            raise ReceiptContractError("artifact_name must be a relative name")
        if artifact.startswith(("/", "\\", "~")) or "/" in artifact or "\\" in artifact or _SCHEME.match(artifact) or _EMBEDDED_SCHEME.search(artifact) or "?" in artifact or "#" in artifact or ":" in artifact:
            raise ReceiptContractError("artifact_name must be a safe relative name")
        path = PurePosixPath(artifact)
        if path.is_absolute() or artifact in {".", ".."} or artifact.startswith(("./", "../")) or ".." in artifact:
            raise ReceiptContractError("artifact_name must be a safe relative name")
        object.__setattr__(self, "artifact_name", artifact)
        object.__setattr__(self, "hdf5_sha256", _digest(self.hdf5_sha256, field="hdf5_sha256"))
        object.__setattr__(self, "schema_version", _identifier(self.schema_version, field="schema_version"))
        object.__setattr__(self, "release_version", _safe_text(self.release_version, field="output.release_version"))
        if type(self.year) is not int or self.year < 1:
            raise ReceiptContractError("output.year must be a positive native integer")
        object.__setattr__(self, "required_groups", tuple(self.required_groups))
        if self.required_groups != REQUIRED_GROUPS:
            raise ReceiptContractError("output.required_groups must contain the ten matrix groups in order")
        shapes = _mapping(self.shapes, field="output.shapes")
        checked_shapes: dict[str, tuple[int, int]] = {}
        for name, shape in shapes.items():
            if name not in REQUIRED_GROUPS or not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ReceiptContractError(f"output.shapes.{name} is invalid")
            if any(type(dim) is not int or dim < 1 for dim in shape):
                raise ReceiptContractError(f"output.shapes.{name} contains invalid dimensions")
            checked_shapes[name] = (shape[0], shape[1])
        if set(checked_shapes) != set(REQUIRED_GROUPS):
            raise ReceiptContractError("output.shapes must declare every matrix group")
        object.__setattr__(self, "shapes", checked_shapes)
        if self.total_output_column != "Total Output":
            raise ReceiptContractError("output.total_output_column must be 'Total Output'")
        axis_lengths = _mapping(self.axis_lengths, field="output.axis_lengths")
        if set(axis_lengths) != set(AXIS_NAMES) or any(type(value) is not int or value < 1 for value in axis_lengths.values()):
            raise ReceiptContractError("output.axis_lengths must declare positive lengths for every axis")
        object.__setattr__(self, "axis_lengths", {name: axis_lengths[name] for name in AXIS_NAMES})
        axis_hashes = _mapping(self.axis_hashes, field="output.axis_hashes")
        if set(axis_hashes) != set(AXIS_NAMES):
            raise ReceiptContractError("output.axis_hashes must declare every axis")
        object.__setattr__(self, "axis_hashes", {name: _digest(axis_hashes[name], field=f"output.axis_hashes.{name}") for name in AXIS_NAMES})
        source_fingerprints = _mapping(self.source_fingerprints, field="output.source_fingerprints")
        checked_sources: dict[str, str] = {}
        for name, value in source_fingerprints.items():
            checked_sources[_runtime_source_id(name, field="output.source_fingerprints key")] = _digest(value, field=f"output.source_fingerprints.{name}")
        object.__setattr__(self, "source_fingerprints", checked_sources)
        object.__setattr__(self, "backend_identity", _backend(self.backend_identity, field="output.backend_identity"))
        object.__setattr__(self, "profile", _profile(self.profile, field="output.profile"))
        bindings = self.release_map_bindings
        if bindings is None:
            if self.profile == "production":
                raise ReceiptContractError("production output requires release_map_bindings")
        else:
            object.__setattr__(
                self,
                "release_map_bindings",
                _validate_release_map_bindings(bindings, release_version=self.release_version),
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "artifact_name": self.artifact_name,
            "hdf5_sha256": self.hdf5_sha256,
            "schema_version": self.schema_version,
            "release_version": self.release_version,
            "year": self.year,
            "required_groups": list(self.required_groups),
            "shapes": {name: list(shape) for name, shape in self.shapes.items()},
            "total_output_column": self.total_output_column,
            "axis_lengths": dict(self.axis_lengths),
            "axis_hashes": dict(self.axis_hashes),
            "source_fingerprints": dict(self.source_fingerprints),
            "backend_identity": self.backend_identity,
            "profile": self.profile,
        }
        if self.release_map_bindings is not None:
            result["release_map_bindings"] = dict(self.release_map_bindings)
        return result


def _profile(value: Any, *, field: str) -> str:
    value = _decode(value)
    if type(value) is not str or value not in PUBLIC_PROFILES:
        raise ReceiptContractError(f"{field} must be 'production' or 'test-fixture'")
    return value


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    passed: bool
    required: bool
    reason: str
    message: str
    evidence: str
    observed: Any
    expected: Any
    tolerance: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="check.name"))
        status = _decode(self.status)
        if type(status) is not str or status not in CHECK_STATUSES:
            raise ReceiptContractError("check.status is invalid")
        object.__setattr__(self, "status", status)
        if type(self.passed) is not bool or type(self.required) is not bool:
            raise ReceiptContractError("check.passed and check.required must be native booleans")
        if (self.status == "passed" and not self.passed) or (self.status == "failed" and self.passed):
            raise ReceiptContractError("check.status and check.passed disagree")
        if self.status == "not_run" and (self.passed or self.required):
            raise ReceiptContractError("not_run checks must be optional and not passed")
        for field_name in ("reason", "message", "evidence"):
            text = _safe_text(getattr(self, field_name), field=f"check.{field_name}") if getattr(self, field_name) else ""
            if self.status == "not_run" and not text:
                raise ReceiptContractError("not_run checks require reason, message, and evidence")
            object.__setattr__(self, field_name, text)
        object.__setattr__(self, "observed", _safe_scalar(self.observed, field="check.observed"))
        object.__setattr__(self, "expected", _safe_scalar(self.expected, field="check.expected"))
        if self.tolerance is not None:
            if type(self.tolerance) not in {int, float} or not np.isfinite(self.tolerance) or self.tolerance < 0:
                raise ReceiptContractError("check.tolerance must be finite and non-negative")
            object.__setattr__(self, "tolerance", float(self.tolerance))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "required": self.required,
            "reason": self.reason,
            "message": self.message,
            "evidence": self.evidence,
            "observed": self.observed,
            "expected": self.expected,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class Fallback:
    name: str
    status: str
    reason: str
    backend: str | None = None
    source: str | None = None
    policy: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "status": self.status, "reason": self.reason}
        for field_name in ("backend", "source", "policy", "evidence"):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


class ReceiptDocument(dict):
    """Normalized path-free receipt mapping with typed accessors."""

    @property
    def output_identity(self) -> OutputIdentity:
        return OutputIdentity.from_mapping(self["output"])

    @property
    def output(self) -> OutputIdentity:
        return self.output_identity

    @property
    def checks(self) -> tuple[CheckResult, ...]:
        return tuple(_check_from_mapping(item, required_default=True) for item in self["checks"])

    @property
    def optional_checks(self) -> tuple[CheckResult, ...]:
        return tuple(_check_from_mapping(item, required_default=False) for item in self["optional_checks"])


def _source_from_mapping(value: Any) -> SourceIdentity:
    payload = _mapping(value, field="source record")
    fields = {"source_id", "publisher", "acquisition_url", "expected_format", "attribution", "reuse_condition", "version", "retrieved_at", "sha256"}
    if set(payload) != fields:
        raise ReceiptContractError("source record fields are missing or unknown")
    return SourceIdentity(**payload)


def _output_from_mapping(value: Any) -> OutputIdentity:
    return OutputIdentity.from_mapping(_mapping(value, field="output"))


def _check_from_mapping(value: Any, *, required_default: bool) -> CheckResult:
    payload = _mapping(value, field="check")
    fields = {"name", "status", "passed", "required", "reason", "message", "evidence", "observed", "expected", "tolerance"}
    if set(payload) != fields:
        raise ReceiptContractError("check fields are missing or unknown")
    result = CheckResult(**payload)
    if required_default and not result.required:
        raise ReceiptContractError("checks must contain required checks")
    if not required_default and result.required:
        raise ReceiptContractError("optional_checks must contain optional checks")
    return result


def _fallback_from_mapping(value: Any) -> Fallback:
    payload = _mapping(value, field="fallback")
    fields = {"name", "status", "reason", "backend", "source", "policy", "evidence"}
    if set(payload) - fields or not {"name", "status", "reason"}.issubset(payload):
        raise ReceiptContractError("fallback fields are missing or unknown")
    name = _identifier(payload["name"], field="fallback.name")
    status = _decode(payload["status"])
    if type(status) is not str or status not in FALLBACK_STATUSES:
        raise ReceiptContractError("fallback.status is invalid")
    reason = _safe_text(payload["reason"], field="fallback.reason")
    backend = payload.get("backend")
    if "backend" in payload:
        if backend is None:
            raise ReceiptContractError("fallback.backend is unknown")
        backend = _backend(backend, field="fallback.backend")
    source = payload.get("source")
    policy = payload.get("policy")
    if "source" in payload:
        if source is None:
            raise ReceiptContractError("fallback.source must not be null")
        source = _runtime_source_id(source, field="fallback.source")
    if "policy" in payload:
        if policy is None:
            raise ReceiptContractError("fallback.policy must not be null")
        policy = _identifier(policy, field="fallback.policy")
    evidence = payload.get("evidence")
    if "evidence" in payload:
        if evidence is None:
            raise ReceiptContractError("fallback.evidence must not be null")
        evidence = _safe_text(evidence, field="fallback.evidence")
    return Fallback(name, status, reason, backend, source, policy, evidence)


def _normalize(value: Mapping[str, Any]) -> ReceiptDocument:
    payload = _mapping(value, field="receipt")
    fields = {"receipt_schema", "release_version", "command_name", "configuration_fingerprint", "source_records", "output", "checks", "optional_checks", "backend_identity", "software_identity", "fallbacks"}
    if set(payload) != fields:
        raise ReceiptContractError("receipt fields are missing or unknown")
    if payload["receipt_schema"] != RECEIPT_SCHEMA_VERSION:
        raise ReceiptContractError("receipt schema version is unsupported")
    release = _safe_text(payload["release_version"], field="release_version")
    command = _identifier(payload["command_name"], field="command_name")
    if command not in COMMANDS:
        raise ReceiptContractError("command_name is invalid")
    config_fp = _digest(payload["configuration_fingerprint"], field="configuration_fingerprint")
    backend = _backend(payload["backend_identity"], field="backend_identity")
    software = _safe_text(payload["software_identity"], field="software_identity")
    source_values = payload["source_records"]
    if not isinstance(source_values, list):
        raise ReceiptContractError("source_records must be an array")
    sources = tuple(_source_from_mapping(item) for item in source_values)
    if len({source.source_id for source in sources}) != len(sources):
        raise ReceiptContractError("source_records contains duplicate source IDs")
    output = _output_from_mapping(payload["output"])
    if output.release_version != release or output.backend_identity != backend:
        raise ReceiptContractError("receipt and output identity fields disagree")
    source_fingerprints = {source.source_id: source.sha256 for source in sources}
    if source_fingerprints != output.source_fingerprints:
        raise ReceiptContractError("source_records do not bind output source_fingerprints")
    if not isinstance(payload["checks"], list) or not payload["checks"]:
        raise ReceiptContractError("checks must be a nonempty array")
    if not isinstance(payload["optional_checks"], list) or not isinstance(payload["fallbacks"], list):
        raise ReceiptContractError("optional_checks and fallbacks must be arrays")
    checks = tuple(_check_from_mapping(item, required_default=True) for item in payload["checks"])
    optional = tuple(_check_from_mapping(item, required_default=False) for item in payload["optional_checks"])
    if any(check.status != "passed" for check in checks):
        raise ReceiptContractError("required checks must all pass")
    if len({check.name for check in checks}) != len(checks) or len({check.name for check in optional}) != len(optional):
        raise ReceiptContractError("receipt checks must have unique names")
    if {check.name for check in checks} & {check.name for check in optional}:
        raise ReceiptContractError("a check cannot be both required and optional")
    if output.profile == "production" and output.release_map_bindings is not None:
        missing_map_checks = REQUIRED_RELEASE_MAP_CHECKS - {check.name for check in checks}
        if missing_map_checks:
            raise ReceiptContractError(
                f"production receipt is missing required release-map checks: {sorted(missing_map_checks)}"
            )
    fallbacks = tuple(_fallback_from_mapping(item) for item in payload["fallbacks"])
    normalized = ReceiptDocument(
        {
            "receipt_schema": RECEIPT_SCHEMA_VERSION,
            "release_version": release,
            "command_name": command,
            "configuration_fingerprint": config_fp,
            "source_records": [source.to_dict() for source in sources],
            "output": output.to_dict(),
            "checks": [check.to_dict() for check in checks],
            "optional_checks": [check.to_dict() for check in optional],
            "backend_identity": backend,
            "software_identity": software,
            "fallbacks": [fallback.to_dict() for fallback in fallbacks],
        }
    )
    return normalized


def read_rebuild_receipt(path: str | Path) -> ReceiptDocument:
    """Read and normalize one strict receipt JSON document."""

    try:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptContractError("cannot read a valid rebuild receipt") from exc
    try:
        return _normalize(payload)
    except ReceiptContractError:
        raise
    except Exception as exc:  # defensive public boundary
        raise ReceiptContractError("rebuild receipt is malformed") from exc


def parse_rebuild_receipt(payload: Mapping[str, Any]) -> ReceiptDocument:
    """Normalize an already-decoded receipt mapping without file access."""

    try:
        return _normalize(payload)
    except ReceiptContractError:
        raise
    except Exception as exc:  # defensive public boundary
        raise ReceiptContractError("rebuild receipt is malformed") from exc


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return deterministic JSON for receipt/configuration fingerprints."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AXIS_NAMES",
    "COMMANDS",
    "CheckResult",
    "Fallback",
    "MATRIX_AXIS_ROLES",
    "OutputIdentity",
    "PUBLIC_BACKENDS",
    "PUBLIC_OIL_TYPES",
    "PUBLIC_PARENT_IDS",
    "PUBLIC_PROFILES",
    "REQUIRED_RELEASE_MAP_CHECKS",
    "RECEIPT_SCHEMA_VERSION",
    "REQUIRED_GROUPS",
    "ReceiptDocument",
    "SourceIdentity",
    "canonical_json",
    "file_sha256",
    "parse_rebuild_receipt",
    "read_rebuild_receipt",
]
