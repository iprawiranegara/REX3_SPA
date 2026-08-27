"""Public input and configuration boundary for ``oil-split-spa``."""

from .config import AnalysisConfig, validate_analysis_config
from .errors import (
    ConfigError,
    ConfigurationError,
    InputContractError,
    OilSplitError,
    ReceiptContractError,
    ReceiptError,
)
from .preflight import ValidatedInput, preflight_input
from .run import RunArtifact, RunResult, run_analysis
from .spa import StructuralPath, StructuralPathResult, forward_structural_paths
from .receipt import (
    AXIS_NAMES,
    COMMANDS,
    MATRIX_AXIS_ROLES,
    PUBLIC_BACKENDS,
    PUBLIC_OIL_TYPES,
    PUBLIC_PARENT_IDS,
    PUBLIC_PROFILES,
    RECEIPT_SCHEMA_VERSION,
    REQUIRED_GROUPS,
    CheckResult,
    Fallback,
    OutputIdentity,
    ReceiptDocument,
    SourceIdentity,
    parse_rebuild_receipt,
    read_rebuild_receipt,
)

__all__ = [
    "AXIS_NAMES",
    "AnalysisConfig",
    "COMMANDS",
    "CheckResult",
    "ConfigError",
    "ConfigurationError",
    "Fallback",
    "InputContractError",
    "MATRIX_AXIS_ROLES",
    "OilSplitError",
    "OutputIdentity",
    "PUBLIC_BACKENDS",
    "PUBLIC_OIL_TYPES",
    "PUBLIC_PARENT_IDS",
    "PUBLIC_PROFILES",
    "RECEIPT_SCHEMA_VERSION",
    "REQUIRED_GROUPS",
    "ReceiptContractError",
    "ReceiptDocument",
    "ReceiptError",
    "RunArtifact",
    "RunResult",
    "SourceIdentity",
    "StructuralPath",
    "StructuralPathResult",
    "ValidatedInput",
    "forward_structural_paths",
    "parse_rebuild_receipt",
    "preflight_input",
    "read_rebuild_receipt",
    "run_analysis",
    "validate_analysis_config",
]
