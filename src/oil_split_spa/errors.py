"""Public error types for the oil-split application boundary."""

from __future__ import annotations


class OilSplitError(Exception):
    """Base class for public application errors."""


class ReceiptContractError(OilSplitError):
    """A rebuild receipt is malformed or violates the public contract."""


class InputContractError(OilSplitError):
    """The selected HDF5 and receipt do not satisfy the input contract."""


class ConfigurationError(OilSplitError):
    """An analysis configuration is malformed or outside the public scope."""


# Short names keep the public API readable while retaining one canonical class
# for each failure category.
ReceiptError = ReceiptContractError
ConfigError = ConfigurationError


__all__ = [
    "OilSplitError",
    "ReceiptContractError",
    "ReceiptError",
    "InputContractError",
    "ConfigurationError",
    "ConfigError",
]
