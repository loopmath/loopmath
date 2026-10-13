"""ADE-to-OCP adapter contract and registry.

An adapter lives in one module named for its ADE and registers one class::

    from loopmath.adapters import Adapter, Selection, register

    @register
    class ExampleAdapter(Adapter):
        name = "example"
        ...

``lookup`` returns the class, not an instance. ``list_adapters`` returns a
sorted tuple of canonical names and imports each adapter module so decorator
registration has run.
"""

from .base import (
    Adapter,
    AdapterServices,
    PricingResult,
    Selection,
    Usage,
    VENDOR_SESSION_REASONS,
    VendorSession,
    VendorSessionReason,
    VendorSessionResult,
)
from .registry import UnknownAdapterError, list_adapters, lookup, register

__all__ = [
    "Adapter",
    "AdapterServices",
    "PricingResult",
    "Selection",
    "Usage",
    "UnknownAdapterError",
    "VENDOR_SESSION_REASONS",
    "VendorSession",
    "VendorSessionReason",
    "VendorSessionResult",
    "list_adapters",
    "lookup",
    "register",
]
