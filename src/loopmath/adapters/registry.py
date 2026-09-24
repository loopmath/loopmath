"""Registration and lazy discovery for ADE adapters."""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import TypeVar

from .base import Adapter


_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_REGISTRY: dict[str, type[Adapter]] = {}
_AdapterType = TypeVar("_AdapterType", bound=type[Adapter])


class UnknownAdapterError(LookupError):
    """Raised when no registered adapter has the requested canonical name."""

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = available
        choices = ", ".join(available) if available else "none"
        super().__init__(f"unknown adapter {name!r}; available adapters: {choices}")


def register(adapter_type: _AdapterType) -> _AdapterType:
    """Register an :class:`Adapter` subclass by its canonical ``name``.

    Use this as ``@register`` on the adapter class in ``<name>.py``. A dash in
    a public name maps to an underscore in the module name. Re-registering the
    same class is idempotent; a different class may never replace a name.
    """

    if not isinstance(adapter_type, type) or not issubclass(adapter_type, Adapter):
        raise TypeError("registered adapters must be Adapter subclasses")
    name = getattr(adapter_type, "name", None)
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(
            "adapter name must be lowercase kebab-case beginning with a letter"
        )
    previous = _REGISTRY.get(name)
    if previous is not None and previous is not adapter_type:
        raise ValueError(f"adapter {name!r} is already registered by {previous.__name__}")
    _REGISTRY[name] = adapter_type
    return adapter_type


def _import_for_name(name: str) -> None:
    if _NAME.fullmatch(name) is None:
        return
    module_name = f"{__package__}.{name.replace('-', '_')}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # A missing adapter module means the name is unknown. A missing import
        # *inside* an adapter is a real packaging error and must stay visible.
        if exc.name != module_name:
            raise


def _import_all() -> None:
    package = importlib.import_module(__package__)
    for module in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if module.name.startswith("_") or module.name in {"base", "registry"}:
            continue
        importlib.import_module(f"{__package__}.{module.name}")


def lookup(name: str) -> type[Adapter]:
    """Return the registered adapter class for ``name``, loading its module."""

    if name not in _REGISTRY:
        _import_for_name(name)
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownAdapterError(name, list_adapters()) from None


def list_adapters() -> tuple[str, ...]:
    """Load adapter modules and return their canonical names in sorted order."""

    _import_all()
    return tuple(sorted(_REGISTRY))
