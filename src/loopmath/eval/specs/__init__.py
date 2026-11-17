"""The catalog of `eval simulate` specs.

    from loopmath.eval import specs
    specs.names()                      # every catalog name
    specs.get("linear")                # a spec (`loopmath.eval-sim/1`) as a plain dict
    specs.get("linear", inputs=12)     # the same design on fewer inputs, for a quick test
    specs.title("linear")              # one line on what the spec is for

A spec is a plain mapping, so a test can change any member before it simulates. The specs are built in
`catalog.py`. They are Python, not data files: a data file would need a packaging entry outside
`loopmath.eval`. All of it is synthetic.
"""

from __future__ import annotations

from .catalog import CATALOG


def names() -> list[str]:
    """The catalog names, the most basic specs first."""
    return list(CATALOG)


def title(name: str) -> str:
    """One line on what a catalog spec is for."""
    return CATALOG[name][0]


def get(name: str, inputs: int | None = None) -> dict:
    """A catalog spec as a new plain dict. `inputs` overrides the number of inputs. KeyError when unknown."""
    if name not in CATALOG:
        raise KeyError(f"unknown catalog spec {name!r}; the catalog holds {', '.join(CATALOG)}")
    text, build = CATALOG[name]
    spec = build()
    spec["name"], spec["title"] = name, text
    if inputs is not None:
        spec["inputs"] = inputs
    return spec
