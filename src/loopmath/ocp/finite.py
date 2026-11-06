"""Given JSON read as JSON: `loads(text)` is `json.loads` refusing what JSON or a float cannot hold.

`NaN`, `Infinity` and `-Infinity` are not JSON numbers (RFC 8259); Python's reader takes them only as an
extension. A number past what a float holds (`1e400`) is JSON but reads as an infinity, which a writer gives back
as `Infinity`. `loads` raises `NotFinite`, a ValueError naming each such place in document order, three at most
and then `and N more`: `stages.1.cost.usd: NaN is not a JSON number`, `run.ext.x: 1e400 is past what a float
holds` with the literal as written. A member given twice is read as `json.loads` reads it, the later one kept, but
such a value in the member it replaces is refused too.

A place is written as the rows checker writes one: member names and list indexes joined with `.`. A member name
that is empty, all digits, or holds `.`, `[`, `]`, `"`, white space or a character that does not print is
written as a JSON string in brackets, `run.ext["dev.example"].x`. A value at the top is `the document`.

The readers of given files (run documents, rows, task and configuration files, flow documents) use it; stored
runs are read as before.
"""

from __future__ import annotations

import json
import math
from typing import Any


class NotFinite(ValueError):
    """A JSON text holding NaN, an infinity or a float past range: `places` holds every place with its words, in
    document order; the message names the first three."""

    def __init__(self, places: list[str]) -> None:
        super().__init__("; ".join(places[:3]) + (f" and {len(places) - 3} more" if len(places) > 3 else ""))
        self.places = list(places)


class _Refused:
    """What a hook gives in place of a value JSON or a float cannot hold."""

    __slots__ = ("words",)

    def __init__(self, words: str) -> None:
        self.words = words


class _Members(list):
    """An object's members as the text gives them, a later duplicate beside the one it replaces."""


def _step(name: str) -> str:
    if (name and not name.isdigit() and name.isprintable()
            and not any(ch in '.[]"' or ch.isspace() for ch in name)):
        return name
    return "[" + json.dumps(name) + "]"


def _place(path: tuple[Any, ...]) -> str:
    if not path:
        return "the document"
    out = ""
    for step in path:
        text = str(step) if isinstance(step, int) else _step(step)
        out += text if not out or text.startswith("[") else "." + text
    return out


def loads(text: str) -> Any:
    """`json.loads(text)`, raising NotFinite for NaN, Infinity, -Infinity and a float past range."""
    fired = []

    def constant(name: str) -> _Refused:
        fired.append(name)
        return _Refused(f"{name} is not a JSON number")

    def number(literal: str) -> Any:
        value = float(literal)
        if math.isfinite(value):
            return value
        fired.append(literal)
        return _Refused(f"{literal} is past what a float holds")

    value = json.loads(text, parse_constant=constant, parse_float=number)
    if not fired:
        return value
    # read again with every member kept, so a value a later duplicate member replaced is named too
    tree = json.loads(text, parse_constant=constant, parse_float=number, object_pairs_hook=_Members)
    places, stack = [], [((), tree)]
    while stack:  # once, in document order, with no recursion
        path, here = stack.pop()
        if isinstance(here, _Refused):
            places.append(f"{_place(path)}: {here.words}")
        elif isinstance(here, _Members):
            stack += [((*path, k), v) for k, v in reversed(here)]
        elif isinstance(here, list):
            stack += [((*path, i), here[i]) for i in range(len(here) - 1, -1, -1)]
    raise NotFinite(places)
