"""Codex rollout scanner implementation split from :mod:`loopmath.graph.codexio`."""
from __future__ import annotations

import json
import os
import re
import shlex
from collections import Counter
from pathlib import Path

from . import bashwrites
from .scan import iter_jsonl

_ORIGIN_FIELDS = ("originator", "source", "cwd", "cli_version")
_SHELL_FUNCTIONS = {"shell", "shell_command", "exec_command", "container.exec", "local_shell"}
_SHELL_WRAPPERS = {"bash", "zsh", "sh", "fish", "dash", "/bin/bash", "/bin/zsh", "/bin/sh"}
_CALL_TYPES = ("function_call", "custom_tool_call", "local_shell_call")
_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output", "local_shell_call_output")
_PATCH_HEADER = re.compile(r"^\*\*\* (Add File|Update File|Delete File|Move to): (.+?)\s*$", re.MULTILINE)
_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"
_SUCCESS_BANNER = "Success. Updated the following files:"
_LISTED_PATH = re.compile(r"^[AMD] (.+?)\s*$", re.MULTILINE)
_EXIT_CODE = re.compile(r"^Process exited with code (\d+)", re.MULTILINE)
# Exit codes an `exec` script's output carries for the commands it wrapped: the
# `Process exited with code N` line of a nested result, the result object the script
# printed (`{"chunk_id": ..., "exit_code": N, ...}`), or an `exit_code=N` line.
_EMBEDDED_EXIT = re.compile(r'^Process exited with code (\d+)|^\{[^\n]*"exit_code":\s*(\d+)|^exit_code=(\d+)\s*$', re.MULTILINE)
_JS_EXEC = re.compile(r"\btools\.exec_command\s*\(")
_JS_PATCH = re.compile(r"\btools\.apply_patch\s*\(")
_JS_IDENT = re.compile(r"^[A-Za-z_$][\w$]*$")
_JS_RAW = re.compile(r"^String\.raw\s*(?=`)")
_JS_BINDING = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(String\.raw\s*)?(?=[\"'`])")
_BARE_KEY = re.compile(r'([{,]\s*)([A-Za-z_$][\w$]*)(\s*:)')
_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_MAX_SAMPLES = 3


def _abs(path: str, cwd: str | None) -> tuple[str, bool]:
    """`(path, relative)`: the path resolved against `cwd`, or unchanged and flagged
    when there is no cwd to resolve it against."""
    if path.startswith("/"):
        return path, False
    if path.startswith("~/"):
        return str(Path(path).expanduser()), False
    if cwd:
        return os.path.normpath(str(Path(cwd) / path)), False
    return path, True


def _entry(ts: str | None, path: str, tier: str, how: str, cwd: str | None) -> dict:
    p, rel = _abs(path, cwd)
    d = {"ts": ts, "path": p, "tier": tier, "how": how}
    if rel:
        d["relative"] = True
    if "$" in path:
        d["unresolved"] = True
    return d


def patch_paths(patch: str) -> tuple[list[tuple[str, str]], int]:
    """`([(path, how), ...], n_deletes)` from the header lines of an apply_patch
    envelope. Reads only the `*** ... File:` / `*** Move to:` lines."""
    out: list[tuple[str, str]] = []
    deletes = 0
    for m in _PATCH_HEADER.finditer(patch):
        kind, path = m.group(1), m.group(2)
        if kind == "Add File":
            out.append((path, "apply_patch add"))
        elif kind == "Update File":
            out.append((path, "apply_patch update"))
        elif kind == "Move to":
            out.append((path, "apply_patch move"))
        else:
            deletes += 1
    return out, deletes


def patch_artifact_facts(patch: str) -> list[dict]:
    """Per-path operations and recorded diff counts from an apply-patch envelope.

    This is the scanner's private measurement side channel.  ``patch_paths`` stays
    the public write-record parser: callers which only need paths keep receiving
    exactly the old records.  Counts here come only from ``+`` and ``-`` body lines
    that the rollout itself recorded; context and patch control lines do not count.
    """
    matches = list(_PATCH_HEADER.finditer(patch))
    operation_paths: list[str] = []
    for match in matches:
        path = match.group(2)
        if path not in operation_paths:
            operation_paths.append(path)

    out: list[dict] = []
    for index, match in enumerate(matches):
        kind, path = match.group(1), match.group(2)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(patch)
        body = patch[match.end():end]
        added = sum(line.startswith("+") for line in body.splitlines())
        removed = sum(line.startswith("-") for line in body.splitlines())
        operation = {
            "Add File": "write",
            "Update File": "edit",
            "Delete File": "delete",
            "Move to": "move",
        }[kind]
        fact: dict = {
            "path": path,
            "operation": operation,
            "operation_paths": list(operation_paths),
        }
        # Add and update sections record their complete patch-line counts, including
        # real zeroes.  A Delete header records the fate but not the deleted body;
        # Move-to is a path operation, not another diff body.
        if kind in {"Add File", "Update File"}:
            fact.update({"lines_added": added, "lines_removed": removed})
        elif kind == "Move to" and added + removed:
            # The v0.1 extractor records the move but does not attribute the
            # following diff body to either path. The scanner surfaces that
            # accepted loss without changing the existing measurement.
            fact["_unmodeled_diff_lines"] = added + removed
        out.append(fact)
    return out


# --- JS snippets (the custom `exec` tool) -------------------------------------------


def _balanced_argument(text: str, start: int) -> str | None:
    """The text between the `(` at `start - 1` and its matching `)`, honouring
    string literals; None when unbalanced."""
    depth = 1
    i = start
    quote: str | None = None
    n = len(text)
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _quote_bare_keys(text: str) -> str:
    """Quote unquoted object keys outside string literals (`{cmd:"x"}` to `{"cmd":"x"}`)."""
    out: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    while i < n:
        c = text[i]
        if quote:
            if c == "\\" and i + 1 < n:
                nxt = text[i + 1]
                out.append(nxt if nxt in "'`" else c + nxt)
                i += 2
                continue
            if c == quote:
                out.append('"')
                quote = None
            elif c == '"':
                out.append('\\"')
            else:
                out.append(c)
            i += 1
            continue
        if c in "\"'`":
            quote = c
            out.append('"' if c != '"' else c)
            i += 1
            continue
        m = _BARE_KEY.match(text, i)
        if m:
            out.append(m.group(1) + '"' + m.group(2) + '"' + m.group(3))
            i = m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _js_object(text: str) -> dict | None:
    """A JS object literal as a dict: JSON first, then with bare keys quoted."""
    text = text.strip().rstrip(",")
    for candidate in (text, _quote_bare_keys(text)):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _js_string(text: str) -> str | None:
    """The value of one JS string literal (`"..."`, `'...'`, a template literal
    without substitutions, or `String.raw` applied to one); None when `text` is not
    exactly one literal."""
    text = text.strip()
    raw = _JS_RAW.match(text)
    if raw:
        text = text[raw.end():]
    if len(text) < 2 or text[0] not in "\"'`" or text[-1] != text[0]:
        return None
    quote, body = text[0], text[1:-1]
    if raw:
        return None if (quote != "`" or "${" in body) else body
    if quote == '"':
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, str):
            return value
    if quote == "`" and "${" in body:
        return None
    out: list[str] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == "\\" and i + 1 < n:
            nxt = body[i + 1]
            if nxt == "u" and i + 5 < n:
                try:
                    out.append(chr(int(body[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(_JS_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        if c == quote:
            return None
        out.append(c)
        i += 1
    return "".join(out)


def _js_bindings(script: str, before: int) -> dict[str, str]:
    """Executable string-literal bindings ending before one tool call.

    Iteration follows source order, so a nearer preceding declaration of the same name
    replaces an earlier one. Declarations in strings, comments, or after the call cannot
    supply the call's argument.
    """
    out: dict[str, str] = {}
    code = _js_code_map(script)
    for m in _JS_BINDING.finditer(script):
        if m.start() >= before or not code[m.start()]:
            continue
        end = _literal_end(script, m.end())
        if end is None or end > before:
            continue
        value = _js_string(script[m.start(2) if m.group(2) else m.end():end])
        if value is not None:
            out[m.group(1)] = value
    return out


def _literal_end(text: str, start: int) -> int | None:
    """Index just past the string literal starting at `start`, or None."""
    quote = text[start]
    i = start + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return None


_JS_OPEN = re.compile(r"[\"'`]|//|/\*")
_JS_LINE_END = re.compile(r"[\r\n]")
_JS_STRING_BODY = {q: re.compile(r"(?:[^\\%s]++|\\[\s\S])*+" % q) for q in "\"'`"}
_code_map_memo: dict[str, bytearray] = {}


def _js_code_map(script: str) -> bytearray:
    """One byte per character, true only for JavaScript code.

    The custom exec payload is source text, so a regex match inside a quoted example
    or a comment is not a tool call. Template literals are treated as literals in
    full; Codex wrappers use them for patch text, not executable interpolations.

    Regex jumps from one string or comment opening to its end; the character loop it
    replaced is the reference in tests/test_graph_codexio.py. The last map is kept,
    since one script is asked about exec calls, then about patches.
    """
    hit = _code_map_memo.get(script)
    if hit is not None:
        return hit
    n = len(script)
    code = bytearray(b"\x01") * n
    i = 0
    while True:
        m = _JS_OPEN.search(script, i)
        if m is None:
            break
        start = m.start()
        opening = m.group()
        if opening == "//":
            nl = _JS_LINE_END.search(script, start + 2)
            end = nl.end() if nl else n
        elif opening == "/*":
            close = script.find("*/", start + 2)
            end = close + 2 if close >= 0 else n
        else:
            body_end = _JS_STRING_BODY[opening].match(script, start + 1).end()
            end = min(body_end + 1, n)  # the closing quote, or a lone final backslash
        code[start:end] = bytes(end - start)
        i = end
    _code_map_memo.clear()
    _code_map_memo[script] = code
    return code


def _code_matches(pattern: re.Pattern, script: str):
    """Regex matches whose first character is executable JS source. The code map is
    built only for a script the pattern matches at all."""
    matches = list(pattern.finditer(script))
    if not matches:
        return iter(())
    code = _js_code_map(script)
    return (m for m in matches if code[m.start()])


def exec_commands_from_js(script: str) -> tuple[list[dict], int]:
    """`([{cmd, workdir}, ...], n_unparsed)` from a codex `exec` tool script, one
    per `tools.exec_command({...})` call."""
    calls: list[dict] = []
    unparsed = 0
    for m in _code_matches(_JS_EXEC, script):
        arg = _balanced_argument(script, m.end())
        obj = _js_object(arg) if arg is not None else None
        if obj is None:
            unparsed += 1
            continue
        calls.append(obj)
    return calls, unparsed


def patches_from_js(script: str) -> tuple[list[str], int]:
    """`([patch, ...], n_unparsed)` from a codex `exec` tool script, one per
    `tools.apply_patch(x)` call whose patch text is in the script: `x` a string
    literal, an object literal with `input`/`patch`, or a name bound by
    `const|let|var` to a string literal. Anything else (a command's output, a
    computed string) is counted as unparsed."""
    patches: list[str] = []
    unparsed = 0
    for m in _code_matches(_JS_PATCH, script):
        arg = _balanced_argument(script, m.end())
        text = arg.strip().rstrip(",") if arg is not None else ""
        value: str | None = None
        if text[:1] in "\"'`" or _JS_RAW.match(text):
            value = _js_string(text)
        elif text[:1] == "{":
            obj = _js_object(text)
            if obj:
                for key in ("input", "patch"):
                    if isinstance(obj.get(key), str):
                        value = obj[key]
                        break
        elif _JS_IDENT.match(text):
            value = _js_bindings(script, m.start()).get(text)
        if value is None:
            unparsed += 1
            continue
        patches.append(value)
    return patches, unparsed


# --- tool call payloads ----------------------------------------------------------------



__all__ = [name for name in globals() if not name.startswith("__")]
