"""Shared shell parsing primitives for Bash artifact recognition."""

from __future__ import annotations

import os
import re
import shlex
from collections import Counter
from pathlib import PurePath

CONTROL = {"&&", "||", ";", "|", "&"}
REDIRECTS = {">", ">>"}
HEREDOC_RE = re.compile(
    r"(?<!<)(?P<operator><<-?)\s*(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|(?P<plain>[^\s;&|<>]+))"
)
STDERR_RE = re.compile(r"(?<!\S)(?:[2-9]|[1-9][0-9]+)>>?\s*(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)")
VARIABLE_RE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\}|\()|`")
SUBSTITUTION_RE = re.compile(r"\$__LOOPMATH_SUB_(\d+)__")


def exclude(exclusions: Counter | None, reason: str, count: int = 1) -> None:
    if exclusions is not None and count:
        exclusions[reason] += count


def heredoc_positions(line: str) -> list[int]:
    positions: list[int] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            break
        elif line.startswith("<<", index) and not line.startswith("<<<", index):
            positions.append(index)
            index += 2
            continue
        index += 1
    return positions


def without_heredoc_bodies(command: str, exclusions: Counter | None = None) -> str:
    """Keep heredoc command lines but remove their arbitrary body text."""
    kept: list[str] = []
    pending: list[tuple[str, bool]] = []
    for line in command.splitlines():
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                pending.pop(0)
            continue
        kept.append(line)
        for position in heredoc_positions(line):
            match = HEREDOC_RE.match(line, position)
            if match is None:
                exclude(exclusions, "unparsable_heredoc")
                continue
            delimiter = match.group("single") or match.group("double") or match.group("plain")
            pending.append((delimiter, match.group("operator") == "<<-"))
    if pending:
        exclude(exclusions, "unparsable_heredoc", len(pending))
    return "\n".join(kept)


def tokens(command: str, exclusions: Counter | None = None, *, prepared: bool = False, count_redirects: bool = True) -> list[str]:
    if not prepared:
        command = without_heredoc_bodies(command, exclusions)
    excluded_redirects = len(STDERR_RE.findall(command))
    if count_redirects:
        exclude(exclusions, "excluded_redirect", excluded_redirects)
    command = STDERR_RE.sub(" ", command)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        tokens: list[str] = []
        for line in command.splitlines():
            try:
                lexer = shlex.shlex(line, posix=True, punctuation_chars="|&;<>")
                lexer.whitespace_split = True
                lexer.commenters = ""
                tokens.extend(lexer)
                tokens.append(";")
            except ValueError:
                exclude(exclusions, "tokenize_error")
        return tokens


def pull_substitutions(command: str, exclusions: Counter | None = None) -> tuple[str, list[str]]:
    """Remove command substitutions and return their bodies for isolated scans."""
    outer: list[str] = []
    inner: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\":
            outer.append(command[index : index + 2])
            index += 2
            continue
        if quote == "'":
            outer.append(char)
            if char == "'":
                quote = None
            index += 1
            continue
        if char == "'" and quote is None:
            quote = "'"
            outer.append(char)
            index += 1
            continue
        if char == '"':
            quote = None if quote == '"' else '"'
            outer.append(char)
            index += 1
            continue
        if command.startswith("$(", index) and not command.startswith("$((", index):
            depth = 1
            cursor = index + 2
            nested_quote: str | None = None
            while cursor < len(command) and depth:
                current = command[cursor]
                if current == "\\":
                    cursor += 2
                    continue
                if nested_quote == "'":
                    if current == "'":
                        nested_quote = None
                elif current == "'":
                    nested_quote = "'"
                elif current == '"':
                    nested_quote = None if nested_quote == '"' else '"'
                elif nested_quote is None and current == "(":
                    depth += 1
                elif nested_quote is None and current == ")":
                    depth -= 1
                cursor += 1
            if depth:
                exclude(exclusions, "unparsable_substitution")
                break
            inner.append(command[index + 2 : cursor - 1])
            outer.append(f"$__LOOPMATH_SUB_{len(inner) - 1}__")
            index = cursor
            continue
        outer.append(char)
        index += 1
    return "".join(outer), inner


def restore_substitutions(token: str, substitutions: list[str]) -> str:
    return SUBSTITUTION_RE.sub(lambda match: f"$({substitutions[int(match.group(1))]})", token)


def segments(tokens_: list[str]) -> list[list[str]]:
    result: list[list[str]] = []
    current: list[str] = []
    for token in tokens_:
        if token in CONTROL:
            if current:
                result.append(current)
                current = []
        else:
            current.append(token)
    if current:
        result.append(current)
    return result


def record(path: str, ts: str | None, cwd: str | None, how: str, *, exclusions: Counter | None = None, **flags: bool) -> dict | None:
    if not path or path == "/dev/null" or path.startswith("&"):
        exclude(exclusions, "excluded_path")
        return None
    unresolved = bool(VARIABLE_RE.search(path)) or path == "~" or path.startswith("~/")
    relative = not PurePath(path).is_absolute()
    if relative and cwd and not unresolved:
        path = os.path.normpath(os.path.join(cwd, path))
        relative = False
    item: dict = {"ts": ts, "path": path, "tier": "heuristic", "how": how}
    if relative and not cwd:
        item["relative"] = True
    if unresolved:
        item["unresolved"] = True
    item.update(flags)
    return item


def command_index(segment: list[str]) -> int | None:
    for index, token in enumerate(segment):
        if "=" in token and not token.startswith(("=", "-")):
            name, _equals, _value = token.partition("=")
            if name.replace("_", "a").isalnum() and not name[0].isdigit():
                continue
        return index
    return None


def plain_args(segment: list[str], start: int) -> list[str]:
    args: list[str] = []
    skip = False
    for token in segment[start + 1 :]:
        if skip:
            skip = False
            continue
        if token in {">", ">>", "<", "<<", "<<-"}:
            skip = True
            continue
        args.append(token)
    return args


def update_cwd(args: list[str], cwd: str | None) -> str | None:
    if not args or VARIABLE_RE.search(args[0]) or args[0].startswith("~"):
        return None
    if PurePath(args[0]).is_absolute():
        return os.path.normpath(args[0])
    return os.path.normpath(os.path.join(cwd, args[0])) if cwd else None
