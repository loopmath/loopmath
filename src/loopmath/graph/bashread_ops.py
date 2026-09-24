"""Read recognition built on :mod:`loopmath.graph.bashparse`."""

from __future__ import annotations

import os
import re
from collections import Counter

from . import bashparse

GREP_LONG_VALUE_OPTIONS = {
    "after-context", "before-context", "binary-files", "colors", "context", "context-separator", "dfa-size-limit", "devices", "directories", "encoding", "engine", "exclude", "exclude-dir", "exclude-from", "field-context-separator", "field-match-separator", "glob", "group-separator", "iglob", "ignore-file", "include", "label", "max-columns", "max-count", "max-depth", "max-filesize", "path-separator", "pre", "pre-glob", "regex-size-limit", "replace", "sort", "sortr", "threads", "type", "type-add", "type-not",
}
GREP_SHORT_VALUE_OPTIONS = set("ABCDdefm")
RG_SHORT_VALUE_OPTIONS = set("ABCEMTefgjmrt")
PYTEST_VALUE_OPTIONS = {"-c", "-k", "-m", "-n", "-o", "-p", "--basetemp", "--confcutdir", "--deselect", "--ignore", "--ignore-glob", "--maxfail", "--override-ini", "--rootdir", "--tb"}


def optionless(args: list[str], options_with_values: set[str]) -> list[str]:
    result: list[str] = []
    skip = False
    after_options = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "--":
            after_options = True
            continue
        if not after_options and arg in options_with_values:
            skip = True
            continue
        if not after_options and arg.startswith("-"):
            continue
        result.append(arg)
    return result


def grep_paths(args: list[str], name: str) -> list[str]:
    paths: list[str] = []
    have_pattern = False
    after_options = False
    short_values = RG_SHORT_VALUE_OPTIONS if name == "rg" else GREP_SHORT_VALUE_OPTIONS
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if after_options or arg == "-" or not arg.startswith("-"):
            if arg != "-":
                if not have_pattern:
                    have_pattern = True
                else:
                    paths.append(arg)
            continue
        if arg == "--":
            after_options = True
            continue
        if arg.startswith("--"):
            option, equals, value = arg[2:].partition("=")
            if option in {"regexp", "file"}:
                if not equals and index < len(args):
                    value = args[index]
                    index += 1
                have_pattern = True
                if option == "file" and value and value != "-":
                    paths.append(value)
            elif option in GREP_LONG_VALUE_OPTIONS and not equals and index < len(args):
                index += 1
            continue
        cluster = arg[1:]
        cursor = 0
        while cursor < len(cluster):
            option = cluster[cursor]
            cursor += 1
            if option not in short_values:
                continue
            value = cluster[cursor:]
            cursor = len(cluster)
            if not value and index < len(args):
                value = args[index]
                index += 1
            if option in {"e", "f"}:
                have_pattern = True
                if option == "f" and value and value != "-":
                    paths.append(value)
    return paths


def pytest_paths(args: list[str]) -> list[str]:
    paths: list[str] = []
    index = 0
    after_options = False
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--":
            after_options = True
            continue
        if not after_options and arg in PYTEST_VALUE_OPTIONS:
            index += 1
            continue
        if not after_options and arg.startswith("--") and "=" in arg:
            continue
        if not after_options and arg.startswith("-"):
            continue
        path = arg.split("::", 1)[0]
        if path:
            paths.append(path)
    return paths


def reads_walk(command: str, ts: str | None, cwd: str | None, exclusions: Counter | None) -> list[dict]:
    reads: list[dict] = []
    prepared = bashparse.without_heredoc_bodies(command)
    outer, substitutions = bashparse.pull_substitutions(prepared)
    active_cwd = cwd
    for segment in bashparse.segments(bashparse.tokens(outer, prepared=True, count_redirects=False)):
        plain_segment: list[str] = []
        for token in segment:
            for match in bashparse.SUBSTITUTION_RE.finditer(token):
                reads.extend(reads_walk(substitutions[int(match.group(1))], ts, active_cwd, exclusions))
            if not bashparse.SUBSTITUTION_RE.fullmatch(token):
                plain_segment.append(bashparse.restore_substitutions(token, substitutions))
        command_at = bashparse.command_index(plain_segment)
        if command_at is None:
            continue
        name = os.path.basename(plain_segment[command_at])
        args = bashparse.plain_args(plain_segment, command_at)
        paths: list[str] = []
        if name == "cd":
            active_cwd = bashparse.update_cwd(args, active_cwd)
            continue
        if name == "cat" and not any(token in {"<<", "<<-"} for token in plain_segment):
            paths = optionless(args, set())
        elif name in {"head", "tail"}:
            paths = optionless(args, {"-n", "--lines", "-c", "--bytes"})
        elif name == "sed":
            has_script_option = any(arg in {"-e", "--expression", "-f", "--file"} or arg.startswith(("-e", "-f", "--expression=", "--file=")) for arg in args)
            operands = optionless(args, {"-e", "--expression", "-f", "--file", "-i", "--in-place"})
            paths = operands if has_script_option else operands[1:]
        elif name in {"grep", "egrep", "fgrep", "rg"}:
            paths = grep_paths(args, name)
        elif re.fullmatch(r"python(?:[23](?:\.\d+)?)?", name):
            if "-c" not in args and "-m" not in args:
                operands = optionless(args, {"-W", "-X"})
                if operands and operands[0] != "-":
                    paths = operands[:1]
        elif name in {"pytest", "py.test"}:
            paths = pytest_paths(args)
        for path in paths:
            item = bashparse.record(path, ts, active_cwd, name)
            if item:
                reads.append(item)
    return reads


def reads_from_command(command: str, ts: str | None, cwd: str | None = None, *, exclusions: Counter | None = None) -> list[dict]:
    """Return heuristic reads made by the supported file-consuming commands."""
    return reads_walk(command, ts, cwd, exclusions)
