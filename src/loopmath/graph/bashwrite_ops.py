"""Write recognition built on :mod:`loopmath.graph.bashparse`."""

from __future__ import annotations

import os
from collections import Counter

from . import bashparse


def copy_move_targets(args: list[str]) -> tuple[list[str], str | None, bool]:
    operands: list[str] = []
    target_directory: str | None = None
    no_target_directory = False
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            operands.extend(args[index + 1 :])
            break
        if arg.startswith("--target-directory="):
            target_directory = arg.split("=", 1)[1]
        elif arg == "--target-directory":
            if index + 1 < len(args):
                target_directory = args[index + 1]
                index += 1
        elif arg == "--no-target-directory":
            no_target_directory = True
        elif arg == "--suffix":
            index += 1
        elif arg.startswith("--"):
            pass
        elif arg.startswith("-") and arg != "-":
            cluster = arg[1:]
            if "T" in cluster:
                no_target_directory = True
            target_option = cluster.find("t")
            if target_option >= 0:
                attached_target = cluster[target_option + 1 :]
                if attached_target:
                    target_directory = attached_target
                elif index + 1 < len(args):
                    target_directory = args[index + 1]
                    index += 1
            elif cluster.endswith("S"):
                index += 1
        else:
            operands.append(arg)
        index += 1
    if target_directory is not None:
        return operands, target_directory, True
    if len(operands) < 2:
        return operands, None, False
    target = operands[-1]
    sources = operands[:-1]
    return sources, target, not no_target_directory and (len(operands) > 2 or target.endswith("/"))


def destination(target: str, source: str, is_directory: bool) -> str:
    if not is_directory:
        return target
    return os.path.join(target, source.rstrip("/").rsplit("/", 1)[-1])


def writes_walk(command: str, ts: str | None, cwd: str | None, exclusions: Counter | None) -> list[dict]:
    writes: list[dict] = []
    prepared = bashparse.without_heredoc_bodies(command, exclusions)
    outer, substitutions = bashparse.pull_substitutions(prepared, exclusions)
    active_cwd = cwd
    for segment in bashparse.segments(bashparse.tokens(outer, exclusions, prepared=True)):
        plain_segment: list[str] = []
        for token in segment:
            for match in bashparse.SUBSTITUTION_RE.finditer(token):
                writes.extend(writes_walk(substitutions[int(match.group(1))], ts, active_cwd, exclusions))
            if not bashparse.SUBSTITUTION_RE.fullmatch(token):
                plain_segment.append(bashparse.restore_substitutions(token, substitutions))
        command_at = bashparse.command_index(plain_segment)
        if command_at is None:
            continue
        name = os.path.basename(plain_segment[command_at])
        has_heredoc = any(token in {"<<", "<<-"} for token in plain_segment)
        args = bashparse.plain_args(plain_segment, command_at)
        if name == "cd":
            active_cwd = bashparse.update_cwd(args, active_cwd)
            continue
        for index, token in enumerate(plain_segment):
            if token not in bashparse.REDIRECTS:
                continue
            if index + 1 >= len(plain_segment) or plain_segment[index + 1] in bashparse.REDIRECTS | {"<", "<<", "<<-"}:
                bashparse.exclude(exclusions, "invalid_redirect")
                continue
            how = "heredoc" if has_heredoc else ("append" if token == ">>" else "redirect")
            item = bashparse.record(plain_segment[index + 1], ts, active_cwd, how, exclusions=exclusions)
            if item:
                writes.append(item)
        if name == "tee":
            append = "-a" in args or "--append" in args
            for path in (arg for arg in args if not arg.startswith("-")):
                how = "tee heredoc" if has_heredoc else ("tee append" if append else "tee")
                item = bashparse.record(path, ts, active_cwd, how, exclusions=exclusions)
                if item:
                    writes.append(item)
        elif name in {"cp", "mv"}:
            sources, target, is_directory = copy_move_targets(args)
            if not sources or target is None:
                bashparse.exclude(exclusions, "invalid_cp_mv")
                continue
            for source in sources:
                item = bashparse.record(destination(target, source, is_directory), ts, active_cwd, name, exclusions=exclusions)
                if item:
                    writes.append(item)
        elif name == "codex" and "exec" in args:
            for index, arg in enumerate(args):
                path = args[index + 1] if arg == "-o" and index + 1 < len(args) else (arg[2:] if arg.startswith("-o") and len(arg) > 2 else None)
                if path:
                    item = bashparse.record(path, ts, active_cwd, "codex -o", exclusions=exclusions, pending_producer=True)
                    if item:
                        writes.append(item)
                    break
    return writes


def writes_from_command(command: str, ts: str | None, cwd: str | None = None, *, exclusions: Counter | None = None) -> list[dict]:
    """Return heuristic writes expressed directly by *command*."""
    return writes_walk(command, ts, cwd, exclusions)
