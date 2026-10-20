"""With `--json`, an error is one JSON object on stdout too.

A command that fails under `--json` without printing anything on stdout gets one object there,
`{"schema": "loopmath.error/1", "ok": false, "command": ..., "error": ..., "exit": ...}`, with the same
exit code and the same message on stderr as before. That covers a usage error from the parser (exit 2),
a handler that reports its error on stderr only, and an uncaught exception (the traceback still prints).
A command that printed its own JSON error, as `run import`, `check-change` and `gate` do, is left alone.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

from .output import emit_json

SCHEMA = "loopmath.error/1"
_MAX_KEPT = 64_000  # characters of stderr kept for the message


def wants_json(argv: list[str]) -> bool:
    """`--json` among the options, before any `--`."""
    for token in argv:
        if token == "--":
            return False
        if token == "--json":
            return True
    return False


def emit_error(command: str, message: str, code: int, hint: str | None = None) -> None:
    emit_json(SCHEMA, {"ok": False, "command": command, "error": message, **({"hint": hint} if hint else {}),
                       "exit": code})


class JsonErrorParser(argparse.ArgumentParser):
    """An ArgumentParser whose usage errors, under --json, also print one JSON object on stdout."""

    json_mode = False  # set by cli.main for each command line
    argv: list[str] = []

    def error(self, message: str):  # type: ignore[override]
        if JsonErrorParser.json_mode:
            self.print_usage(sys.stderr)
            print(f"{self.prog}: error: {message}", file=sys.stderr)
            emit_error(_command(self.prog, JsonErrorParser.argv), message, 2)
            sys.exit(2)
        super().error(message)


def _command(prog: str, argv: list[str]) -> str:
    """The command a usage error belongs to: the parser's words after `loopmath`, else the first word given."""
    words = prog.split()[1:]
    if not words and argv and not argv[0].startswith("-"):
        words = [argv[0]]
    return " ".join(words) or "loopmath"


class _Watch:
    """A pass-through text stream that remembers whether anything was written, and keeps what was."""

    def __init__(self, inner, keep: bool) -> None:
        self._inner = inner
        self._keep = keep
        self.wrote = False
        self.kept: list[str] = []
        self._size = 0

    def write(self, text: str) -> int:
        if text:
            self.wrote = True
            if self._keep and self._size < _MAX_KEPT:
                self.kept.append(text)
                self._size += len(text)
        return self._inner.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _message(stderr_text: str, command: str, code: int) -> tuple[str, str | None]:
    """The error lines on stderr (else its last line), and its `hint:` lines."""
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    errors = [line.split("error: ", 1)[1] for line in lines if line.startswith("error: ") or ": error: " in line]
    hints = [line[len("hint: "):] for line in lines if line.startswith("hint: ")]
    hint = "; ".join(hints) or None
    if errors:
        return "; ".join(errors), hint
    rest = [line for line in lines if not line.startswith("hint: ")]
    if rest:
        return rest[-1], hint
    return f"loopmath {command} stopped with exit {code}", hint


def run_guarded(func: Callable[[argparse.Namespace], int], args: argparse.Namespace, command: str) -> int:
    """Run a handler under --json: when it fails and stdout stayed empty, print the error object there."""
    out_watch, err_watch = _Watch(sys.stdout, keep=False), _Watch(sys.stderr, keep=True)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_watch, err_watch
    try:
        code = func(args)
    except SystemExit as exc:
        sys.stdout, sys.stderr = old_out, old_err
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code and not out_watch.wrote:
            text = "".join(err_watch.kept) or (exc.code if isinstance(exc.code, str) else "")
            message, hint = _message(text, command, code)
            emit_error(command, message, code, hint)
        raise
    except Exception as exc:
        sys.stdout, sys.stderr = old_out, old_err
        if not out_watch.wrote:
            emit_error(command, f"{type(exc).__name__}: {exc}", 1)
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    if code and not out_watch.wrote:
        message, hint = _message("".join(err_watch.kept), command, code)
        emit_error(command, message, code, hint)
    return code
