"""The packaged skill texts, and the commands and JSON fields they show, for the skill tests.

A skill tells the agent which fields to read in sentences of one form: "Read `a`, `b.c`, `d[]`
(`e`, `f`)." A backticked name is a field path (`[]` marks a list), and backticked names in
parentheses right after a field are its fields (of each element, for a list). Each such sentence
belongs to the last `loopmath ...` command shown before it.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any

from loopmath import cli
from loopmath.skill import install as inst

ROOT = Path(__file__).resolve().parents[2]
SH_BLOCK = re.compile(r"```sh\n(.*?)```", re.S)
INLINE = re.compile(r"`(loopmath [^`]+)`")
READ = re.compile(r"\b[Rr]ead (?=`)")
PLACEHOLDER = re.compile(r"[A-Z][A-Z_]*(\.\.\.)?")


def texts() -> dict[str, str]:
    """Each packaged skill by name, then the shared reference as `reference`."""
    return {name: inst.skill_text(name) for name in (*inst.SKILLS, "reference")}


def frontmatter(text: str) -> dict[str, str]:
    head = text.split("---\n", 2)[1]
    out = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"')
    return out


class _Built(Exception):
    pass


def parser() -> argparse.ArgumentParser:
    """The real top-level parser, as `cli.main` builds it."""
    real = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Built(self)

    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except _Built as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise AssertionError("cli.main did not build a parser")


def subcommands(p: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in p._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def walk(top: argparse.ArgumentParser, words: list[str]) -> tuple[list[str], argparse.ArgumentParser, list[str]]:
    """(the subcommand chain, its parser, the words after it) for `loopmath WORDS`."""
    p, chain, words = top, [], list(words)
    while words and words[0] in subcommands(p):
        chain.append(words[0])
        p = subcommands(p)[words.pop(0)]
    return chain, p, words


def key(top: argparse.ArgumentParser, words: list[str]) -> str:
    """What a command's JSON is filed under: its subcommands, and `--dry-run` when given."""
    chain, _, rest = walk(top, words)
    return " ".join(chain) + (" --dry-run" if "--dry-run" in rest else "")


def commands(text: str) -> list[tuple[int, str]]:
    """(offset, `loopmath ...` command) for each command in an sh block (continuations joined) or
    an inline code span, in text order."""
    found = []
    for block in SH_BLOCK.finditer(text):
        for n, line in enumerate(block.group(1).replace("\\\n", " ").splitlines()):
            line = line.split("#", 1)[0].strip()
            if line.startswith("loopmath "):
                found.append((block.start() + n, line))
    outside = SH_BLOCK.sub(lambda m: " " * len(m.group(0)), text)
    found += [(m.start(), m.group(1)) for m in INLINE.finditer(outside)]
    return sorted(found)


def words(command: str) -> list[str]:
    """The words after `loopmath`, as a shell would split them."""
    return shlex.split(command)[1:]


def reads(text: str) -> list[tuple[int, str]]:
    """(offset, clause) for each "Read `...`" sentence: up to a period, colon or semicolon that ends
    a sentence outside code spans and parentheses."""
    out = []
    for m in READ.finditer(text):
        i, depth, tick = m.end(), 0, False
        while i < len(text):
            c = text[i]
            if c == "`":
                tick = not tick
            elif not tick:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth == 0 and c in ".:;" and (i + 1 == len(text) or text[i + 1].isspace()):
                    break
                elif c == "\n" and text[i + 1:i + 2] == "\n":
                    break
            i += 1
        out.append((m.start(), text[m.end():i]))
    return out


Field = tuple[str, list]


def fields(clause: str) -> list[Field]:
    """The field tree of one Read clause: [(path, [(path, [...]), ...]), ...]."""
    top: list[Field] = []
    stack, opened, last, i = [top], [], None, 0
    while i < len(clause):
        c = clause[i]
        if c == "`":
            j = clause.index("`", i + 1)
            last = (clause[i + 1:j], [])
            stack[-1].append(last)
            i = j + 1
            continue
        if c == "(":
            child = last is not None and clause[:i].rstrip().endswith("`")
            opened.append(child)
            if child:
                stack.append(last[1])
        elif c == ")" and opened.pop():
            stack.pop()
        i += 1
    return top


def names(tree: list[Field]) -> list[str]:
    """Every path in a field tree, parents first."""
    return [n for name, kids in tree for n in (name, *names(kids))]


def read_fields(text: str, top: argparse.ArgumentParser) -> list[tuple[str | None, list[Field]]]:
    """(command key, field tree) for each Read sentence; the key is None with no command before it."""
    shown = commands(text)
    out = []
    for at, clause in reads(text):
        before = [c for pos, c in shown if pos < at]
        out.append((key(top, words(before[-1])) if before else None, fields(clause)))
    return out


def resolve(outputs: list[Any], name: str, kids: list[Field]) -> str | None:
    """None when some output has the field NAME and each of its fields is in some element of
    NAME across all outputs; else what is missing. Null values and empty lists say nothing, so a
    field only null or empty everywhere passes."""
    cur = list(outputs)
    for part in name.split("."):
        many = part.endswith("[]")
        k = part[:-2] if many else part
        nxt, missing = [], None
        for o in cur:
            if o is None:
                continue
            if not isinstance(o, dict) or k not in o:
                missing = f"`{name}`: no `{k}`"
                continue
            v = o[k]
            if many and v is not None and not isinstance(v, list):
                return f"`{name}`: `{k}` is not a list"
            nxt += v if many and v is not None else [v]
        if missing and not nxt:
            return missing
        cur = nxt
    items = [o for o in cur if o is not None]
    for kid, grandkids in kids:
        why = resolve(items, kid, grandkids) if items else None
        if why:
            return f"`{name}` > {why}"
    return None
