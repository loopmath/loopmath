"""Which git worktree a folder belongs to, from the file system, to share git's answer.

Onboarding asks git about a few thousand session folders that sit in about a hundred
worktrees. Git finds a worktree by walking up from a folder to the first `.git`, so
two folders whose walk ends at the same `.git` get the same answer from git. `where`
names that end without running git, and says when it cannot be sure (the caller then
asks git about the folder itself, as before).
"""

from __future__ import annotations

import os

# Settings that change where git's own walk stops.
_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM")


def _looks_bare(folder: str) -> bool:
    return (os.path.isfile(os.path.join(folder, "HEAD")) and os.path.isdir(os.path.join(folder, "objects"))
            and os.path.isdir(os.path.join(folder, "refs")))


def where(path: str) -> tuple[str, str | None]:
    """`("repo", folder)`: git's walk from `path` ends at `folder`, the nearest one
    holding `.git`. `("none", None)`: there is no `.git` above `path`, so git finds no
    worktree. `("unsure", None)`: ask git."""
    if any(os.environ.get(name) for name in _GIT_ENV):
        return "unsure", None
    try:
        here = os.path.realpath(path)  # git walks the physical path
        dev = os.stat(here).st_dev
    except OSError:
        return "unsure", None
    if ".git" in here.split(os.sep):
        return "unsure", None
    while True:
        try:
            if os.path.lexists(os.path.join(here, ".git")):
                return "repo", here
            if _looks_bare(here):
                return "unsure", None
            parent = os.path.dirname(here)
            if parent == here:
                return "none", None
            if os.stat(parent).st_dev != dev:
                return "unsure", None
        except OSError:
            return "unsure", None
        here = parent
