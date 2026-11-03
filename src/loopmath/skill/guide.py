"""`loopmath guide [TOPIC]`: the instructions for each job, for the agent to read and follow.

The one installed skill (`skills/loopmath/SKILL.md`) sends the agent here, so a job's
instructions always match the installed loopmath. Each topic is a packaged file in `guides/`:
the five jobs, then the command reference they share. Read only: it needs no store and writes
nothing, so this module imports nothing from the rest of loopmath.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources

# Each topic, with one line on when to use it, in the order `loopmath guide` lists them.
TOPICS = {
    "onboard": "set up, start or try loopmath from the user's Claude Code and Codex history, or loopmath has no runs yet",
    "import": "OCP files or a folder of finished runs to bring into loopmath",
    "plan": "a task to do: pick its workflow, models and effort, then start the run",
    "record": "a task is done: record the run and show its receipt, or judge a pair blind",
    "fit": "update the fit, say what loopmath learned, or the fit is not usable",
    "reference": "every command, flag, JSON field and exit code the guides use",
}
FOLLOW = "Run `loopmath guide <topic>` and follow it."
EXIT_UNKNOWN = 2  # the CLI's "not found"


def text(topic: str) -> str:
    """One topic's text, as packaged. KeyError on another name."""
    if topic not in TOPICS:
        raise KeyError(topic)
    return resources.files("loopmath.skill").joinpath("guides", f"{topic}.md").read_text(encoding="utf-8")


def listing() -> str:
    """What `loopmath guide` alone prints: each topic with when to use it, then how to read one."""
    width = max(map(len, TOPICS))
    rows = [f"  {name:<{width}}  {when}" for name, when in TOPICS.items()]
    return "loopmath guides, one per job, and the command reference:\n\n" + "\n".join(rows) + f"\n\n{FOLLOW}\n"


def command(args: argparse.Namespace) -> int:
    topic = getattr(args, "topic", None)
    if topic is None:
        sys.stdout.write(listing())
        return 0
    try:
        sys.stdout.write(text(topic))
    except KeyError:
        print(f"error: no guide {topic!r}; the topics are {', '.join(TOPICS)}. {FOLLOW}", file=sys.stderr)
        return EXIT_UNKNOWN
    return 0
