"""Usual workflow per (type, repo).

Owner: lane 03. Spec: design/0.1/08-lanes.md section 3, 05-recommender.md section 1;
decision D5 in the lane questions log.

The usual workflow for a (type, repo) is the configuration its runs used most often;
a tie goes to the one used most recently. The type level (`"*"`) counts every repo.
Config layout (D5):

    [usual.bug_fix]
    "*" = "cfg_3f2a9c1b0d4e"
    "acme/app" = "cfg_8e1d22aa9f07"

    [usual_meta.bug_fix]
    "*" = 41
    "acme/app" = 12

`usual_meta` holds the number of runs behind each pick.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

TYPE_LEVEL = "*"


@dataclass(frozen=True)
class UsualPick:
    type: str
    repo: str          # a repo, or "*" for the type level
    config: str        # cfg_ id
    label: str         # Configuration.label(), for people
    runs: int          # runs that used this configuration
    total: int         # runs of this (type, repo)

    def to_dict(self) -> dict:
        return {"type": self.type, "repo": self.repo, "config": self.config, "label": self.label,
                "runs": self.runs, "total": self.total}


def usual_picks(rows: Iterable[dict]) -> list[UsualPick]:
    """Rows are {type, repo, config, label, started_at}; returns one pick per (type, repo)
    and per (type, "*"), sorted by type then by run count (type level first)."""
    counts: dict[tuple[str, str], Counter] = {}
    last: dict[tuple[str, str, str], str] = {}
    labels: dict[str, str] = {}
    for row in rows:
        kind, repo, config = row.get("type"), row.get("repo"), row.get("config")
        if not kind or not config:
            continue
        labels.setdefault(config, str(row.get("label") or config))
        when = str(row.get("started_at") or "")
        for key in ((kind, str(repo or "unknown")), (kind, TYPE_LEVEL)):
            counts.setdefault(key, Counter())[config] += 1
            if when > last.get((*key, config), ""):
                last[(*key, config)] = when
    picks = []
    for (kind, repo), counter in counts.items():
        config = max(counter, key=lambda c: (counter[c], last.get((kind, repo, c), ""), c))
        picks.append(UsualPick(kind, repo, config, labels[config], counter[config], sum(counter.values())))
    picks.sort(key=lambda p: (p.type, p.repo != TYPE_LEVEL, -p.total, p.repo))
    return picks


def usual_tables(picks: Iterable[UsualPick]) -> dict[str, dict[str, dict[str, Any]]]:
    """The config tables D5 describes: {"usual": {type: {repo: cfg}}, "usual_meta": {type: {repo: runs}}}."""
    usual: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for p in picks:
        usual.setdefault(p.type, {})[p.repo] = p.config
        meta.setdefault(p.type, {})[p.repo] = p.total
    return {"usual": usual, "usual_meta": meta}


def write_usual(config: Any, picks: Iterable[UsualPick]) -> int:
    """Write the picks into a store `Config` (lane 7: dotted `set`, then `save`).
    Returns the number of keys written. Other keys under `usual` are left alone."""
    n = 0
    for p in picks:
        config.set(f'usual.{p.type}."{p.repo}"', p.config)
        config.set(f'usual_meta.{p.type}."{p.repo}"', p.total)
        n += 1
    if n:
        config.save()
    return n
