"""Usual workflow per (type, repo) and its config tables (D5)."""

from __future__ import annotations

from loopmath.onboard.usual import TYPE_LEVEL, UsualPick, usual_picks, usual_tables, write_usual


def _row(kind, repo, cfg, when):
    return {"type": kind, "repo": repo, "config": cfg, "label": f"wf: {cfg}", "started_at": when}


def test_most_common_wins_and_type_level_counts_every_repo():
    rows = [
        _row("bug_fix", "acme/app", "cfg_a", "2026-09-01T10:00:00-07:00"),
        _row("bug_fix", "acme/app", "cfg_a", "2026-09-02T10:00:00-07:00"),
        _row("bug_fix", "acme/app", "cfg_b", "2026-09-03T10:00:00-07:00"),
        _row("bug_fix", "docs", "cfg_b", "2026-09-04T10:00:00-07:00"),
        _row("bug_fix", "docs", "cfg_b", "2026-09-05T10:00:00-07:00"),
        _row("feature", None, "cfg_c", "2026-09-05T10:00:00-07:00"),
        {"type": None, "repo": "x", "config": "cfg_z"},
    ]
    picks = {(p.type, p.repo): p for p in usual_picks(rows)}
    assert picks[("bug_fix", "acme/app")] == UsualPick("bug_fix", "acme/app", "cfg_a", "wf: cfg_a", 2, 3)
    assert picks[("bug_fix", "docs")].config == "cfg_b"
    # type level: cfg_a 2, cfg_b 3
    assert picks[("bug_fix", TYPE_LEVEL)].config == "cfg_b" and picks[("bug_fix", TYPE_LEVEL)].total == 5
    assert picks[("feature", "unknown")].config == "cfg_c"
    assert len(picks) == 5
    order = [(p.type, p.repo) for p in usual_picks(rows)]
    assert order[0] == ("bug_fix", TYPE_LEVEL)


def test_tie_goes_to_most_recent():
    rows = [
        _row("docs", "r", "cfg_old", "2026-09-01T10:00:00-07:00"),
        _row("docs", "r", "cfg_new", "2026-09-09T10:00:00-07:00"),
    ]
    assert {p.repo: p.config for p in usual_picks(rows)} == {"r": "cfg_new", TYPE_LEVEL: "cfg_new"}


class FakeConfig:
    def __init__(self):
        self.values = {}
        self.saved = 0

    def set(self, key, value):
        self.values[key] = value

    def save(self):
        self.saved += 1


def test_tables_and_write():
    picks = usual_picks([_row("docs", "acme/app", "cfg_a", "2026-09-01T10:00:00-07:00")])
    assert usual_tables(picks) == {"usual": {"docs": {"*": "cfg_a", "acme/app": "cfg_a"}},
                                   "usual_meta": {"docs": {"*": 1, "acme/app": 1}}}
    config = FakeConfig()
    assert write_usual(config, picks) == 2
    assert config.values == {'usual.docs."*"': "cfg_a", 'usual_meta.docs."*"': 1,
                             'usual.docs."acme/app"': "cfg_a", 'usual_meta.docs."acme/app"': 1}
    assert config.saved == 1
    empty = FakeConfig()
    assert write_usual(empty, []) == 0 and empty.saved == 0
