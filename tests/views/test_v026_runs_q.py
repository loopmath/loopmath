"""The runs view reads `outcome.q.<tier>` from config (0.2.6, with fix (1)).

Before 0.2.6 the view showed the built-in q beside each run while `run finish` and, from 0.2.6, the fit
used the configured one. Unset, the view is 0.2.5's. Synthetic runs only.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from loopmath.belief.outcome import TIER_Q
from loopmath.cli import main
from loopmath.views import runs
from tests.share.test_v026_share_q import BROKEN, break_config

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")


def _store(tmp_path, v03):
    s = v03.signal
    docs = [v03.run_doc("run_v", signals=[s("sig_v", "verdict", "tests", "pass", "2026-09-20T10:59:00-07:00")]),
            v03.run_doc("run_r", signals=[s("sig_r", "verdict", "tests", "pass", "2026-09-20T10:59:00-07:00",
                                            "reported")])]
    return v03.write_store(tmp_path / "home", docs)


def _q(root):
    return {row["run"]: (row["tier"], row["q"]) for row in runs.build_view(root, now=NOW)["runs"]}


def test_the_runs_view_shows_the_configured_q(tmp_path, v03):
    root = _store(tmp_path, v03)
    assert _q(root) == {"run_v": ("verified", TIER_Q["verified"]), "run_r": ("reported", TIER_Q["reported"])}
    assert main(["config", "set", "outcome.q.verified", "0.9", "--home", str(root)]) == 0
    assert _q(root) == {"run_v": ("verified", 0.9), "run_r": ("reported", TIER_Q["reported"])}


@pytest.mark.parametrize("how", BROKEN)
def test_an_unreadable_config_keeps_the_built_in_q_and_says_so(tmp_path, v03, capsys, how):
    root = _store(tmp_path, v03)
    unset = runs.build_view(root, now=NOW)
    assert main(["runs", "--json", "--home", str(root)]) == 0
    unset_cli = json.loads(capsys.readouterr().out)
    path = break_config(root, how)
    try:
        view = runs.build_view(root, now=NOW)
        code = main(["runs", "--json", "--home", str(root)])
        out, err = capsys.readouterr()
    finally:
        if path.is_file():
            path.chmod(0o644)
    assert view["runs"] == unset["runs"]  # 0.2.5's rows, with the built-in q
    assert {row["run"]: row["q"] for row in view["runs"]} == {"run_v": TIER_Q["verified"], "run_r": TIER_Q["reported"]}
    (note,) = [n for n in view["notes"] if "outcomes use the built-in q" in n]
    assert note.startswith(f"{path}: ") and note.endswith("; outcomes use the built-in q")
    assert code == 0 and "Traceback" not in err  # the command, as well as the view
    assert json.loads(out)["runs"] == unset_cli["runs"]
