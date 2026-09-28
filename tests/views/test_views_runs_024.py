"""0.2.4 runs output: times in local time with the zone (P2-6) and, under `runs --slate`, which run the referee
preferred (P3-24). The store is test_views_runs's."""

from __future__ import annotations

from loopmath.cli import main
from loopmath.views import common

from tests.views.test_views_runs import pacific, store  # noqa: F401 (fixtures)


def test_the_table_and_one_run_show_local_times_with_the_zone(tmp_path, v03, capsys, pacific):
    v03.write_store(tmp_path / "home", [v03.run_doc("run_utc", started="2026-09-27T22:42:14Z"),
                                        v03.run_doc("run_off", started="2026-09-28T01:00:00+02:00")])
    assert main(["runs", "--home", str(tmp_path / "home")]) == 0
    out = capsys.readouterr().out
    assert "2026-09-27 15:42 PDT" in out and "2026-09-27 16:00 PDT" in out
    assert "T22:42" not in out and "Z " not in out
    assert main(["runs", "--home", str(tmp_path / "home"), "--run", "run_utc"]) == 0
    assert capsys.readouterr().out.splitlines()[0].endswith("started 2026-09-27 15:42 PDT")


def test_fmt_time_rule(pacific):
    assert common.fmt_time("2026-09-27T22:42:14.000Z") == "2026-09-27 15:42 PDT"
    assert common.fmt_time("2026-12-27T10:00:00-07:00") == "2026-12-27 09:00 PST"
    assert common.fmt_time(None) == "n/a"


def test_runs_slate_says_which_run_the_referee_preferred(store, capsys):
    assert main(["runs", "--home", str(store), "--slate", "slt_1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "slate slt_1: the referee preferred run_b" in lines
    row_b = next(x for x in lines if x.startswith("run_b "))
    row_c = next(x for x in lines if x.startswith("run_c "))
    assert row_b.endswith("; preferred by the referee") and row_c.endswith("; the referee preferred run_b")
    assert main(["runs", "--home", str(store), "--run", "run_c"]) == 0
    assert "slate slt_1  the referee preferred run_b" in capsys.readouterr().out
