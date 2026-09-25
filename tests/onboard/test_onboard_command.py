"""`loopmath onboard` end to end on the synthetic history, with fakes for lanes 4, 5 and 7."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys

import pytest

from loopmath import cli
from loopmath.onboard import commands as C
from onboard_fixture import FakeStore, fake_infer

MARKERS = ("SECRET-PROMPT-MARKER", "DOCS-PROMPT-MARKER")
WHICH = {"claude": "/opt/bin/claude", "codex": "/opt/bin/codex"}.get


class Env:
    def __init__(self, fx, tmp_path, monkeypatch, **overrides):
        self.fx = fx
        self.home = tmp_path / "home"
        self.stores: dict = {}
        self.calls: list = []
        self.fits: list = []
        self.answer = self._claude
        self.store_config = overrides.pop("config", None)
        self.reject = overrides.pop("reject", None)

        def store(home):
            if home not in self.stores:
                self.stores[home] = FakeStore(home, self.store_config, self.reject)
            return self.stores[home]

        def fit(home):
            self.fits.append(home)
            return home / "fits" / "fit_20260923T153000.json"

        def runner(argv, *, input, capture_output, text, timeout, cwd):
            self.calls.append((argv, input))
            return self.answer(argv, input)

        deps = dict(logs=fx.logs, store=store, infer=fake_infer, fit=fit, runner=runner, which=WHICH,
                    isatty=lambda: False)
        deps.update(overrides)
        monkeypatch.setattr(C, "_deps", lambda: C.Deps(**deps))

    @property
    def store(self) -> FakeStore:
        return self.stores[self.home]

    def _claude(self, argv, stdin):
        labels = []
        for line in stdin.splitlines()[1:]:
            item = json.loads(line)
            kind = "docs" if "README" in (item.get("prompt") or "") else "bug_fix"
            labels.append({"id": item["id"], "type": kind, "subtype": None, "features": {"size": "s"},
                           "confidence": 0.85, "title": f"{kind} task"})
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "type": "result", "is_error": False, "structured_output": {"labels": labels},
            "total_cost_usd": 0.004, "usage": {"input_tokens": 2500, "output_tokens": 150}}), "")

    def run(self, *argv, capsys):
        code = cli.main(["onboard", "--since", "7d", "--home", str(self.home), *argv])
        out, err = capsys.readouterr()
        return code, out, err


@pytest.fixture
def env(history_dir, tmp_path, monkeypatch):
    return lambda **kw: Env(history_dir, tmp_path, monkeypatch, **kw)


def _clean(*texts):
    for text in texts:
        for marker in MARKERS:
            assert marker not in text


UNPRICED = "codex:loopmath-test-unpriced"  # a model no price table has, so the estimate has no dollars


def test_dry_run_writes_nothing_and_shows_the_estimate(env, capsys):
    e = env()
    code, out, err = e.run("--dry-run", "--labeler", UNPRICED, "--json", capsys=capsys)
    assert code == 0
    obj = json.loads(out)
    assert next(iter(obj)) == "schema" and obj["schema"] == "loopmath.onboard/1"
    assert obj["dry_run"] is True and obj["groups"] == {"total": 3, "to_label": 2, "before_window": 0, "skipped": {}}
    assert obj["sessions"] == {"claude-code": 3, "codex": 2} and obj["files"] == {"claude-code": 3, "codex": 2}
    lab = obj["labeler"]
    assert lab["chosen"] and lab["spec"] == UNPRICED and lab["from"] == "flag" and lab["saved"] is False
    assert lab["expected"]["calls"] == 1 and lab["expected"]["groups"] == 2
    assert lab["expected"]["usd"] is None
    assert lab["expected"]["usd_unknown"] == "no price row for loopmath-test-unpriced"
    assert obj["preview"]["by_type"] == {"bug_fix": 1, "docs": 1}
    assert obj["unclassified"]["by_reason"] == {"no prompt in the session": 1}
    assert obj["runs"] == {"written": 0, "ids": []} and obj["fit"] is None
    assert e.calls == [] and e.store.imports == 0 and e.store.cfg.saved == 0 and e.fits == []
    assert not e.home.exists()
    _clean(out, err)


def test_dry_run_terminal_summary(env, capsys):
    e = env()
    code, out, err = e.run("--dry-run", "--labeler", "claude:claude-haiku-4-5", capsys=capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("loopmath onboard (dry run): last 7d: sessions from ")  # the window, then the span
    assert any(line.startswith("  labeller: claude (claude-haiku-4-5), 2 groups in 1 call, expected about $") for line in lines)
    assert "  not classified: 1 (no prompt in the session 1)" in lines
    assert lines[-1].startswith("  dry run: no labelling call, nothing written")
    assert len(lines) <= 25
    _clean(out, err)


def test_no_labeller_does_everything_else_and_exits_2(env, capsys):
    e = env()
    code, out, err = e.run("--json", capsys=capsys)
    assert code == 2
    obj = json.loads(out)
    assert obj["needs"] == "labeler" and obj["labeler"]["chosen"] is False
    assert obj["labeler"]["forms"][0].startswith("claude:<model>") and obj["groups"]["to_label"] == 2
    assert "no labeller chosen. Choose the model that labels your history" in err and "never picks one" in err
    assert "command:<cmd>" in err and "onboard.labeler" in err
    assert e.calls == [] and e.store.imports == 0 and e.store.cfg.saved == 0
    code, out, _ = e.run("--dry-run", capsys=capsys)
    assert code == 0 and "labeller: not chosen" in out and "note: no labeller chosen" in out


def test_bad_labeller_values(env, capsys):
    e = env(which=lambda name: None)
    code, _, err = e.run("--labeler", UNPRICED, capsys=capsys)
    assert code == 1 and "codex is not on PATH" in err
    e = env(config={"onboard.labeler": "claude"})
    code, _, err = e.run("--dry-run", capsys=capsys)
    assert code == 1 and "claude:<model>" in err
    with pytest.raises(SystemExit):
        cli.main(["onboard", "--labeler", "claude"])


def test_no_yes_and_no_terminal_stops_before_any_call(env, capsys):
    e = env()
    code, out, err = e.run("--labeler", "claude:claude-haiku-4-5", capsys=capsys)
    assert code == 1 and "rerun with --yes to approve" in err and "2 session groups with claude (claude-haiku-4-5)" in err
    assert e.calls == [] and e.store.imports == 0 and e.store.cfg.saved == 0 and e.fits == []


def test_interactive_answer(env, capsys):
    asked = []
    e = env(isatty=lambda: True, ask=lambda text: asked.append(text) or "n\n")
    code, _, err = e.run("--labeler", "claude:claude-haiku-4-5", capsys=capsys)
    assert code == 1 and e.calls == [] and "nothing was written" in err
    assert asked and asked[0].startswith("Label 2 session groups with claude (claude-haiku-4-5), about $")
    e = env(isatty=lambda: True, ask=lambda text: "y\n")
    code, _, _ = e.run("--labeler", "claude:claude-haiku-4-5", capsys=capsys)
    assert code == 0 and len(e.calls) == 1 and len(e.store.runs) == 2


def test_full_run_writes_runs_usual_and_fit(env, capsys):
    e = env()
    code, out, err = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    assert code == 0, err
    obj = json.loads(out)
    assert len(e.calls) == 1 and e.calls[0][0][:4] == ["/opt/bin/claude", "-p", "--model", "claude-haiku-4-5"]
    assert "SECRET-PROMPT-MARKER" in e.calls[0][1]  # the prompt goes to the user's own CLI, and nowhere else
    assert obj["labeler"]["saved"] is True and e.store.cfg.values["onboard.labeler"] == "claude:claude-haiku-4-5"
    assert obj["labeler"]["actual"] == {"usd": 0.004, "tokens": 2650, "basis": "reported by the CLI"}
    assert obj["classified"] == {"total": 2, "by_type": {"bug_fix": 1, "docs": 1}}
    runs = e.store.runs
    assert sorted(runs) == sorted(obj["runs"]["ids"]) and len(runs) == 2
    by_type = {d["run"]["task"]["type"]: d for d in runs.values()}
    lead = by_type["bug_fix"]
    assert lead["ocp"] == "0.3" and lead["run"]["task"]["repo"] == "acme/app"
    assert lead["run"]["task"]["labeled_by"] == {"how": "labeler", "tier": "heuristic", "labeler": "claude",
                                                 "version": "label/1", "model": "claude-haiku-4-5", "confidence": 0.85}
    assert lead["run"]["configuration"]["source"] == "habit" and lead["run"]["provenance"]["chooser"] == "habit"
    assert by_type["docs"]["run"]["task"]["repo"] == "docs"
    values = e.store.cfg.values
    assert values['usual.bug_fix."acme/app"'] == lead["run"]["configuration"]["id"]
    assert values['usual.bug_fix."*"'] == lead["run"]["configuration"]["id"] and values['usual_meta.bug_fix."*"'] == 1
    assert values['usual.docs."docs"'] == by_type["docs"]["run"]["configuration"]["id"]
    assert [u["repo"] for u in obj["usual"]] == ["*", "acme/app", "*", "docs"]
    assert e.fits == [e.home] and obj["fit"] == {"id": "fit_20260923T153000.json", "error": None}
    _clean(out, err, json.dumps(runs))

    # Re-running replaces the same runs and does not save the labeller again.
    first_ids = sorted(runs)
    code, out, _ = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    assert code == 0 and sorted(e.store.runs) == first_ids and e.store.imports == 4
    assert json.loads(out)["labeler"]["saved"] is False


def test_terminal_summary_after_a_run(env, capsys):
    e = env()
    code, out, err = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", capsys=capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("loopmath onboard: last 7d: sessions from ")
    assert "  classified: 2 runs written: bug_fix 1, docs 1" in lines
    assert any(line.startswith("  labeller: claude (claude-haiku-4-5), 2 groups in 1 call,") and "actual $0.00" in line
               for line in lines)
    assert "  first fit: fit_20260923T153000.json" in lines and len(lines) <= 25
    _clean(out, err)


def test_configured_labeller_is_used_without_the_flag(env, capsys, tmp_path):
    script = tmp_path / "local_model.py"
    script.write_text(
        "import json, sys\n"
        "req = json.load(sys.stdin)\n"
        "print(json.dumps({'labels': [{'id': it['id'], 'type': 'refactor', 'subtype': None, 'features': {},\n"
        "                              'confidence': 0.6, 'title': 't'} for it in req['items']]}))\n")
    spec = f"command:{sys.executable} {script}"
    e = env(config={"onboard.labeler": spec}, runner=subprocess.run)
    code, out, err = e.run("--yes", "--json", capsys=capsys)
    assert code == 0, err
    obj = json.loads(out)
    assert obj["labeler"]["from"] == "config" and obj["labeler"]["saved"] is False
    assert obj["labeler"]["expected"]["usd"] is None and obj["labeler"]["actual"]["usd"] is None
    assert obj["labeler"]["actual"]["basis"] == "reported by your command"
    assert obj["classified"]["by_type"] == {"refactor": 2}
    doc = next(iter(e.store.runs.values()))
    assert doc["run"]["task"]["labeled_by"] == {"how": "labeler", "tier": "heuristic", "labeler": "command",
                                                "version": "label/1", "confidence": 0.6}
    assert spec not in json.dumps(e.store.runs)

    # The terminal says the command reported no spend, not `$?`.
    code, out, err = e.run("--yes", capsys=capsys)
    assert code == 0, err
    (line,) = [x for x in out.splitlines() if x.startswith("  labeller: ")]
    assert line.endswith("; actual: not reported by your command") and "$?" not in line


def test_the_header_names_the_window_then_the_sessions_found(env, capsys):
    e = env()
    start = (e.fx.t0 - dt.timedelta(days=2)).date().isoformat()
    code, out, _ = e.run("--dry-run", "--labeler", "none", "--since", start, capsys=capsys)
    assert code == 0 and out.splitlines()[0].startswith(f"loopmath onboard (dry run): since {start}: sessions from ")
    assert [C._window_words(s) for s in ("90d", "12w", "36H", "2026-07-01")] == [
        "last 90d", "last 12w", "last 36H", "since 2026-07-01"]


def test_since_goes_through_the_one_store_reader(env, capsys):
    """`3m` (months or minutes?) exits 2 before anything is read; other bad text exits 1."""
    def never(*a, **k):
        raise AssertionError("history must not be read")

    e = env(load_history=never)
    code, out, err = e.run("--dry-run", "--labeler", "none", "--since", "3m", capsys=capsys)
    assert code == 2 and out == "" and err == "error: --since 3m: ambiguous: use 90d, 12w or a date\n"
    for bad in ("soon", "30", "2026-13-01"):
        code, out, err = e.run("--dry-run", "--labeler", "none", "--since", bad, capsys=capsys)
        assert code == 1 and out == "" and err.startswith(f"error: --since {bad}")
    code, out, _ = env().run("--dry-run", "--labeler", "none", "--since", "12w", capsys=capsys)
    assert code == 0 and out.startswith("loopmath onboard (dry run): last 12w: sessions from ")
    code, out, _ = env().run("--dry-run", "--labeler", "none", "--since", " ", capsys=capsys)
    assert code == 0 and out.startswith("loopmath onboard (dry run): last 90d: sessions from ")  # blank is the default


def test_the_opening_line_warns_of_the_wait_even_when_captured(env, capsys):
    e = env()
    code, _, err = e.run("--dry-run", "--labeler", "none", "--json", capsys=capsys)
    assert code == 0 and err.splitlines()[0] == (
        "onboard: reading Claude Code and Codex history, last 7d; a large history can take several minutes")


def test_counts_of_one_are_singular():
    assert [C._n(n, "call") for n in (0, 1, 2, 5974)] == ["0 calls", "1 call", "2 calls", "5,974 calls"]
    claude = C.L.parse_labeler("claude:claude-haiku-4-5", which=WHICH)
    assert C._actual_line({"usd": None, "tokens": 0}, claude) == "actual: not reported by the CLI"
    assert C._actual_line({"usd": None, "tokens": 2650}, claude) == "actual: 2.6k tokens reported by the CLI, no price"
    assert C._actual_line({"usd": 0.004, "tokens": 2650}, claude) == "actual $0.00 (2.6k tokens reported)"


def test_keyword_guess_with_none(env, capsys):
    e = env()
    code, out, _ = e.run("--labeler", "none", "--json", capsys=capsys)
    assert code == 0 and e.calls == []
    obj = json.loads(out)
    assert obj["classified"]["by_type"] == {"bug_fix": 1, "docs": 1} and obj["labeler"]["saved"] is True
    doc = next(iter(e.store.runs.values()))
    assert doc["run"]["task"]["labeled_by"] == {"how": "inferred", "tier": "heuristic", "rule": "keywords",
                                                "version": "label/1", "confidence": 0.4}


def test_none_labeller_says_how_many_got_no_type_and_what_types_them(env, capsys, monkeypatch):
    real = C.L.taskmodel.guess_type
    monkeypatch.setattr(C.L.taskmodel, "guess_type", lambda text: (None, 0.0) if "README" in text else real(text))
    hint = ("  1 group got no type from keywords; a model labeller types them "
            "(--labeler claude:MODEL, codex:MODEL or command:CMD)")
    e = env()
    code, out, _ = e.run("--dry-run", "--labeler", "none", capsys=capsys)
    lines = out.splitlines()
    assert code == 0 and any(x.endswith("; 1 matched no keyword") for x in lines) and hint in lines
    code, out, _ = e.run("--labeler", "none", "--yes", capsys=capsys)
    lines = out.splitlines()
    assert code == 0 and "  not classified: 1 (no prompt in the session 1)" in lines  # G1: stored as unknown
    assert hint in lines and "  classified: 2 runs written: bug_fix 1, unknown 1" in lines
    assert "  usual workflow per task type and repo (runs with it, of all runs):" in lines
    assert any(x.startswith("    bug_fix, all repos: ") and x.endswith(" (1 of 1)") for x in lines)
    code, out, _ = e.run("--dry-run", "--labeler", "claude:claude-haiku-4-5", capsys=capsys)
    assert code == 0 and "got no type from keywords" not in out  # a model labeller decides for real


def test_a_failed_fit_keeps_the_runs_and_says_how_to_retry(env, capsys):
    def failing_fit(home):
        raise RuntimeError("the fit lock is held")

    e = env(fit=failing_fit)
    code, out, _ = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    obj = json.loads(out)
    assert code == 0 and obj["runs"]["written"] == 2 and len(e.store.runs) == 2
    assert obj["fit"] == {"id": None, "error": "the fit lock is held; run `loopmath fit` to retry"}


def test_store_rejection_is_reported_per_group(env, capsys):
    e = env(reject={"docs"})
    code, out, _ = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    obj = json.loads(out)
    assert code == 0 and obj["classified"]["total"] == 1
    assert obj["unclassified"]["by_reason"] == {"no prompt in the session": 1, "run not stored": 1}
    rejected = [g for g in obj["unclassified"]["groups"] if g["reason"].startswith("run not stored")]
    assert rejected[0]["reason"] == "run not stored (E120 run.task.type rejected by the fake store)"


def test_real_store_infer_emit_and_fit_on_this_build(history_dir, tmp_path, capsys, monkeypatch):
    """Unpatched except the logs folder: lane 7's store, lane 4's infer, lane 1's emit and lane 5's
    fit. Runs are stored and strictly valid, config gets the labeller and the usual rows, and the
    first fit is made."""
    from loopmath.ocp.emit import validate_strict
    from loopmath.store.home import Store

    monkeypatch.setattr(C, "_deps", lambda: C.Deps(logs=history_dir.logs))
    home = tmp_path / "home"
    code = cli.main(["onboard", "--since", "7d", "--home", str(home), "--labeler", "none", "--yes", "--json"])
    out, err = capsys.readouterr()
    assert code == 0, err
    obj = json.loads(out)
    assert obj["runs"]["written"] == 2 and obj["notes"] == []
    store = Store(home)
    docs = [json.loads(p.read_text()) for p in sorted(store.runs_dir.glob("*.ocp.json"))]
    assert sorted(d["run"]["id"] for d in docs) == sorted(obj["runs"]["ids"])
    assert all([f for f in validate_strict(d) if f["level"] == "error"] == [] for d in docs)
    config = store.config()
    assert config.get("onboard.labeler") == "none"
    assert config.usual("bug_fix", "acme/app") == config.usual("bug_fix") and config.usual("docs", "docs")
    assert obj["fit"]["error"] is None and (store.fits_dir / obj["fit"]["id"]).exists()
    _clean(out, err, json.dumps(docs))


def test_real_infer_and_emit_give_strictly_valid_runs(env, capsys):
    """Lane 4's `infer` and lane 1's emit helpers, fake store: every run passes strict OCP 0.3."""
    from loopmath.ocp.emit import validate_strict

    e = env(infer=C._infer)
    code, out, err = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    assert code == 0, err
    assert json.loads(out)["runs"]["written"] == 2 and len(e.store.runs) == 2
    for doc in e.store.runs.values():
        assert [f for f in validate_strict(doc) if f["level"] == "error"] == []
        vertices = {p["id"] for p in doc["run"]["configuration"]["workflow"]["pieces"]}
        assert doc["nodes"] and all(n["vertex"] in vertices for n in doc["nodes"])


def test_a_bad_run_document_does_not_lose_the_other_labels(env, capsys, monkeypatch):
    from loopmath.onboard import history as H

    real = H.run_doc

    def flaky(group, **kw):
        if group.repo == "docs":
            raise KeyError("attempts")
        return real(group, **kw)

    monkeypatch.setattr(H, "run_doc", flaky)
    e = env()
    code, out, _ = e.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    obj = json.loads(out)
    assert code == 0 and obj["classified"]["total"] == 1
    assert obj["unclassified"]["by_reason"]["run document failed"] == 1


def test_groups_still_running_are_left_out(env, capsys):
    """The docs session ran from t0 + 120 to t0 + 126 minutes; at t0 + 140 it counts as still running."""

    e = env()
    e2 = env(now=lambda: e.fx.t0.astimezone(dt.timezone.utc) + dt.timedelta(minutes=140))
    code, out, _ = e2.run("--labeler", "claude:claude-haiku-4-5", "--yes", "--json", capsys=capsys)
    obj = json.loads(out)
    assert code == 0 and obj["classified"]["by_type"] == {"bug_fix": 1}
    assert obj["unclassified"]["by_reason"] == {"still running": 1, "no prompt in the session": 1}


def test_dry_run_preview_is_not_unclassified_with_a_model_labeller(env, capsys, monkeypatch):
    from loopmath.onboard import label as L

    monkeypatch.setattr(L.taskmodel, "guess_type", lambda text: (None, 0.0))
    e = env()
    code, out, _ = e.run("--dry-run", "--labeler", UNPRICED, "--json", capsys=capsys)
    obj = json.loads(out)
    assert code == 0 and obj["preview"]["unmatched"] == 2 and obj["preview"]["by_type"] == {}
    assert obj["unclassified"]["by_reason"] == {"no prompt in the session": 1}
    code, out, _ = e.run("--dry-run", "--labeler", "none", "--json", capsys=capsys)
    obj = json.loads(out)
    assert obj["preview"]["how"] == "keyword guess"
    assert obj["unclassified"]["by_reason"] == {"no prompt in the session": 1}  # G1: the 2 would be unknown
    code, out, _ = e.run("--dry-run", "--labeler", UNPRICED, capsys=capsys)
    assert "  keyword preview: nothing matched; 2 matched no keyword" in out.splitlines()
