"""`loopmath ocp validate` and `loopmath ocp migrate` (spec 02 section 7)."""

from __future__ import annotations

import json
import subprocess
import sys

from loopmath.cli import main

from ._common import EXAMPLES, MIGRATE_GOLDEN, ROOT, V03_EXAMPLES, V03_GOLDEN, load


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_validate_passes_the_examples(capsys):
    paths = [str(p) for p in sorted(V03_EXAMPLES.glob("*.ocp.json"))] + [str(EXAMPLES / "swarm-v02.ocp.json")]
    code, out, _ = run(capsys, "ocp", "validate", *paths)
    assert code == 0
    assert out.count("PASS") == len(paths) and "FAIL" not in out


def test_validate_fails_on_errors_and_lists_them(capsys):
    code, out, _ = run(capsys, "ocp", "validate", str(V03_GOLDEN / "E190-fail.ocp.json"))
    assert code == 1
    assert out.startswith("FAIL") and "error E190" in out


def test_warnings_alone_do_not_fail(capsys):
    code, out, _ = run(capsys, "ocp", "validate", str(V03_GOLDEN / "W182-fail.ocp.json"))
    assert code == 0 and "warning W182" in out


def test_validate_missing_file_exits_2(capsys, tmp_path):
    code, out, _ = run(capsys, "ocp", "validate", str(tmp_path / "nope.json"), str(V03_EXAMPLES / "solo.ocp.json"))
    assert code == 2 and "E001" in out


def test_validate_json(capsys):
    code, out, _ = run(capsys, "ocp", "validate", "--json", str(V03_EXAMPLES / "solo.ocp.json"),
                       str(V03_GOLDEN / "E195-fail.ocp.json"))
    payload = json.loads(out)
    assert code == 1
    assert payload["schema"] == "loopmath.ocp.validate/1" and payload["ok"] is False
    first, second = payload["files"]
    assert first == {"path": str(V03_EXAMPLES / "solo.ocp.json"), "ocp": "0.3", "ok": True, "errors": 0,
                     "warnings": 0, "findings": []}
    assert second["ok"] is False and second["findings"][0]["code"] == "E195"
    assert set(second["findings"][0]) == {"level", "code", "path", "message"}


def test_validate_terminal_output_is_short(capsys, tmp_path):
    doc = load(V03_EXAMPLES / "solo.ocp.json")
    doc["events"] += [{"at": "2026-09-23T11:00:00-07:00", "type": f"custom_{i}"} for i in range(40)]
    path = tmp_path / "noisy.ocp.json"
    path.write_text(json.dumps(doc))
    _, out, _ = run(capsys, "ocp", "validate", str(path))
    assert len(out.splitlines()) <= 25 and "more (use --json for all)" in out


def test_migrate_to_stdout(capsys):
    code, out, err = run(capsys, "ocp", "migrate", str(MIGRATE_GOLDEN / "labeled-v02.ocp.json"))
    assert code == 0 and err == ""
    migrated = json.loads(out)
    assert migrated["ocp"] == "0.3" and migrated["run"]["task"]["id"] == "tsk_widget_since"


def test_migrate_out_dir_writes_one_file_per_input(capsys, tmp_path):
    inputs = [MIGRATE_GOLDEN / "contract-v3.run.json", MIGRATE_GOLDEN / "minimal-v01.ocp.json"]
    code, out, _ = run(capsys, "ocp", "migrate", *map(str, inputs), "--out", str(tmp_path / "out"))
    assert code == 0 and out.count("PASS") == 2
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == ["contract-v3.run.ocp.json", "minimal-v01.ocp.json"]
    code, _, _ = run(capsys, "ocp", "validate", *[str(tmp_path / "out" / n) for n in written])
    assert code == 0


def test_migrate_never_overwrites_an_input(capsys, tmp_path):
    source = tmp_path / "run.ocp.json"
    source.write_text((MIGRATE_GOLDEN / "minimal-v01.ocp.json").read_text())
    before = source.read_text()
    code, _, err = run(capsys, "ocp", "migrate", str(source), "--out", str(tmp_path))
    assert code == 1 and "overwrite" in err
    assert source.read_text() == before


def test_migrate_several_files_need_out_or_json(capsys):
    inputs = [str(MIGRATE_GOLDEN / "minimal-v01.ocp.json"), str(MIGRATE_GOLDEN / "swarm-v02.ocp.json")]
    code, _, err = run(capsys, "ocp", "migrate", *inputs)
    assert code == 1 and "--out" in err
    code, out, _ = run(capsys, "ocp", "migrate", "--json", *inputs)
    payload = json.loads(out)
    assert code == 0 and payload["schema"] == "loopmath.ocp.migrate/1"
    assert [f["from"] for f in payload["files"]] == ["0.1", "0.2"]
    assert all(f["doc"]["ocp"] == "0.3" for f in payload["files"])


def test_migrate_reports_unreadable_and_missing_inputs(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    future = tmp_path / "future.json"
    future.write_text(json.dumps({"ocp": "0.9"}))
    code, out, _ = run(capsys, "ocp", "migrate", "--json", str(bad), str(future))
    payload = json.loads(out)
    assert code == 1 and [f["ok"] for f in payload["files"]] == [False, False]
    assert "cannot read JSON" in payload["files"][0]["error"] and "0.9" in payload["files"][1]["error"]
    code, _, _ = run(capsys, "ocp", "migrate", "--json", str(tmp_path / "missing.json"))
    assert code == 2


def test_validate_says_what_a_file_without_a_version_is(capsys, tmp_path):
    """Dogfood: a contract run file, a manifest, broken JSON and a folder never print 'ocp None'."""
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"runs": 3}))
    (tmp_path / "cut.ocp.json").write_text('{\n  "ocp": "0.2",\n')
    paths = [MIGRATE_GOLDEN / "contract-v3.run.json", tmp_path / "MANIFEST.json", tmp_path / "cut.ocp.json", tmp_path]
    code, out, _ = run(capsys, "ocp", "validate", *map(str, paths))
    assert code == 2 and "None" not in out
    status = [line.rsplit("  (", 1)[1] for line in out.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert [s.split(";")[0] for s in status] == ["no ocp version", "no ocp version", "not JSON", "not a file"]
    assert status[2] == "not JSON; 1 error, 0 warnings)"
    assert out.count("no 'ocp' version field") == 2
    assert out.count("this is a contract run file: 'loopmath ocp migrate' converts it") == 1


def test_migrate_says_when_a_file_is_not_ocp(capsys, tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"runs": 3}))
    code, _, err = run(capsys, "ocp", "migrate", str(path))
    assert code == 1 and "None" not in err
    assert "not an OCP document (no 'ocp' version field) or a contract run file" in err


def test_migrate_names_both_inputs_that_would_share_an_output(capsys, tmp_path):
    inputs = [tmp_path / name / "swarm-a.ocp.json" for name in ("night", "viz")]
    for path in inputs:
        path.parent.mkdir()
        path.write_text((EXAMPLES / "swarm-v02.ocp.json").read_text())
    code, _, err = run(capsys, "ocp", "migrate", "--out", str(tmp_path / "out"), *map(str, inputs))
    assert code == 1 and f"{inputs[0]} and {inputs[1]} would both be written to" in err
    assert not (tmp_path / "out").exists()


def test_console_entry_point_runs_validate():
    result = subprocess.run([sys.executable, "-c", "import sys; from loopmath.cli import main; "
                             "sys.exit(main(sys.argv[1:]))", "ocp", "validate",
                             str(V03_EXAMPLES / "swarm.ocp.json")],
                            capture_output=True, text=True, cwd=ROOT, env=_env())
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("PASS")


def _env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env
