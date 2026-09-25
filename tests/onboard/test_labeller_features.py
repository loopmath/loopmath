"""The labeller asks for the declared keys it fills (lane 2C, I14; spec 04 section 1)."""

from __future__ import annotations

import json
import subprocess

from test_onboard_command import Env

from loopmath.onboard import label as L
from loopmath.taskmodel import FeatureSet

WHICH = {"claude": "/opt/bin/claude"}.get
DECLARED = FeatureSet.from_config({
    "framework": {"fill": "labeller", "types": ["feature", "bug_fix"], "description": "Main framework."},
    "grid": {"kind": "bool", "fill": "orchestrator"},
})


def test_run_batches_passes_the_feature_set():
    calls = []

    def runner(argv, *, input, capture_output, text, timeout, cwd):
        calls.append(argv)
        ids = [json.loads(line)["id"] for line in input.splitlines() if line.startswith('{"id"')]
        labels = [{"id": i, "type": "bug_fix", "subtype": None, "confidence": 0.9, "title": "A task",
                   "features": {"size": "s", "framework": "django", "grid": "yes"}} for i in ids]
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "type": "result", "is_error": False, "result": "", "structured_output": {"labels": labels},
            "total_cost_usd": 0.01, "usage": {"input_tokens": 10, "output_tokens": 5}}), "")

    items = [{"id": f"grp_{i:03d}", "prompt": f"task {i}"} for i in range(2)]
    run = L.run_batches([items], L.parse_labeler("claude:claude-haiku-4-5", which=WHICH), features=DECLARED,
                        runner=runner)
    argv = calls[0]
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    props = schema["properties"]["labels"]["items"]["properties"]["features"]["properties"]
    assert "framework" in props and "grid" not in props  # grid is filled by the orchestrator
    assert "- framework (a short lowercase word, or unknown)" in argv[argv.index("--system-prompt") + 1]
    assert {g: v["features"] for g, v in run.labels.items()} == {
        "grp_000": {"size": "s", "framework": "django"}, "grp_001": {"size": "s", "framework": "django"}}
    plain = L.parse_answer({"labels": [{"id": "grp_000", "type": "bug_fix", "subtype": None, "confidence": 0.9,
                                        "title": "t", "features": {"framework": "django"}}]}, ["grp_000"], None)
    assert plain[0]["grp_000"]["features"] == {}  # without the set, only the built-ins are kept


def test_the_offered_price_is_for_the_declared_keys(history_dir, tmp_path, monkeypatch, capsys):
    """Onboard's labeller options (lane 2D) price the prompt the chosen labeller is sent."""
    def expected(config, *argv):
        e = Env(history_dir, tmp_path, monkeypatch, config=config)
        code, out, _ = e.run("--dry-run", *argv, "--json", capsys=capsys)
        assert code == 0
        return json.loads(out)["labeler"]

    declared = {"features": {"framework": {"fill": "labeller", "description": "Main framework the change touches."}}}
    spec = "claude:claude-haiku-4-5"
    offered = next(o for o in expected(declared)["options"] if o["spec"] == spec)["expected"]
    assert offered == expected(declared, "--labeler", spec)["expected"]
    plain = next(o for o in expected(None)["options"] if o["spec"] == spec)["expected"]
    assert offered["tokens"]["total"] > plain["tokens"]["total"]
