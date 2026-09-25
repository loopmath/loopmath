"""`ocp validate` ends with `N passed, M failed`."""

from __future__ import annotations

import json

from loopmath.cli import main

from ._common import V03_EXAMPLES, load


def test_validate_ends_with_a_summary_line(capsys, tmp_path):
    good = V03_EXAMPLES / "solo.ocp.json"
    bad = tmp_path / "bad.ocp.json"
    doc = load(good)
    doc.pop("run")
    bad.write_text(json.dumps(doc))
    assert main(["ocp", "validate", str(good), str(good), str(bad)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "2 passed, 1 failed"
    assert main(["ocp", "validate", str(good)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "1 passed, 0 failed"


def test_validate_json_counts(capsys, tmp_path):
    good = V03_EXAMPLES / "solo.ocp.json"
    code = main(["ocp", "validate", str(good), str(tmp_path / "missing.ocp.json"), "--json"])
    obj = json.loads(capsys.readouterr().out)
    assert code == 2 and obj["passed"] == 1 and obj["failed"] == 1 and len(obj["files"]) == 2
