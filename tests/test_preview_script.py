"""scripts/preview.sh's recommend summary: a curve level that no candidate reaches
reads as recommend's own text says it, not as the `None: None` of a null config."""

import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "preview.sh"


def _digest_functions() -> dict:
    """The summary helpers from the Python program the script runs, without running it."""
    program = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    ns: dict = {}
    for node in ast.parse(program).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_iv", "_pred", "recommend_digest"):
            exec(compile(ast.Module([node], []), str(SCRIPT), "exec"), ns)
    return ns


def test_a_level_no_candidate_reaches_reads_as_recommend_says_it():
    rec = {
        "message": "m",
        "usual": {"label": "u", "from": "default", "prediction": None},
        "default_pick": {"label": "d"},
        "curve": [
            {"levels": [50], "config": "cfg_2e65bf8345ea", "reached": True,
             "prediction": {"p_success": {"mean": 0.54, "lo": 0.0, "hi": 1.0},
                            "cost": {"usd": {"mean": 4.71, "lo": 0.07, "hi": 11.3}}}},
            {"levels": [70, 80], "config": None, "reached": False, "prediction": None},
        ],
    }
    lines = _digest_functions()["recommend_digest"](rec)
    assert "  levels [50] cfg_2e65bf8345ea: p_success 0.54 [0.00, 1.00], usd 4.71 [0.07, 11.30]" in lines
    assert "  levels [70, 80]: not reached by any candidate" in lines
    assert not any("None" in line for line in lines if "levels" in line)
