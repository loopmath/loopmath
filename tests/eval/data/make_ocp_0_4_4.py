"""Writes `tests/eval/data/ocp-0-4-4/`: OCP run documents made by another tool, and the study that loopmath
0.4.4 built from them.

Run once by hand with loopmath 0.4.4 on the path, from the repository root:

    git archive v0.4.4 src | tar -x -C /some/empty/folder
    PYTHONPATH=/some/empty/folder/src python tests/eval/data/make_ocp_0_4_4.py

It refuses to run with any other version. The files it wrote are committed, so tests read them and never
run this. Everything in them is made up.

The documents: runs `a.b` and `r2` of input `case-01` on the slate `slate-1`, each with a score signal `s1`
and the preference `f1` between them; runs `r3` and `r4` of input `case-02` with the preference `g1` on
the default slate `eval.case-02`. No record has a `dev.loopmath.eval` member. 0.4.4 named the readings
`a.b.s1` and `r2.s1` and the pairs `f1` and `g1`.
"""

from __future__ import annotations

import json
from pathlib import Path

import loopmath
from loopmath import cli

HERE = Path(__file__).parent
OUT = HERE / "ocp-0-4-4"
RUBRIC = HERE / "study-small" / "rubric.json"
AT = "2026-02-03T10:{:02d}:00-08:00"


def document(run: str, input_id: str, arm: str, **more) -> dict:
    settings = {"final": {"harness": "cli", "effort": "low", "model": {"id": f"model-{arm}", "family": arm}}}
    body = {"id": run, "task": {"id": input_id}, "labels": {"arm": arm},
            "configuration": {"workflow": {"pieces": [{"id": "final", "role": "dev"}], "edges": []},
                              "settings": settings}, "signals": [], **more}
    return {"ocp": "0.4", "producer": {"name": "example", "version": "1"}, "privacy": {"profile": "metadata_only"},
            "run": body, "nodes": [], "edges": [], "attempts": [], "artifacts": []}


def score(ident: str, value: float, minute: int) -> dict:
    return {"id": ident, "kind": "score", "name": "depth", "value": value, "tier": "reported",
            "observed_at": AT.format(minute), "source": {"kind": "judge", "ref": "reviewer"}}


def main() -> None:
    if loopmath.__version__ != "0.4.4":
        raise SystemExit(f"run this with loopmath 0.4.4 on the path, not {loopmath.__version__}")
    slate = {"id": "slate-1", "members": ["a.b", "r2"]}
    f1 = {"id": "f1", "slate": "slate-1", "winner": "a.b", "members": ["a.b", "r2"], "observed_at": AT.format(5),
          "judge": {"kind": "model", "model": "judge-1"}}
    g1 = {"id": "g1", "slate": "eval.case-02", "winner": "r4", "members": ["r3", "r4"], "observed_at": AT.format(6),
          "judge": {"kind": "model", "model": "judge-1"}}
    docs = {"ab.json": document("a.b", "case-01", "baseline", slate=slate, signals=[score("s1", 4, 1)],
                                preferences=[f1]),
            "r2.json": document("r2", "case-01", "strong", slate=slate, signals=[score("s1", 7, 2)], preferences=[f1]),
            "r3.json": document("r3", "case-02", "baseline", preferences=[g1]),
            "r4.json": document("r4", "case-02", "strong", preferences=[g1])}
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; move it away first")
    (OUT / "in").mkdir(parents=True)
    paths = []
    for name, doc in docs.items():
        paths.append(OUT / "in" / name)
        paths[-1].write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    code = cli.main(["eval", "scores", "import", "--ocp", *map(str, paths), "--rubric", str(RUBRIC),
                     "--out", str(OUT / "study")])
    if code != 0:
        raise SystemExit(f"loopmath 0.4.4 refused the documents (exit {code})")


if __name__ == "__main__":
    main()
