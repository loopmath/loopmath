"""`docs/eval-recording.md` and the agent guide `eval-record`, held to the code.

The tutorial tells people how to record runs and judge outputs for `loopmath eval`. What it shows is checked
here against the real commands and the real validators:

- the example study under `docs/examples/eval-recording/` is what `record_study.py` writes, byte for byte, and
  every example passes `eval rubric check` and `eval scores check`, with the counts its design gives;
- a block marked `<!-- file: NAME -->` holds lines of that example file, word for word;
- a block marked `<!-- output: NAME -->` is the standard output of the command of that name, run here;
- a table marked `<!-- fields: SCHEMA -->` names members of that format: each one is known to the validator
  (a name it does not know is caught), and its "Required" cell is what the validator does when it is left out;
- a table marked `<!-- columns: TABLE -->` names the columns `eval scores import` reads from a CSV;
- a block marked `<!-- refusal: NAME -->` is the message `eval scores check` prints for that mistake, made here
  by breaking a copy of the example in one place;
- what the tutorial says the check lets through, it lets through;
- every `loopmath eval` command shown parses, every link to a heading lands, and the agent guide's JSON
  examples validate.

A block marked `<!-- recorded: NAME -->` is output that depends on the sampler. The test marked `slow` runs the
fit of the walk (about 3 minutes on 2 cores) and compares those blocks with the numbers masked. It runs only
with LOOPMATH_EVAL_SLOW=1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc as engine
from loopmath.eval import scores as S

REPO = Path(__file__).parents[2]
DOC = REPO / "docs" / "eval-recording.md"
GUIDE = REPO / "src" / "loopmath" / "skill" / "guides" / "eval-record.md"
EX = REPO / "docs" / "examples" / "eval-recording"
needs_engine = pytest.mark.skipif(not engine.available(), reason="needs the [bayes] extra")
EM_DASH = chr(0x2014)


def lm(*argv: object) -> tuple[int, str, str]:
    """`loopmath ARGV` in this process: the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main([str(word) for word in argv])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def check(study: object, rubric: object = None, judges: object = None, *more: str) -> tuple[int, str, str]:
    """`eval scores check` on a study folder, with the rubric and the judges file of that folder by default."""
    study = Path(str(study))
    return lm("eval", "scores", "check", study, "--rubric", rubric or study / "rubric.json",
              "--judges", judges or study / "judges.json", *more)


def report(study: object, rubric: object = None, judges: object = None) -> dict:
    """The JSON report of `eval scores check`, whatever the exit code."""
    return json.loads(check(study, rubric, judges, "--json")[1])


# ---------------------------------------------------------------------------------------------------
# Reading the tutorial

TAG = re.compile(r"<!-- (file|output|recorded|refusal|fields|columns): (.+?) -->")


def tagged(text: str, kind: str) -> list[tuple[str, list[str]]]:
    """`(argument, lines)` of each fenced block or table that follows a `<!-- kind: argument -->` line."""
    lines, found = text.splitlines(), []
    for n, line in enumerate(lines):
        match = TAG.fullmatch(line.strip())
        if match is None or match.group(1) != kind:
            continue
        rest, body = lines[n + 1:], []
        if rest and rest[0].startswith("```"):
            for row in rest[1:]:
                if row.startswith("```"):
                    break
                body.append(row)
        else:
            for row in rest:
                if not row.startswith("|"):
                    break
                body.append(row)
        assert body, f"nothing follows the mark on line {n + 1}"
        found.append((match.group(2), body))
    return found


def table(rows: list[str]) -> list[list[str]]:
    """The cells of a Markdown table, without its two heading rows."""
    return [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows[2:]]


def named(cell: str) -> str:
    """The name a table cell gives in backticks."""
    match = re.fullmatch(r"`([^`]+)`", cell)
    assert match, f"the cell {cell!r} is not one name in backticks"
    return match.group(1)


def shown_in(block: list[str], real: list[str]) -> bool:
    """Whether the lines of a block are lines of the real output, in order. A `...` line stands for lines left
    out; a block without one is the whole output. Blank lines do not count."""
    want = [line.rstrip() for line in block if line.strip()]
    have = [line.rstrip() for line in real if line.strip()]
    if "..." not in want:
        return want == have
    rest = iter(have)
    return all(any(line == other for other in rest) for line in want if line != "...")


def masked(lines: list[str]) -> list[str]:
    """Lines with every number replaced, for output that depends on the sampler."""
    return [re.sub(r"[-+]?\d+(?:\.\d+)?", "N", line) for line in lines]


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# The walk of the tutorial, up to the fit

STUDY_ARGS = ("--rubric", "my-study/rubric.json", "--judges", "my-study/judges.json")
REVIEWER_IMPORT = ("eval", "scores", "import", "--csv", "my-study-reviewer.csv", "--pairs-csv",
                   "my-study-reviewer-pairs.csv", "--rubric", "my-study/rubric.json", "--out", "my-study")


@pytest.fixture(scope="module")
def walk(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    """The commands of the tutorial, run as written in a new folder: the folder, and each standard output under
    the name the tutorial marks it with. `generated` is a copy of the study before the reviewer's sheets."""
    root, before, out = tmp_path_factory.mktemp("recording"), Path.cwd(), {}
    os.chdir(root)
    try:
        done = subprocess.run([sys.executable, str(EX / "record_study.py"), "--out", "my-study"],
                              capture_output=True, text=True, check=True)
        out["record"] = done.stdout
        shutil.copytree("my-study", "generated")

        def step(name: str, *argv: object) -> None:
            code, text, err = lm(*argv)
            assert code == 0, (name, err.strip()[-600:])
            out[name] = text

        step("rubric-check", "eval", "rubric", "check", "my-study/rubric.json", "--judges", "my-study/judges.json",
             "--value", "my-study/value.json")
        step("import-reviewer", *REVIEWER_IMPORT)
        step("import-reviewer-again", *REVIEWER_IMPORT)
        step("check", "eval", "scores", "check", "my-study", *STUDY_ARGS)
        step("import-ocp", "eval", "scores", "import", "--ocp", EX / "ocp" / "r201.ocp.json",
             EX / "ocp" / "r202.ocp.json", "--rubric", "my-study/rubric.json", "--out", "ocp-study")
        step("check-ocp", "eval", "scores", "check", "ocp-study", *STUDY_ARGS)
    finally:
        os.chdir(before)
    return root, out


def test_the_committed_study_is_what_the_generator_writes(walk):
    """`record_study.py` writes the files under `docs/examples/eval-recording/`, byte for byte."""
    root, _ = walk
    names = sorted(path.name for path in (EX / "study").iterdir())
    assert names == ["judges.json", "pairs.jsonl", "rubric.json", "runs.jsonl", "scores.jsonl", "value.json"]
    assert sorted(path.name for path in (root / "generated").iterdir()) == names
    for name in names:
        assert (root / "generated" / name).read_bytes() == (EX / "study" / name).read_bytes(), name
    for name in ("reviewer.csv", "reviewer-pairs.csv"):
        assert (root / f"my-study-{name}").read_bytes() == (EX / f"study-{name}").read_bytes(), name


def test_the_generated_study_has_the_readings_its_design_gives(walk):
    """Counts worked out from the design, not from the files: 16 inputs on 2 arms; `judge-main` scores 4
    components per output and 2 again on every other output; `judge-cheap` makes one panel call per output and
    a second on every fourth; the reviewer scores 12 outputs, 4 of them twice on 2 components, and corrects
    one slip; `judge-main` compares 8 pairs in both orders and the reviewer 6 pairs."""
    root, _ = walk
    got = report(root / "my-study")
    assert got["ok"] is True and got["problems"] == [] and got["warnings"] == [] and got["mixed_arm"] == []
    main, cheap, reviewer = 32 * 4 + 16 * 2, (32 + 8) * 4, 12 * 4 + 4 * 2 + 1
    assert got["counts"] == {"runs": 32, "readings": main + cheap + reviewer, "readings_read": main + cheap + reviewer - 1,
                             "pairs": 8 * 2 + 6, "pairs_read": 22, "items": 0}
    assert got["not_read"] == [{"table": "readings", "id": "rv.r026.depth", "reason": "replaced",
                                "by": "rv.r026.depth.fixed"}]
    design = {(row["term"], row["judge"], row["dimension"]): row for row in got["design"]}
    for dim, repeated in (("final/correct", 0), ("final/clarity", 0), ("final/depth", 16), ("final/overall", 16)):
        row = design[("sigma", "judge-main", dim)]
        assert (row["count"], row["met"]) == (repeated, repeated > 0), dim
        assert design[("sigma", "judge-cheap", dim)]["count"] == 8, dim
        assert (design[("anchor", None, dim)]["count"], design[("anchor", None, dim)]["state"]) == (12, "weak"), dim
    assert design[("halo", "judge-cheap", None)]["count"] == 8
    assert design[("position", "judge-main", "final/overall")]["count"] == 8


@pytest.mark.parametrize("folder, counts", [
    ("study", {"runs": 32, "readings": 320, "readings_read": 320, "pairs": 16, "pairs_read": 16, "items": 0}),
    ("scales", {"runs": 2, "readings": 20, "readings_read": 19, "pairs": 2, "pairs_read": 2, "items": 3}),
])
def test_each_committed_example_passes_the_checks(folder, counts):
    """`eval rubric check` and `eval scores check` pass on the example folders as they are in the repository.
    The counts of `scales` are counted by hand from its files."""
    study = EX / folder
    extra = ["--items", study / "items.jsonl"] if (study / "items.jsonl").is_file() else []
    extra += ["--value", study / "value.json"] if (study / "value.json").is_file() else []
    code, _, err = lm("eval", "rubric", "check", study / "rubric.json", "--judges", study / "judges.json", *extra)
    assert code == 0, err
    code, out, err = check(study)
    assert code == 0 and out.rstrip().endswith("\nok"), err
    got = report(study)
    assert got["ok"] is True and got["problems"] == [] and got["counts"] == counts


def test_the_ocp_examples_import_as_the_tutorial_says(walk):
    """Two runs come in. The signals that carry a call are two repeats; the two that carry none are read as a
    reading and its correction; the signal that names no component is left out."""
    root, _ = walk
    code, _, err = lm("ocp", "validate", EX / "ocp" / "r201.ocp.json", EX / "ocp" / "r202.ocp.json")
    assert code == 0, err
    got = report(root / "ocp-study", root / "my-study" / "rubric.json", root / "my-study" / "judges.json")
    assert got["ok"] is True
    assert got["counts"] == {"runs": 2, "readings": 6, "readings_read": 5, "pairs": 0, "pairs_read": 0, "items": 0}
    rows = {row["id"]: row for row in C.read_lines(root / "ocp-study" / "scores.jsonl", C.SCHEMA_SCORES)}
    assert (rows["c0901.depth"]["call"], rows["c0902.depth"]["call"]) == ("c0901", "c0902")
    assert "replaces" not in rows["c0901.depth"] and "replaces" not in rows["c0902.depth"]
    assert rows["c0903.clarity"]["value"] == "ok" and rows["h0901.correct"]["judge"] == "reviewer-1"
    assert rows["r202.s2"]["replaces"] == "r202.s1" and len(rows) == 6
    runs = {row["run"]: row for row in C.read_lines(root / "ocp-study" / "runs.jsonl", C.SCHEMA_RUNS)}
    assert (runs["r201"]["input"], runs["r201"]["arm"], runs["r202"]["arm"]) == ("q17", "baseline", "candidate")
    assert runs["r202"]["steps"] == [{"step": "final", "model": {"id": "model-large", "family": "alpha"},
                                      "effort": "high"}]


# ---------------------------------------------------------------------------------------------------
# Blocks of the tutorial against the files and the commands

def test_every_file_block_holds_lines_of_its_example_file(doc):
    blocks = tagged(doc, "file")
    assert len(blocks) >= 20
    for name, block in blocks:
        real = (EX / name).read_text(encoding="utf-8").splitlines()
        rest = iter(real)
        for line in block:
            assert any(line == other for other in rest), f"{name}: not a line of the file, or out of order: {line[:120]}"
    assert {name for name, _ in blocks} >= {
        "record_study.py", "study/runs.jsonl", "study/scores.jsonl", "study/pairs.jsonl", "study/value.json",
        "study-reviewer.csv", "study-reviewer-pairs.csv", "scales/rubric.json", "scales/judges.json",
        "scales/runs.jsonl", "scales/scores.jsonl", "scales/pairs.jsonl", "scales/items.jsonl", "ocp/r201.ocp.json"}


def test_every_json_line_shown_is_a_valid_record(doc):
    """Each line of a `.jsonl` block validates on its own, so a reader can copy any of them."""
    validators = {C.SCHEMA_RUNS: C.validate_run, C.SCHEMA_SCORES: C.validate_score, C.SCHEMA_PAIRS: C.validate_pair,
                  C.SCHEMA_ITEMS: C.validate_item}
    seen = 0
    for name, block in tagged(doc, "file"):
        if name.endswith(".jsonl"):
            for line in block:
                row = json.loads(line)
                validators[row["schema"]](row, name)
                seen += 1
    assert seen >= 30


def test_every_output_block_is_what_the_command_prints(doc, walk):
    _, out = walk
    blocks = tagged(doc, "output")
    assert {name for name, _ in blocks} == set(out), "an output block has no command here, or a command no block"
    for name, block in blocks:
        assert shown_in(block, out[name].splitlines()), f"{name}:\n{out[name]}"


@needs_engine
def test_the_fit_report_shown_is_the_one_check_only_prints(doc, walk, monkeypatch):
    """The part of the fit output above its last two lines does not depend on the sampler: it is what
    `eval fit --check-only` prints, and that command writes nothing."""
    root, _ = walk
    monkeypatch.chdir(root)
    code, out, err = lm("eval", "fit", "my-study", *STUDY_ARGS, "--check-only")
    assert code == 0, err
    assert not (root / "my-study" / "fit.json").exists() or os.environ.get("LOOPMATH_EVAL_SLOW") == "1"
    (block,) = [block for name, block in tagged(doc, "recorded") if name == "fit"]
    assert block[-2].startswith("fit: my-study/fit.json") and block[-1].startswith("converged: yes")
    assert shown_in(block[:-2] + ["..."], out.splitlines()), out
    assert "readings: 376 of 377 used; pairs: 22 of 22 used; 1 rows not used" in out


# ---------------------------------------------------------------------------------------------------
# Field tables against the validators

FULL = {
    C.SCHEMA_RUBRIC: (C.validate_rubric, {
        "schema": C.SCHEMA_RUBRIC, "id": "answer", "version": "1", "steps": ["draft", "final"],
        "components": [{"id": "depth", "title": "Depth", "role": "criterion",
                        "scale": {"kind": "linear", "range": [0, 10]}, "better": "higher", "applies_to": ["final"],
                        "anchors": "anchors/depth.md", "review_minutes": 2, "compare_minutes": 1,
                        "human_only": False}]}),
    C.SCHEMA_JUDGES: (C.validate_judges, {
        "schema": C.SCHEMA_JUDGES, "anchor": ["reviewer-1"], "min_anchor_targets": 6,
        "judges": [{"id": "judge-main", "kind": "model",
                    "model": {"id": "model-large", "family": "alpha", "provider": "example-cloud"},
                    "effort": "high", "prompt": "rubric-prompt-v2", "mode": "panel",
                    "reference": {"shown": True, "version": "ref-1",
                                  "author": {"kind": "verified", "model": {"id": "model-small", "family": "beta"}}},
                    "blinded": True, "cost_usd_per_call": 0.03, "components": ["depth"]},
                   {"id": "reviewer-1", "kind": "person"}]}),
    C.SCHEMA_RUNS: (C.validate_run, {
        "schema": C.SCHEMA_RUNS, "run": "r001", "input": "q01", "arm": "baseline", "split": "dev",
        "steps": [{"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
                   "variant": "prompt-a"}],
        "cost_usd": 0.01, "simulated": False, "refs": {"output": "outputs/r001.md"}}),
    C.SCHEMA_SCORES: (C.validate_score, {
        "schema": C.SCHEMA_SCORES, "id": "c0001.depth", "run": "r001", "step": "final", "component": "depth",
        "judge": "judge-main", "call": "c0001", "value": 6.5, "rubric": "answer@1",
        "observed_at": "2026-03-02T09:00:00+00:00", "replaces": "c0000.depth", "assignment": "lb-001",
        "cost_usd": 0.03}),
    C.SCHEMA_PAIRS: (C.validate_pair, {
        "schema": C.SCHEMA_PAIRS, "id": "c0002", "a": "r001", "b": "r002", "step": "final", "component": "overall",
        "judge": "judge-main", "call": "c0002", "shown_first": "a", "verdict": "tie", "rubric": "answer@1",
        "observed_at": "2026-03-02T09:00:20+00:00", "replaces": "c0001", "assignment": "lb-002", "cost_usd": 0.05}),
    C.SCHEMA_ITEMS: (C.validate_item, {
        "schema": C.SCHEMA_ITEMS, "id": "q01.deadline", "input": "q01", "component": "covers",
        "label": "States the deadline", "source": {"kind": "model_verified", "model": "model-large"}}),
    C.SCHEMA_VALUE: (C.validate_value, {
        "schema": C.SCHEMA_VALUE, "weights": {"depth": 2}, "gated_value": 0,
        "gates": [{"component": "correct", "at_least": 1, "step": "final"}],
        "objective": {"kind": "best_value", "margin": 0.1, "reference": "baseline", "cost_cap_usd": 1.0}}),
}
FREE = ("ext", "refs", "weights")       # objects whose member names are the user's own


def holder(obj: dict, path: str) -> tuple[dict, str]:
    """The object that holds the last name of a dotted path, and that name. `name[]` is the first entry of a list."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[part[:-2]][0] if part.endswith("[]") else obj[part]
        assert isinstance(obj, dict), path
    return obj, parts[-1]


def code_of(validate, obj: dict) -> str | None:
    """The code of the refusal a validator gives an object, or None when it takes it."""
    try:
        validate(obj)
    except C.ContractError as exc:
        return exc.code
    return None


def paths(obj: object, base: str = "") -> set[str]:
    """Every member path of a JSON value, in the tutorial's notation."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for name, value in obj.items():
            here = f"{base}.{name}" if base else name
            found.add(here)
            if name not in FREE:
                found |= paths(value, here)
    elif isinstance(obj, list) and any(isinstance(one, dict) for one in obj):
        for one in obj:
            found |= {base + "[]" + path[len(base):] for path in paths(one, base)}
    return found


def field_tables(doc: str) -> dict[str, list[list[str]]]:
    tables = {schema: table(rows) for schema, rows in tagged(doc, "fields")}
    assert set(tables) == set(FULL), "one field table per format"
    return tables


def test_the_full_examples_behind_the_field_tables_are_valid():
    for schema, (validate, full) in FULL.items():
        assert code_of(validate, copy.deepcopy(full)) is None, schema


@pytest.mark.parametrize("schema", sorted(FULL))
def test_every_member_a_field_table_names_is_known_to_the_validator(doc, schema):
    """A name the format does not have is refused as an unknown member. So: each documented name, set to null in
    a full valid record, must not be refused that way, and a made-up name in the same place must be."""
    validate, full = FULL[schema]
    rows = field_tables(doc)[schema]
    assert rows[0][0] == "`schema`" and schema in rows[0][3]
    for row in rows:
        path = named(row[0])
        obj = copy.deepcopy(full)
        where, name = holder(obj, path)
        where[name.removesuffix("[]")] = None
        assert code_of(validate, obj) != "unknown_member", f"{schema}: {path} is not a member"
        obj = copy.deepcopy(full)
        where, name = holder(obj, path)
        where[name.removesuffix("[]") + "_x"] = None
        assert code_of(validate, obj) == "unknown_member", f"{schema}: the check does not reach {path}"


@pytest.mark.parametrize("schema", sorted(FULL))
def test_every_required_cell_is_what_the_validator_does(doc, schema):
    """Left out of a full valid record, a member marked `yes` is refused as missing and a member marked `no` is
    not missed. Cells that state a condition are not checked here."""
    validate, full = FULL[schema]
    checked = 0
    for row in field_tables(doc)[schema]:
        path, required = named(row[0]), row[2]
        obj = copy.deepcopy(full)
        where, name = holder(obj, path)
        where.pop(name.removesuffix("[]"), None)
        if required == "yes":
            code = code_of(validate, obj)
            assert code is not None if path == "schema" else code == "missing_member", f"{schema}: {path}: {code}"
        elif required.startswith("no"):
            assert code_of(validate, obj) is None, f"{schema}: {path} is marked optional and is missed"
        else:
            continue
        checked += 1
    assert checked >= 5


def test_the_field_tables_leave_no_member_of_the_examples_out(doc):
    """Every member used by the full records above and by the committed example files has a row."""
    tables = {schema: {named(row[0]).replace("[]", "") for row in rows} for schema, rows in field_tables(doc).items()}
    used = {schema: paths(full) for schema, (_, full) in FULL.items()}
    for folder in ("study", "scales"):
        for name, schema in (("rubric.json", C.SCHEMA_RUBRIC), ("judges.json", C.SCHEMA_JUDGES),
                             ("value.json", C.SCHEMA_VALUE)):
            if (EX / folder / name).is_file():
                used[schema] |= paths(json.loads((EX / folder / name).read_text(encoding="utf-8")))
        for name, schema in ((C.RUNS_FILE, C.SCHEMA_RUNS), (C.SCORES_FILE, C.SCHEMA_SCORES),
                             (C.PAIRS_FILE, C.SCHEMA_PAIRS), (C.ITEMS_FILE, C.SCHEMA_ITEMS)):
            if (EX / folder / name).is_file():
                for line in (EX / folder / name).read_text(encoding="utf-8").splitlines():
                    used[schema] |= paths(json.loads(line))
    for schema, names in used.items():
        missing = {path.replace("[]", "") for path in names} - tables[schema]
        assert not missing, f"{schema}: no row for {sorted(missing)}"


# ---------------------------------------------------------------------------------------------------
# CSV columns against the import

CSV_ROW = {
    "scores": {"run": "r102", "step": "final", "component": "depth", "judge": "reviewer-1", "call": "h0900",
               "value": "6", "item": "", "of": "", "observed_at": "2026-03-02T15:00:00+00:00", "id": "h0900",
               "replaces": "", "assignment": "", "rubric": "report@2", "cost_usd": "0"},
    "pairs": {"a": "r101", "b": "r102", "step": "final", "component": "overall", "judge": "reviewer-1",
              "call": "h0901", "shown_first": "a", "verdict": "", "choice": "right",
              "observed_at": "2026-03-02T15:01:00+00:00", "id": "h0901", "replaces": "", "assignment": "",
              "rubric": "report@2", "cost_usd": "0"},
}


def import_csv(tmp_path: Path, kind: str, row: dict) -> tuple[int, dict]:
    """Import one CSV row into a new copy of `scales`: the exit code and the JSON the command prints."""
    study = tmp_path / f"study-{len(list(tmp_path.iterdir()))}"
    shutil.copytree(EX / "scales", study)
    sheet = study.with_suffix(".csv")
    with open(sheet, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    code, out, _ = lm("eval", "scores", "import", "--csv" if kind == "scores" else "--pairs-csv", sheet,
                      "--rubric", study / "rubric.json", "--out", study, "--json")
    got = json.loads(out)
    if code == 0:
        assert check(study)[0] == 0
    return code, got


@pytest.mark.parametrize("kind", ["scores", "pairs"])
def test_every_csv_column_the_tutorial_names_is_read(doc, kind, tmp_path):
    """A sheet with every documented column comes in whole, with no column ignored. A column of another name is
    reported as ignored. A row without a column marked `yes` does not come in."""
    (rows,) = [table(rows) for name, rows in tagged(doc, "columns") if name == kind]
    columns = [named(row[0]) for row in rows]
    known = (S.SCORE_COLUMNS if kind == "scores" else S.PAIR_COLUMNS + ("choice",)) + S.MORE_COLUMNS
    assert sorted(columns) == sorted(known)
    counted = "readings" if kind == "scores" else "pairs"
    row = {column: CSV_ROW[kind][column] for column in columns}
    code, got = import_csv(tmp_path, kind, row)
    assert code == 0 and got[counted]["new"] == 1 and "ignored_columns" not in got, got
    code, got = import_csv(tmp_path, kind, {**row, "confidence": "high"})
    assert code == 0 and list(got["ignored_columns"].values()) == [["confidence"]], got
    for cells in rows:
        if cells[1] == "yes":
            code, got = import_csv(tmp_path, kind, {k: v for k, v in row.items() if k != named(cells[0])})
            assert code != 0 or got[counted]["new"] == 0, f"{cells[0]} is marked required and the row came in"


def test_a_choice_is_read_by_the_output_shown_first(tmp_path):
    """`left` is the output shown first and `right` the other one, as the tutorial says."""
    for first, choice, verdict in (("a", "left", "a"), ("a", "right", "b"), ("b", "left", "b"), ("b", "tie", "tie")):
        folder = tmp_path / f"{first}-{choice}"
        folder.mkdir()
        code, got = import_csv(folder, "pairs", {**CSV_ROW["pairs"], "shown_first": first, "choice": choice})
        assert code == 0, got
        pairs = C.read_lines(folder / "study-0" / "pairs.jsonl", C.SCHEMA_PAIRS)
        assert [row["verdict"] for row in pairs if row["id"] == "h0901"] == [verdict]


# ---------------------------------------------------------------------------------------------------
# The mistakes the tutorial names, made on a copy of `scales`

def broken(tmp_path: Path, change, source: str = "scales") -> Path:
    """A copy of an example study with one change made to its tables."""
    study = tmp_path / "study"
    shutil.copytree(EX / source, study)
    names = [name for name in ("runs", "scores", "pairs", "items") if (study / f"{name}.jsonl").is_file()]
    tables = {name: [json.loads(line) for line in (study / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in names}
    change(tables)
    for name in names:
        (study / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tables[name]),
                                             encoding="utf-8")
    return study


def score(n: int):
    """A change to line `n` of `scales/scores.jsonl` (1 is the first line)."""
    def with_change(**members):
        def change(tables):
            row = tables["scores"][n - 1]
            for name, value in members.items():
                if value is None:
                    del row[name]
                else:
                    row[name] = value
        return change
    return with_change


def _another(tables, **members):
    tables["scores"].append({**tables["scores"][3], **members})


def _other_input(tables):
    tables["runs"].append({**tables["runs"][1], "run": "r103", "input": "doc-02"})
    tables["pairs"][0]["b"] = "r103"


# name in the tutorial: (the change, the code of the refusal)
MISTAKES = {
    "call-twice": (lambda t: _another(t, id="c0004.depth.again", value=8), "call_twice"),
    "id-twice": (lambda t: _another(t, call="c0099"), "duplicate"),
    "no-call": (score(4)(call=None), "missing_member"),
    "no-id": (score(4)(id=None), "missing_member"),
    "no-rubric": (score(4)(rubric=None), "missing_member"),
    "no-time": (score(4)(observed_at=None), "missing_member"),
    "bad-time": (score(4)(observed_at="2026-03-02T09:01:00"), "bad_time"),
    "binary-word": (score(1)(value="pass"), "bad_value"),
    "level-case": (score(2)(value="Good"), "bad_value"),
    "level-text": (score(3)(value="4"), "bad_value"),
    "fraction-percent": (score(6)(value=80), "bad_value"),
    "linear-range": (score(4)(value=75), "bad_value"),
    "unknown-judge": (score(4)(judge="judge-new"), "unknown_judge"),
    "unknown-component": (score(4)(component="Depth"), "unknown_component"),
    "unknown-run": (score(4)(run="r999"), "no_run"),
    "rubric-version": (score(4)(rubric="report@1"), "unknown_rubric"),
    "extra-member": (score(4)(model="model-large"), "unknown_member"),
    "items-both": (lambda t: t["scores"].append({k: v for k, v in {**t["scores"][7], "id": "c0008.covers", "value": 2,
                                                                  "of": 3}.items() if k != "item"}),
                   "items_and_count"),
    "unknown-item": (score(8)(item="doc-01.none"), "unknown_item"),
    "pair-inputs": (_other_input, "different_inputs"),
    "replaces-unknown": (score(20)(replaces="h0000"), "replaces_unknown"),
}


def refusal(study: Path) -> tuple[list[str], str]:
    """What `eval scores check` prints on stderr for a broken study, and the code of the refusal. A line that is
    not a valid record is refused when the file is read; anything else is a problem of the check's report."""
    code, _, err = check(study)
    assert code == 1, err
    try:
        C.read_study(study)
    except C.ContractError as exc:
        return err.splitlines(), exc.code
    (problem,) = report(study)["problems"]
    return err.splitlines(), problem["code"]


def test_every_refusal_shown_is_the_message_the_check_prints(doc, tmp_path):
    blocks = tagged(doc, "refusal")
    used = [name.strip() for names, _ in blocks for name in names.split(",")]
    assert sorted(used) == sorted(MISTAKES), "each mistake is shown once"
    for names, block in blocks:
        real = []
        for name in (name.strip() for name in names.split(",")):
            change, code = MISTAKES[name]
            folder = tmp_path / name
            folder.mkdir()
            lines, got = refusal(broken(folder, change))
            assert got == code, (name, got, lines)
            assert len(lines) == 1 and lines[0].startswith("error: "), (name, lines)
            real += lines
        assert block == real, names


def test_whole_number_levels_are_refused_as_text_and_as_a_decimal(tmp_path):
    """The tutorial says to write a number level as `4`, not `"4"` and not `4.0`."""
    for n, value in enumerate(("4", 4.0)):
        folder = tmp_path / str(n)
        folder.mkdir()
        assert refusal(broken(folder, score(3)(value=value)))[1] == "bad_value"
    assert check(broken(tmp_path, score(3)(value=4)))[0] == 0


def _own_calls(tables):
    for n, row in enumerate(tables["scores"]):
        if row["judge"] == "judge-cheap":
            row["call"] = f"own-{n}"


def _tenth(tables):
    for row in tables["scores"]:
        if row["judge"] == "judge-cheap" and row["component"] == "depth":
            row["value"] = round(row["value"] / 10, 2)


def _arm_in_run_id(tables):
    new = {row["run"]: f'{row["arm"]}-{row["run"]}' for row in tables["runs"]}
    for row in tables["runs"]:
        row["run"] = new[row["run"]]
    for row in tables["scores"]:
        row["run"] = new[row["run"]]
    for row in tables["pairs"]:
        row["a"], row["b"] = new[row["a"]], new[row["b"]]


@pytest.mark.parametrize("change", [_own_calls, _tenth, _arm_in_run_id], ids=["panel-calls-split", "wrong-scale-in-range",
                                                                               "arm-in-run-id"])
def test_what_the_tutorial_says_the_check_lets_through_passes(change, tmp_path):
    """The tutorial warns that these pass. If the check learns to refuse one, the warning goes."""
    code, out, err = check(broken(tmp_path, change, "study"))
    assert code == 0 and out.rstrip().endswith("\nok"), err


def test_a_misspelt_judge_passes_only_without_the_judges_file(tmp_path):
    study = broken(tmp_path, score(4)(judge="judge-new"))
    assert lm("eval", "scores", "check", study, "--rubric", study / "rubric.json")[0] == 0
    assert check(study)[0] == 1


def test_a_shared_call_and_a_mixed_arm_are_named_and_the_check_still_passes(tmp_path):
    """A call id used by two judges is a warning. An arm whose runs differ in a setting is listed."""
    def change(tables):
        tables["scores"][6]["call"] = "c0011"           # a `judge-main` reading under the call of a `judge-cheap` panel
        steps = copy.deepcopy(tables["runs"][0]["steps"])
        steps[1]["effort"] = "low"
        tables["runs"].append({**tables["runs"][0], "run": "r103", "input": "doc-02", "steps": steps})
    study = broken(tmp_path, change)
    code, out, _ = check(study)
    assert code == 0
    assert 'warning: scores.jsonl: call "c0011": shared by 2 pairs of judge and target' in out
    assert "mixed_arm: short-prompt: 2 settings over 2 runs" in out


@needs_engine
def test_the_fit_leaves_out_a_row_the_check_refuses(tmp_path):
    """`eval fit` counts a reading by an unknown judge as not used, where `eval scores check` refuses it."""
    def change(tables):
        tables["scores"][0]["judge"] = "judge-new"
    study = broken(tmp_path, change, "study")
    assert check(study)[0] == 1
    code, out, err = lm("eval", "fit", study, "--rubric", study / "rubric.json", "--judges", study / "judges.json",
                        "--check-only")
    assert code == 0, err
    assert "readings: 319 of 320 used; pairs: 16 of 16 used; 1 rows not used" in out


# ---------------------------------------------------------------------------------------------------
# The text itself

def real_parser() -> argparse.ArgumentParser:
    """The top-level parser, as `cli.main` builds it."""
    class Built(Exception):
        pass

    def grab(self, *args, **kwargs):
        raise Built(self)

    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except Built as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise AssertionError("cli.main did not build a parser")


def written_commands(text: str) -> list[str]:
    """Every `loopmath ...` line of the code blocks of a document."""
    found, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside and line.lstrip("$ ").startswith("loopmath "):
            found.append(line.lstrip("$ ").strip())
    return found


TUTORIAL_COMMANDS = ("eval rubric check", "eval scores import", "eval scores check", "eval fit", "eval judges",
                     "eval value", "eval labels next")
GUIDE_COMMANDS = ("eval rubric check", "eval scores import", "eval scores check")     # the guide stops at the check


@pytest.mark.parametrize("path, at_least, named", [(DOC, 10, TUTORIAL_COMMANDS), (GUIDE, 5, GUIDE_COMMANDS)],
                         ids=["tutorial", "guide"])
def test_every_command_shown_parses(path, at_least, named, capsys):
    top = real_parser()
    lines = written_commands(path.read_text(encoding="utf-8"))
    assert len(lines) >= at_least, lines
    for line in lines:
        try:
            top.parse_args(shlex.split(line)[1:])
        except SystemExit:
            raise AssertionError(f"{path.name}: does not parse: {line}: {capsys.readouterr().err.strip()[-300:]}")
    shown = " ".join(lines)
    for command in named:
        assert command in shown, f"{path.name} shows no {command}"


def slug(heading: str) -> str:
    """The anchor a Markdown heading gets: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def test_every_link_in_the_tutorial_lands(doc):
    """Links to a heading of this page or of `eval.md` name a real heading, and linked files are there."""
    anchors = {"": {slug(line.lstrip("#")) for line in doc.splitlines() if line.startswith("#")}}
    links = re.findall(r"\]\(([^)\s]+)\)", doc)
    assert len(links) >= 10
    for link in links:
        file, _, anchor = link.partition("#")
        if file:
            assert (DOC.parent / file).exists(), link
        if anchor:
            if file not in anchors:
                other = (DOC.parent / file).read_text(encoding="utf-8")
                anchors[file] = {slug(line.lstrip("#")) for line in other.splitlines() if line.startswith("#")}
            assert anchor in anchors[file], link


def test_the_tutorial_says_eval_is_experimental_and_links_the_limits(doc):
    assert "experimental in 0.4.4" in doc and "(eval.md#limits)" in doc
    assert "## Limits" in (DOC.parent / "eval.md").read_text(encoding="utf-8")


def test_no_em_dash_in_the_tutorial_the_guide_or_the_examples():
    files = [DOC, GUIDE, Path(__file__)] + [path for path in EX.rglob("*") if path.is_file()]
    assert len(files) >= 18
    for path in files:
        assert EM_DASH not in path.read_text(encoding="utf-8"), path


# ---------------------------------------------------------------------------------------------------
# The agent guide

def test_the_guide_examples_are_valid_records(guide, tmp_path):
    """Each JSON block of the guide validates under the format its `schema` names, and its blocks together make
    a study that passes `eval rubric check` and `eval scores check`."""
    validators = {C.SCHEMA_RUBRIC: C.validate_rubric, C.SCHEMA_JUDGES: C.validate_judges,
                  C.SCHEMA_RUNS: C.validate_run, C.SCHEMA_SCORES: C.validate_score, C.SCHEMA_PAIRS: C.validate_pair,
                  C.SCHEMA_ITEMS: C.validate_item, C.SCHEMA_VALUE: C.validate_value}
    files = {C.SCHEMA_RUBRIC: "rubric.json", C.SCHEMA_JUDGES: "judges.json", C.SCHEMA_RUNS: C.RUNS_FILE,
             C.SCHEMA_SCORES: C.SCORES_FILE, C.SCHEMA_PAIRS: C.PAIRS_FILE, C.SCHEMA_ITEMS: C.ITEMS_FILE}
    blocks = re.findall(r"```json\n(.*?)```", guide, flags=re.S)
    assert len(blocks) >= 5
    study, seen = tmp_path / "study", set()
    study.mkdir()
    for block in blocks:
        try:
            rows = [json.loads(block)]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in block.splitlines() if line.strip()]
        for row in rows:
            validators[row["schema"]](row)
            seen.add(row["schema"])
            with open(study / files[row["schema"]], "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    assert seen >= {C.SCHEMA_RUBRIC, C.SCHEMA_JUDGES, C.SCHEMA_RUNS, C.SCHEMA_SCORES, C.SCHEMA_PAIRS}
    code, _, err = lm("eval", "rubric", "check", study / "rubric.json", "--judges", study / "judges.json")
    assert code == 0, err
    code, out, err = check(study)
    assert code == 0 and out.rstrip().endswith("\nok"), err


def test_the_guide_names_the_rules_the_tutorial_gives(guide):
    """The agent version carries the same rules: experimental, one call id per call, repeats and corrections,
    neutral run ids, the check before the fit."""
    assert guide.startswith("# ") and "experimental" in guide and "loopmath guide eval" in guide
    for words in ("run `--help`", "`call`", "`replaces`", "repeat", "neutral", "`ext`", "eval scores check",
                  "`cost_usd`", "`cost_usd_per_call`", "## Never"):
        assert words in guide, words


def test_the_guide_gives_the_refusals_their_real_codes_and_words(guide, tmp_path):
    """Each fix the guide lists starts from words the check really prints for that mistake."""
    quoted = re.findall(r"^\| `([^`]+)` \|", guide, flags=re.M)
    assert len(quoted) >= 8, quoted
    printed = []
    for name, (change, _) in MISTAKES.items():
        folder = tmp_path / name
        folder.mkdir()
        printed += refusal(broken(folder, change))[0]
    for words in quoted:
        assert any(words in line for line in printed), f"the check prints no message with: {words}"


# ---------------------------------------------------------------------------------------------------
# The fit of the walk (slow: it samples)

@pytest.mark.slow
@needs_engine
def test_the_walk_fits_and_the_recorded_outputs_have_their_shape(doc, walk, monkeypatch, tmp_path):
    """The fit, the judges table, the value and the reviewer plan of the walk, run as written. The blocks marked
    `recorded` are compared with every number masked: the sampler does not give the same draws on every machine."""
    root, _ = walk
    recorded = dict(tagged(doc, "recorded"))
    assert set(recorded) == {"fit", "judges", "value", "labels", "not-blind"}
    monkeypatch.chdir(root)
    code, out, err = lm("eval", "fit", "my-study", *STUDY_ARGS, "--chains", "2")
    assert code == 0 and "converged: yes" in out, (out[-400:], err[-400:])
    assert shown_in(masked(recorded["fit"]), masked(out.splitlines())), out
    code, out, err = lm("eval", "judges", "my-study/fit.json")
    assert code == 0, err
    assert shown_in(masked(recorded["judges"]), masked(out.splitlines())), out
    code, out, err = lm("eval", "value", "my-study/fit.json", "--value", "my-study/value.json")
    assert code == 0, err
    assert masked(recorded["value"])[:-1] == masked(out.splitlines())[:-1], out   # the last line holds the decision
    assert out.splitlines()[-1].startswith("  candidate against baseline: value ")
    code, out, err = lm("eval", "labels", "next", "my-study/fit.json", "--mode", "pilot", "--n", "6", "--out",
                        "my-study/labels-001.json", "--form", "labels-001.csv")
    assert code == 0, err
    assert masked(recorded["labels"])[1:] == masked(out.splitlines())[1:], out
    with open("labels-001.csv", encoding="utf-8", newline="") as handle:
        form = list(csv.DictReader(handle))
    plan = (root / "my-study" / "labels-001.json").read_text(encoding="utf-8")
    assert form and set(form[0]) == {"assignment", "run", "step", "component", "judge", "item", "of", "value",
                                     "observed_at"}
    assert all(re.fullmatch(r"r\d{3}", row["run"]) for row in form)
    for word in ("baseline", "candidate", "model-small", "model-large", "alpha", "beta"):
        assert word not in plan, f"the plan names {word}"

    # the same study under run ids that name the arm: the check passes, the plan is refused
    named_study = broken(tmp_path, _arm_in_run_id, "study")
    assert check(named_study)[0] == 0
    code, out, err = lm("eval", "fit", named_study, "--rubric", named_study / "rubric.json", "--judges",
                        named_study / "judges.json", "--components", "depth", "--chains", "2", "--draws", "300")
    assert code == 0, err
    code, out, err = lm("eval", "labels", "next", named_study / "fit.json", "--mode", "pilot", "--n", "6", "--out",
                        tmp_path / "labels.json", "--allow-unconverged")
    assert code == 1 and masked(err.splitlines()) == masked(recorded["not-blind"]), err
