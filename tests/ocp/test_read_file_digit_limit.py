"""A file whose JSON holds an integer with more digits than Python reads from text is an unreadable file: E001,
as for malformed JSON, from read_file, validate_file, validate_files, the checker's command line and
`loopmath ocp validate`."""

from __future__ import annotations

import json
import sys

import pytest

from loopmath.cli import main
from loopmath.ocp import conformance

from ._common import V03_EXAMPLES

LIMIT = 4300  # CPython's default limit on the digits of an integer read from text
SOLO = V03_EXAMPLES / "solo.ocp.json"


@pytest.fixture(autouse=True)
def digit_limit():
    """The default limit for each test, whatever the environment set, put back after."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(LIMIT)
    yield
    sys.set_int_max_str_digits(old)


def _with_digits(tmp_path, digits, name="long.ocp.json"):
    """The solo example with one more score signal whose value is a `digits`-digit integer, written as text so the
    integer is never built here."""
    doc = json.loads(SOLO.read_text(encoding="utf-8"))
    doc["run"]["signals"].append({"id": "sig_long", "kind": "score", "name": "quality", "value": "@VALUE@",
                                  "observed_at": "2026-09-23T11:00:00-07:00", "tier": "reported"})
    path = tmp_path / name
    path.write_text(json.dumps(doc).replace('"@VALUE@"', "1" + "0" * (digits - 1)), encoding="utf-8")
    return path


def test_the_files_sit_on_either_side_of_the_limit(tmp_path):
    with pytest.raises(ValueError) as caught:
        json.loads(_with_digits(tmp_path, LIMIT + 1).read_text(encoding="utf-8"))
    assert not isinstance(caught.value, json.JSONDecodeError)
    doc = json.loads(_with_digits(tmp_path, LIMIT, "at.ocp.json").read_text(encoding="utf-8"))
    assert doc["run"]["signals"][-1]["value"] == 10 ** (LIMIT - 1)


def test_read_file_gives_one_e001(tmp_path):
    path = _with_digits(tmp_path, LIMIT + 1)
    doc, finding = conformance.read_file(path)
    assert doc is None and (finding.level, finding.code, finding.path) == ("error", "E001", "$")
    assert finding.message.startswith(f"cannot read/parse {path}: ")
    assert conformance.validate_file(path) == [finding]


def test_an_integer_at_the_limit_is_read_and_checked(tmp_path):
    path = _with_digits(tmp_path, LIMIT)
    doc, finding = conformance.read_file(path)
    assert finding is None and doc["run"]["signals"][-1]["value"] == 10 ** (LIMIT - 1)
    assert [f.code for f in conformance.validate_file(path)] == ["E195"]  # read, then refused by name


def test_the_file_beside_it_is_still_checked(tmp_path):
    first, second = conformance.validate_files([_with_digits(tmp_path, LIMIT + 1), SOLO])
    assert [(f.level, f.code, f.path) for f in first] == [("error", "E001", "$")] and second == []


def test_the_checkers_command_line_fails_the_file(tmp_path, capsys):
    path = _with_digits(tmp_path, LIMIT + 1)
    assert conformance.main([str(path), str(SOLO)]) == 1
    out, err = capsys.readouterr()
    assert f"FAIL  {path}  (1 errors, 0 warnings)\n  error E001 $: cannot read/parse {path}: " in out
    assert f"PASS  {SOLO}  (0 errors, 0 warnings)" in out and "Traceback" not in out + err


@pytest.mark.parametrize("kind", ["long integer", "malformed"])
def test_ocp_validate_reads_it_as_any_unparseable_file(kind, tmp_path, capsys):
    if kind == "long integer":
        path = _with_digits(tmp_path, LIMIT + 1)
    else:
        path = tmp_path / "broken.ocp.json"
        path.write_text("{", encoding="utf-8")
    assert main(["ocp", "validate", str(path), str(SOLO)]) == 1  # an existing file that cannot be parsed
    out, err = capsys.readouterr()
    assert f"FAIL  {path}  (not JSON; 1 error, 0 warnings)\n  error E001 $: cannot read/parse {path}: " in out
    assert out.endswith("1 passed, 1 failed\n") and "Traceback" not in out + err
    assert main(["ocp", "validate", "--json", str(path), str(SOLO)]) == 1
    first, second = json.loads(capsys.readouterr().out)["files"]
    assert (first["ok"], first["ocp"], first["errors"], first["warnings"]) == (False, None, 1, 0)
    assert [(f["level"], f["code"], f["path"]) for f in first["findings"]] == [("error", "E001", "$")]
    assert second["ok"] is True
    assert main(["ocp", "validate", str(tmp_path / "nope.json"), str(path)]) == 2  # a missing path still exits 2
