"""Every field v0.3 adds, given a malformed value: the checker, migrate and the CLI report, never raise.

Review finding 1 on 9bd45a3. Each path inside a v0.3 field of
full-fields.ocp.json (first element of each array) is replaced by values the
semantic rules must not use as keys: an object, a nested array, null and a
boolean.
"""

from __future__ import annotations

import copy
import json

import pytest

from loopmath.cli import main
from loopmath.ocp import conformance
from loopmath.ocp.migrate import MigrationError, migrate_doc

from ._common import MIGRATE_GOLDEN, example, load

FULL = example("full-fields")
NEW_FIELDS = [("run", k) for k in ("task", "configuration", "provenance", "acceptance_rule", "signals", "slate",
                                   "preferences", "rescue", "receipt")]
NEW_FIELDS += [("nodes", 1, "vertex"), ("nodes", 1, "gate"), ("attempts", 0, "vertex"), ("attempts", 0, "round"),
               ("attempts", 0, "setting"), ("attempts", 0, "cwd"), ("attempts", 0, "cost", "tariff"),
               ("artifacts", 2, "vertex"), ("artifacts", 2, "version"), ("artifacts", 2, "supersedes"),
               ("artifacts", 0, "transfer_tokens")]
BAD = [{"x": {}}, [[]], None, True]


def _get(doc, path):
    for key in path:
        doc = doc[key]
    return doc


def _paths(value, path):
    yield path
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _paths(item, path + (key,))
    elif isinstance(value, list) and value:
        yield from _paths(value[0], path + (0,))


def _all_paths():
    out = []
    for root in NEW_FIELDS:
        out.extend(_paths(_get(FULL, root), root))
    return out


PATHS = _all_paths()


def _with(path, value):
    doc = copy.deepcopy(FULL)
    target = _get(doc, path[:-1])
    target[path[-1]] = copy.deepcopy(value)
    return doc


def test_the_sweep_covers_every_new_field():
    assert all(_get(FULL, root) is not None for root in NEW_FIELDS)
    assert len(PATHS) > 150


@pytest.mark.parametrize("path", PATHS, ids=lambda p: ".".join(map(str, p)))
def test_malformed_value_is_reported_not_raised(path):
    for value in BAD:
        doc = _with(path, value)
        findings = conformance.validate_doc(doc)
        assert all(f.level in ("error", "warning") for f in findings)
        conformance.validate_many([doc, copy.deepcopy(FULL)])
        try:
            migrate_doc(doc, infer=None)
        except MigrationError:
            pass


def test_cli_reports_every_malformed_field(tmp_path, capsys):
    paths = []
    for n, path in enumerate(PATHS):
        file = tmp_path / f"m{n}.ocp.json"
        file.write_text(json.dumps(_with(path, {"x": {}})))
        paths.append(str(file))
    code = main(["ocp", "validate", "--json", *paths])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and len(payload["files"]) == len(paths)
    assert all(row["errors"] >= 1 for row in payload["files"])  # an object where a scalar or list belongs


# The v0.1 and v0.2 rules, found by a whole-document fuzz after the review:
# each path is one hash lookup or membership check in the referential rules.
SWARM = load(MIGRATE_GOLDEN / "swarm-v02.ocp.json")
OLD_PATHS = [("ocp",), ("edges", 2, "kind"), ("edges", 2, "from"), ("edges", 2, "to"), ("edges", 3, "from_attempt"),
             ("edges", 3, "to_attempt"), ("attempts", 0, "node"), ("attempts", 3, "node"), ("attempts", 0, "status"),
             ("attempts", 3, "origin", "launched_by"), ("artifacts", 0, "producer"), ("artifacts", 0, "writers", 0),
             ("artifacts", 0, "consumers", 0), ("groups", 0, "parent")]


@pytest.mark.parametrize("path", OLD_PATHS, ids=lambda p: ".".join(map(str, p)))
@pytest.mark.parametrize("version", ["0.2", "0.3"])
def test_malformed_value_in_the_older_rules_is_reported_not_raised(path, version):
    base = copy.deepcopy(SWARM) if version == "0.2" else migrate_doc(SWARM, infer=None)
    base["groups"] = [{"id": "g1", "title": "group"}]
    for value in BAD:
        doc = copy.deepcopy(base)
        _get(doc, path[:-1])[path[-1]] = copy.deepcopy(value)
        errors = [f for f in conformance.validate_doc(doc) if f.level == "error"]
        assert errors or (value is None and path[-1] == "launched_by"), (path, value)  # null: no launcher
