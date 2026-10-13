"""Guards for six facts this repository writes down twice, and proof each fails.

The merge lane inventoried six pairs of duplicated content with nothing
comparing the copies: the project version, the protocol version the package
layout spike encodes, the price table's rate keys, the aggregate graph token
names, the conformance checker's closed vocabularies, and the schema
descriptions that are meant to name every value their enumeration allows. Each
guard below reads every copy from its real home and fails when they disagree.
Each guard is paired with a test that runs the same comparison over a desynced
copy, so no guard sits here looking like coverage while being unable to fail;
the commit message carries the transcript of every guard failing against a
desynced working tree.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import tomllib
from pathlib import Path

import loopmath
from loopmath import price
from loopmath.graph import ocp_support

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec"
SPIKE = ROOT / "layout-spike" / "ocp"
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_INIT = ROOT / "src" / "loopmath" / "__init__.py"
PRICE_SOURCE = ROOT / "src" / "loopmath" / "price.py"
PRICE_VALIDATION_SOURCE = ROOT / "src" / "loopmath" / "price_validation.py"
SPIKE_PYPROJECT = SPIKE / "pyproject.toml"
SPIKE_INIT = SPIKE / "ocp" / "__init__.py"
SCHEMA_FILES = ("ocp-v0.schema.json", "ocp-v0.2.schema.json")

_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_duplicate_guards", SPEC / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def _disagreements(copies: dict[str, object]) -> list[str]:
    """One sentence per copy that differs from the first, naming both values.

    Every guard reports through this, so a failure says which copy drifted and
    what each side holds, not just that two objects were unequal.
    """
    (reference_name, reference), *rest = copies.items()
    return [
        f"{name} is {value!r}, but {reference_name} is {reference!r}"
        for name, value in rest
        if value != reference
    ]


def _module_literal(source: str, name: str) -> object:
    """The literal assigned to module-level `name` in `source`.

    Read from text, not from an import, so a guard can be shown failing on a
    desynced copy of the file without touching the installed package.
    """
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no module-level assignment to {name}")


# (a) The project version: pyproject.toml and src/loopmath/__init__.py.


def _project_version_copies(pyproject_text: str, init_text: str) -> dict[str, object]:
    return {
        "the version in pyproject.toml": tomllib.loads(pyproject_text)["project"]["version"],
        "__version__ in src/loopmath/__init__.py": _module_literal(init_text, "__version__"),
        "the imported loopmath.__version__": loopmath.__version__,
    }


def test_project_version_agrees_with_the_package_version():
    copies = _project_version_copies(
        PYPROJECT.read_text(encoding="utf-8"), PACKAGE_INIT.read_text(encoding="utf-8")
    )

    assert _disagreements(copies) == []


def test_project_version_guard_reports_a_desynced_package_version():
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    released = tomllib.loads(pyproject_text)["project"]["version"]
    desynced_init = PACKAGE_INIT.read_text(encoding="utf-8").replace(
        f'__version__ = "{released}"', '__version__ = "0.9.0.dev0"'
    )
    assert '"0.9.0.dev0"' in desynced_init

    found = _disagreements(_project_version_copies(pyproject_text, desynced_init))

    assert len(found) == 1
    assert "0.9.0.dev0" in found[0]
    assert released in found[0]


# (b) The protocol version the layout spike encodes under layout-spike/ocp/:
# the package's SPEC_VERSION, the schema file name pyproject.toml packages, and
# that schema's own version const and $id.


_SCHEMA_FILE_VERSION = re.compile(r"^ocp-v(\d+(?:\.\d+)?)\.schema\.json$")


def _spike_schema_name(pyproject_text: str) -> str:
    """The single schema file layout-spike/ocp/pyproject.toml ships as package data."""
    package_data = tomllib.loads(pyproject_text)["tool"]["setuptools"]["package-data"]
    [name] = package_data["ocp"]
    return name


def _spike_version_copies(
    pyproject_text: str, init_text: str, schema: dict
) -> dict[str, object]:
    name = _spike_schema_name(pyproject_text)
    in_name = _SCHEMA_FILE_VERSION.match(name)
    assert in_name is not None, f"the packaged schema name carries no version: {name}"
    return {
        "SPEC_VERSION in the spike's ocp/__init__.py": _module_literal(
            init_text, "SPEC_VERSION"
        ),
        "the version in the schema name the spike's pyproject.toml packages": in_name.group(1),
        "the 'ocp' const in the spike's schema": schema["properties"]["ocp"]["const"],
        "the version in the $id of the spike's schema": schema["$id"].rpartition(":")[2],
    }


def test_layout_spike_encodes_one_protocol_version():
    pyproject_text = SPIKE_PYPROJECT.read_text(encoding="utf-8")
    schema_path = SPIKE / "ocp" / _spike_schema_name(pyproject_text)
    assert schema_path.is_file(), f"the spike packages a schema it does not ship: {schema_path}"
    copies = _spike_version_copies(
        pyproject_text,
        SPIKE_INIT.read_text(encoding="utf-8"),
        json.loads(schema_path.read_text(encoding="utf-8")),
    )

    assert _disagreements(copies) == []


def test_layout_spike_version_guard_reports_a_desynced_packaged_schema_name():
    pyproject_text = SPIKE_PYPROJECT.read_text(encoding="utf-8")
    schema_path = SPIKE / "ocp" / _spike_schema_name(pyproject_text)
    desynced_pyproject = pyproject_text.replace(
        f'ocp = ["{schema_path.name}"]', 'ocp = ["ocp-v0.9.schema.json"]'
    )
    assert "ocp-v0.9.schema.json" in desynced_pyproject

    found = _disagreements(
        _spike_version_copies(
            desynced_pyproject,
            SPIKE_INIT.read_text(encoding="utf-8"),
            json.loads(schema_path.read_text(encoding="utf-8")),
        )
    )

    assert len(found) == 1
    assert "'0.9'" in found[0]


# (c) The four price table rate keys: src/loopmath/price.py and
# src/loopmath/price_validation.py. The validators' code lives in price_validation
# and runs on price's globals, so each file writes the tuple out in full.
#
# This pair is read from the source text, not from the two imported modules,
# because price_validation overwrites its own `_RATE_KEYS` with price's on
# import (`_synchronize_from_price`). Comparing the two module attributes
# passes even when the two files disagree, which is the trap this guard exists
# to catch: the third copy below pins the source against what price exports.


def _rate_key_copies(price_source: str, validation_source: str) -> dict[str, object]:
    return {
        "_RATE_KEYS in src/loopmath/price.py": _module_literal(price_source, "_RATE_KEYS"),
        "_RATE_KEYS in src/loopmath/price_validation.py": _module_literal(
            validation_source, "_RATE_KEYS"
        ),
        "_RATE_KEYS in the imported loopmath.price": price._RATE_KEYS,
    }


def test_price_and_price_validation_agree_on_the_rate_keys():
    copies = _rate_key_copies(
        PRICE_SOURCE.read_text(encoding="utf-8"),
        PRICE_VALIDATION_SOURCE.read_text(encoding="utf-8"),
    )

    assert _disagreements(copies) == []
    assert len(price._RATE_KEYS) == 4


def test_rate_key_guard_reports_a_desynced_copy():
    desynced = PRICE_VALIDATION_SOURCE.read_text(encoding="utf-8").replace(
        '_RATE_KEYS = ("input", "cache_read", "cache_write", "output")',
        '_RATE_KEYS = ("input", "cache_read", "cache_write", "out")',
    )
    assert '"cache_write", "out")' in desynced

    found = _disagreements(
        _rate_key_copies(PRICE_SOURCE.read_text(encoding="utf-8"), desynced)
    )

    assert len(found) == 1
    assert "'out'" in found[0]


# (d) The four aggregate graph token names: the pricing tuple in loopmath.price and
# the graph side of the OCP cost mapping in loopmath.graph.ocp_support.


def _graph_token_name_copies(
    token_streams: tuple[str, ...], cost_fields: tuple[tuple[str, str], ...]
) -> dict[str, object]:
    return {
        "_TOKEN_STREAMS in loopmath.price": tuple(token_streams),
        "the graph names in _COST_FIELDS in loopmath.graph.ocp_support": tuple(
            graph_name for graph_name, _ocp_name in cost_fields
        ),
    }


def test_pricing_and_ocp_cost_mapping_agree_on_the_graph_token_names():
    copies = _graph_token_name_copies(price._TOKEN_STREAMS, ocp_support._COST_FIELDS)

    assert _disagreements(copies) == []
    assert len(price._TOKEN_STREAMS) == 4


def test_graph_token_name_guard_reports_a_desynced_copy():
    desynced = (
        ("in", "input_tokens"),
        ("cache_read", "cached_input_tokens"),
        ("cache_creation", "cache_creation_tokens"),
        ("out", "output_tokens"),
    )

    found = _disagreements(_graph_token_name_copies(price._TOKEN_STREAMS, desynced))

    assert len(found) == 1
    assert "'cache_creation'" in found[0]


# (e) The checker's closed vocabularies against the closed enumerations in the
# v0.2 schema: the evidence tiers, and the terminal attempt statuses.


def _closed_vocabulary_copies(
    evidence_tiers: tuple[str, ...], terminal_statuses: set[str], schema: dict
) -> tuple[dict[str, object], dict[str, object]]:
    defs = schema["$defs"]
    tiers = {
        "EVIDENCE_TIERS in spec/ocp_conformance.py": tuple(evidence_tiers),
        "the evidenceTier enum in the v0.2 schema": tuple(defs["evidenceTier"]["enum"]),
    }
    statuses = {
        "TERMINAL_STATUSES in spec/ocp_conformance.py": frozenset(terminal_statuses),
        "the outcome.result enum in the v0.2 schema": frozenset(
            defs["outcome"]["properties"]["result"]["enum"]
        ),
        "the non-queued, non-working attempt.status values in the v0.2 schema": frozenset(
            defs["attempt"]["properties"]["status"]["enum"]
        )
        - {"queued", "working"},
    }
    return tiers, statuses


def test_checker_constants_agree_with_the_closed_schema_enumerations():
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text(encoding="utf-8"))

    tiers, statuses = _closed_vocabulary_copies(
        conf.EVIDENCE_TIERS, conf.TERMINAL_STATUSES, schema
    )

    assert _disagreements(tiers) == []
    assert _disagreements(statuses) == []


def test_checker_constant_guard_reports_a_desynced_schema():
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text(encoding="utf-8"))
    schema["$defs"]["evidenceTier"]["enum"] = ["verified", "heuristic", "asserted"]
    schema["$defs"]["outcome"]["properties"]["result"]["enum"] = [
        "done", "failed", "rejected", "canceled", "settled_unverified",
    ]

    tiers, statuses = _closed_vocabulary_copies(
        conf.EVIDENCE_TIERS, conf.TERMINAL_STATUSES, schema
    )

    assert len(_disagreements(tiers)) == 1
    assert "'asserted'" in _disagreements(tiers)[0]
    assert len(_disagreements(statuses)) == 1
    assert "'lost'" in _disagreements(statuses)[0]


# (f) Every closed enumeration's description names every value it allows. The
# open vocabularies already have this check (test_ocp_conformance.py); the
# closed ones, in both shipped schema versions, did not.


def _closed_enums(schema: dict) -> dict[str, dict]:
    """Every object with an `enum`, keyed by its path in the document."""
    found: dict[str, dict] = {}

    def walk(node, path):
        if isinstance(node, dict):
            if "enum" in node:
                found[path] = node
            for key, child in node.items():
                walk(child, f"{path}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}/{index}")

    walk(schema, "")
    return found


def _values_the_descriptions_do_not_name(schema: dict) -> dict[str, list[str]]:
    """Path -> the enumerated values its own description never names."""
    unnamed = {}
    for path, node in _closed_enums(schema).items():
        words = set(re.findall(r"[A-Za-z0-9_]+", node.get("description", "")))
        absent = [value for value in node["enum"] if value not in words]
        if absent:
            unnamed[path] = absent
    return unnamed


def test_every_closed_enumeration_description_names_its_values():
    unnamed = {
        name: _values_the_descriptions_do_not_name(
            json.loads((SPEC / name).read_text(encoding="utf-8"))
        )
        for name in SCHEMA_FILES
    }

    assert unnamed == {name: {} for name in SCHEMA_FILES}


def test_enumeration_description_guard_reports_an_unnamed_value():
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text(encoding="utf-8"))
    tier = schema["$defs"]["evidenceTier"]
    tier["description"] = tier["description"].replace("verified:", "the top tier:")

    unnamed = _values_the_descriptions_do_not_name(schema)

    assert unnamed == {"/$defs/evidenceTier": ["verified"]}
