"""Paper Table 3.1 coverage (spec/OCP.md section 8.5; spec 01 section 5).

Each symbol maps to OCP v0.3 field paths, or is named as computed (derived
from recorded fields), belief (the fitted model) or intentionally absent.
Every field path must exist in the v0.3 schema and be filled in the
full-fields example. Path syntax: dots between properties, `[]` for array
items, `{}` for the values of a map.
"""

from __future__ import annotations

import json
import re

import pytest

from ._common import SPEC, example

SCHEMA = json.loads((SPEC / "ocp-v0.3.schema.json").read_text())
NOT_A_FIELD = {"computed", "belief", "absent (Q4)", "the document"}

TABLE = {
    "tau, P": {"tau": ["run.task.id"], "P": ["computed"]},
    "psi(tau), eta_tau": {"psi(tau)": ["run.task.features", "run.task.groups[].level", "run.task.groups[].id"],
                          "eta_tau": ["belief"]},
    "rho, pi": {"rho": ["run.configuration.workflow.pieces[].role"],
                "pi": ["run.configuration.settings{}.harness", "run.configuration.settings{}.model",
                       "run.configuration.settings{}.effort", "run.configuration.settings{}.context_policy",
                       "run.configuration.settings{}.options"]},
    "W = (V, E, Gamma), q_v, K_max": {
        "V": ["run.configuration.workflow.pieces[].id", "run.configuration.workflow.artifacts[].id"],
        "E": ["run.configuration.workflow.edges"],
        "Gamma": ["run.configuration.workflow.control.gates", "run.configuration.workflow.control.repair",
                  "run.configuration.workflow.control.rescue"],
        "q_v": ["run.configuration.workflow.pieces[].width"],
        "K_max": ["run.configuration.workflow.control.budget"]},
    "x, X(W)": {"x": ["run.configuration.id"], "X(W)": ["computed"]},
    "a, tok_s(a), reads, writes, C(a)": {
        "a": ["attempts[].id", "attempts[].vertex", "attempts[].round"],
        "tok_s(a)": ["attempts[].cost.input_tokens", "attempts[].cost.cached_input_tokens",
                     "attempts[].cost.output_tokens"],
        "reads": ["artifacts[].consumers"], "writes": ["artifacts[].writers"],
        "C(a)": ["attempts[].cost.usd", "attempts[].cost.tariff.id", "attempts[].cost.tariff.date"]},
    "w, h_j": {"w": ["nodes[].vertex", "attempts[].node", "artifacts[].vertex", "edges[].from", "edges[].to"],
               "h_j": ["computed"]},
    "zeta, J_accept, Z_accept": {
        "zeta": ["run.acceptance_rule.requires", "run.acceptance_rule.score.name",
                 "run.acceptance_rule.excludes_events", "run.acceptance_rule.window_days"],
        "J_accept": ["run.signals[].at_attempt", "computed"], "Z_accept": ["run.signals[].value", "computed"]},
    "C_run, C_accept, C_rescue": {"C_run": ["computed"], "C_accept": ["computed"],
                                  "C_rescue": ["run.rescue.kind", "run.rescue.cost_usd", "computed"]},
    "u, kind(u), prod(u), cons(u)": {
        "u": ["artifacts[].id", "artifacts[].version", "artifacts[].supersedes"],
        "kind(u)": ["artifacts[].kind.value", "run.configuration.workflow.artifacts[].kind"],
        "prod(u)": ["artifacts[].producer"], "cons(u)": ["artifacts[].consumers"]},
    "C_production, C_transfer, val(u)": {"C_production": ["computed"],
                                         "C_transfer": ["artifacts[].transfer_tokens", "computed"],
                                         "val(u)": ["belief"]},
    "theta, phi, (Y, Z), P_theta, ell_theta": {"theta": ["belief"], "phi": ["belief"], "(Y, Z)": ["belief"],
                                               "P_theta": ["belief"], "ell_theta": ["belief"]},
    "xi_n, F_n, S_n": {"xi_n": ["the document"], "F_n": ["belief"], "S_n": ["belief"]},
    "d_n, N, V(S), G_n, Net_n": {
        "d_n": ["run.receipt.before.rec", "run.configuration.source"],
        "N": ["absent (Q4)"],
        "V(S)": ["run.receipt.before.gain_per_run", "run.receipt.after.moved"],
        "G_n": ["run.receipt.before.gain_per_run"],
        "Net_n": ["run.receipt.before.price_usd", "run.receipt.before.payback_runs"]},
}


def spec_rows():
    """Row labels of the table in spec/OCP.md section 8.5 (in the public export, D38; design/ is not)."""
    text = (SPEC / "OCP.md").read_text()
    section = text.split("### 8.5 Paper Table 3.1 coverage", 1)[1].split("\n#", 1)[0]
    rows = re.findall(r"^\| ([^|]+?) \| [^|]+ \|$", section, flags=re.M)
    return [r for r in rows if r not in ("Symbol",) and not set(r) <= set("-")]


def _deref(node):
    while isinstance(node, dict) and "$ref" in node:
        node = SCHEMA["$defs"][node["$ref"].rsplit("/", 1)[1]]
    return node


def _children(node, key):
    """Schemas reachable by `key` from `node`, through anyOf / oneOf alternatives."""
    node = _deref(node)
    options = [node] + [_deref(o) for o in node.get("anyOf", []) + node.get("oneOf", [])]
    found = []
    for option in options:
        if key == "[]" and "items" in option:
            found.append(option["items"])
        elif key == "{}" and isinstance(option.get("additionalProperties"), dict):
            found.append(option["additionalProperties"])
        elif key in option.get("properties", {}):
            found.append(option["properties"][key])
    return found


def _steps(path):
    return [s for s in re.split(r"\.|(\[\])|(\{\})", path) if s]


def in_schema(path):
    nodes = [SCHEMA]
    for step in _steps(path):
        nodes = [child for node in nodes for child in _children(node, step)]
        if not nodes:
            return False
    return True


def in_document(doc, path):
    values = [doc]
    for step in _steps(path):
        nxt = []
        for value in values:
            if step == "[]" and isinstance(value, list):
                nxt.extend(value)
            elif step == "{}" and isinstance(value, dict):
                nxt.extend(value.values())
            elif isinstance(value, dict) and step in value:
                nxt.append(value[step])
        values = nxt
    return any(v is not None for v in values)


def test_every_row_of_the_spec_table_is_mapped():
    assert list(TABLE) == spec_rows()


@pytest.mark.parametrize("row", list(TABLE))
def test_every_symbol_maps_to_a_field_or_is_named_as_not_a_field(row):
    full = example("full-fields")
    for symbol, targets in TABLE[row].items():
        assert targets, symbol
        for target in targets:
            if target in NOT_A_FIELD:
                continue
            assert in_schema(target), f"{symbol}: {target} is not a v0.3 schema path"
            assert in_document(full, target), f"{symbol}: {target} is empty in full-fields.ocp.json"


def test_resolver_rejects_unknown_paths():
    assert not in_schema("run.task.nonsense")
    assert not in_schema("attempts[].cost.tariff.unknown")
    assert in_schema("run.configuration.workflow.pieces[].width")
