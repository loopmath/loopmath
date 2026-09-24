"""Small OCP v0.3 builders shared by the prior converters (lane 11).

The converters write OCP v0.3 documents directly (spec 01, section 2). The
configuration id is lane 1's `loopmath.ocp.canonical.config_id` (Analyst
decision D2); the bundle manifest names it as `config_id_impl`.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..ingest.base import canonical_effort, canonical_model
from ..logmatch.tariff import price_table, table_id  # lane 2
from ..ocp.canonical import config_id  # lane 1 (D2); converters call ocpdoc.config_id

OCP_VERSION = "0.3"
PRODUCER_NAME = "loopmath-prior"
ORG = "loopmath"

# Families under which model versions sit (paper 4.4, tree law). Lane 5 owns the
# forest; this is only the `model.family` field written into the documents.
_FAMILY_PATTERNS = (
    (re.compile(r"^claude-opus-"), "opus", "anthropic"),
    (re.compile(r"^opus-"), "opus", "anthropic"),
    (re.compile(r"^claude-fable-"), "fable", "anthropic"),
    (re.compile(r"^claude-sonnet-"), "sonnet", "anthropic"),
    (re.compile(r"^sonnet-"), "sonnet", "anthropic"),
    (re.compile(r"^claude-haiku-|^haiku-"), "haiku", "anthropic"),
    (re.compile(r"^gpt-[0-9.]+-(sol|terra|luna|astra)\b"), None, "openai"),
    (re.compile(r"^gpt-"), "gpt", "openai"),
)

_EM_DASH = chr(0x2014)  # the character the house rule bans in our own text


def model_ref(raw: str | None, *, tier: str = "reported") -> dict:
    """OCP modelRef: raw label, canonical id, family and provider."""
    ref: dict = {"raw": str(raw or "unknown")}
    mid = canonical_model(raw)
    if mid:
        ref["id"] = mid
        for pattern, family, provider in _FAMILY_PATTERNS:
            m = pattern.match(mid)
            if m:
                ref["family"] = family or m.group(1)
                ref["provider"] = provider
                break
    ref["tier"] = tier
    return ref


def harness_for(family: str | None) -> str:
    """The sweep and RQ1 record the harness family as `claude` or `codex`."""
    return {"claude": "claude-code", "codex": "codex", "openai": "codex"}.get(str(family or ""), str(family or "unknown"))


def effort(raw: str | None) -> str:
    return canonical_effort(raw) or "default"


def setting(harness: str, model: str | None, eff: str | None, **options) -> dict:
    """OCP Setting (spec 01, section 2.4)."""
    out: dict = {"harness": harness}
    if model is not None:
        out["model"] = model_ref(model)
    out["effort"] = effort(eff) if eff is not None else "default"
    out["context_policy"] = "fresh"  # D31
    if options:
        out["options"] = {k: str(v) for k, v in options.items()}
    return out


def clean_text(value: str | None, limit: int = 500) -> str | None:
    """Short text for OCP: no em dashes (house rule), at most `limit` characters."""
    if value is None:
        return None
    text = str(value).replace(" " + _EM_DASH + " ", ", ").replace(_EM_DASH, ", ")
    return text[:limit]


# ---------------------------------------------------------------- tariff and cost
def tariff(path: str | Path | None = None) -> dict:
    """The price table's tariff: lane 2's id (a hash of the table) and the table's as-of date."""
    return {"id": table_id(path), "date": price_table(path).as_of, "source": "loopmath prices.toml"}


def cost_block(model: str | None, tokens: dict, *, reasoning: int | None = None,
               prices: str | Path | None = None, basis: str = "measured") -> dict:
    """OCP cost from four token streams, priced with the packaged price table.

    `tokens` uses the four-stream names `input`, `cache_read`, `cache_write`,
    `output`. `usd` is omitted when the model has no price (never a fake 0).
    """
    table = price_table(prices)
    streams = {k: int(tokens.get(k) or 0) for k in ("input", "cache_read", "cache_write", "output")}
    out: dict = {
        "input_tokens": streams["input"],
        "cached_input_tokens": streams["cache_read"],
        "cache_creation_tokens": streams["cache_write"],
        "output_tokens": streams["output"],
    }
    if reasoning:
        out["reasoning_tokens"] = int(reasoning)
    rate = table.rate(model)
    if rate is not None:
        usd = sum(streams[k] * rate[k] for k in streams) / 1_000_000
        out["usd"] = round(usd, 6)
    out["basis"] = basis
    out["tariff"] = tariff(prices)
    return out


def total_tokens(cost: dict | None) -> int:
    if not cost:
        return 0
    return sum(int(cost.get(k) or 0) for k in
               ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens"))


# ---------------------------------------------------------------- configuration id
CONFIG_ID_IMPL = "loopmath.ocp.canonical.config_id"


def configuration(workflow: dict, settings: dict, *, source: str) -> dict:
    return {"id": config_id(workflow, settings), "workflow": workflow, "settings": settings, "source": source}


# ---------------------------------------------------------------- workflow shapes
def _art(aid: str, kind: str) -> dict:
    return {"id": aid, "kind": kind}


GATE_RULE_DEFAULTS = {"reviewer": "review_approve", "referee": "referee_pick", "tester": "tests_pass"}


def _control(gates: list[tuple[str, str, str | None]], roles: dict[str, str], *, budget: int, rescue: dict) -> dict:
    """OCP control from (after, rule, on_fail) gates, as lane 4 maps it (D29, D30).

    A gate rule goes to `control.ext["dev.loopmath.gate_rules"]` only when it is
    not the default for the judged piece's role (`review_approve` unless the role
    has its own default).
    """
    control = {"gates": [g[0] for g in gates], "repair": {g[0]: g[2] for g in gates if g[2]},
               "budget": max(1, int(budget)), "rescue": rescue}
    rules = {after: rule for after, rule, _ in gates
             if rule != GATE_RULE_DEFAULTS.get(roles.get(after, ""), "review_approve")}
    if rules:
        control["ext"] = {"dev.loopmath.gate_rules": rules}
    return control


def workflow_plan_implement_review(*, budget: int, test_gate: bool = True) -> dict:
    """The catalog `plan_implement_review` shape, plus the sweep's test gate after implement.

    Pieces, artifacts and edges are lane 4's catalog definition, so the only
    difference from the catalog configuration is the extra gate: tests after
    implement (repair to implement) and the reviewer's verdict (a rejection
    sends the work back to implement). Roles use the `types.py` words (D32);
    `budget` is K_max counting the first round (D30).
    """
    roles = {"plan": "planner", "implement": "implementer", "review": "reviewer"}
    gates = ([("implement", "tests_pass", "implement")] if test_gate else []) + [
        ("review", "review_approve", "implement")]
    return {
        "id": "plan_implement_review", "version": 1, "title": "Plan, implement, review",
        "pieces": [{"id": k, "role": v, "width": 1} for k, v in roles.items()],
        "artifacts": [_art("issue", "issue"), _art("repo", "repo"), _art("plan_doc", "plan"),
                      _art("diff", "diff"), _art("verdict", "verdict")],
        "edges": [["issue", "plan"], ["repo", "plan"], ["plan", "plan_doc"], ["issue", "implement"],
                  ["repo", "implement"], ["plan_doc", "implement"], ["implement", "diff"], ["diff", "review"],
                  ["issue", "review"], ["plan_doc", "review"], ["review", "verdict"]],
        "control": _control(gates, roles, budget=budget, rescue={"kind": "none"}),
    }


def workflow_solo() -> dict:
    """The catalog `solo` shape: one implementer, no gate, one round, rescue by the usual configuration."""
    return {
        "id": "solo", "version": 1, "title": "One agent implements",
        "pieces": [{"id": "implement", "role": "implementer", "width": 1}],
        "artifacts": [_art("issue", "issue"), _art("repo", "repo"), _art("diff", "diff")],
        "edges": [["issue", "implement"], ["repo", "implement"], ["implement", "diff"]],
        "control": _control([], {}, budget=1, rescue={"kind": "configuration", "ref": "usual"}),
    }


def run_doc(*, run: dict, nodes: list, attempts: list, edges: list | None = None,
            producer_version: str, emitted_at: str, source_contract: str | None = None) -> dict:
    producer = {"name": PRODUCER_NAME, "version": producer_version, "emitted_at": emitted_at,
                "capabilities": {"task": True, "configuration": True, "signals": True, "slate": False,
                                 "receipt": False, "cost_usd": True, "cost_tokens": True,
                                 "outcome_evidence": True}}
    if source_contract:
        producer["source_contract"] = source_contract
    doc = {
        "ocp": OCP_VERSION,
        "producer": producer,
        "privacy": {"profile": "metadata_only", "note": "loopmath prior bundle; share reduction applied"},
        "run": run,
        "nodes": nodes,
    }
    if edges:
        doc["edges"] = edges
    doc["attempts"] = attempts
    return doc
