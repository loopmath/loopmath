"""What the gate reads from a run: its task, arm, harness digest, outcome and dollars.

Runs are OCP run documents (from a store), `loopmath.harness-runs/0.1` rows (from a rows file or a
live wrapper), or observations already made by `loopmath.compare`. The gate hands every one of them
to `compare.paired` as it is, scored or not, so compare pairs first; it reads them here only to count
scored, unscored and unpriced runs per arm and to pick the arms. When `loopmath.compare.observe` is
available it is the reader, so the gate and the comparison never disagree about which runs are scored;
the local reading below follows the same rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

ROWS_SCHEMA = "loopmath.harness-runs/0.1"
GATE_EXT = "dev.loopmath.gate"
PAIR_EXT = "dev.loopmath.pair"
SELECTOR_FORMS = ("digest:sha256:<hex> or sha256:<hex> (a prefix that matches one digest will do), "
                  "a cfg_ configuration id, or arm:<name> or arm=<name>")

try:  # the harness record's ext key, read through its owner when present
    from loopmath.harness import EXT_KEY as HARNESS_KEY
except ImportError:  # pragma: no cover - the record module is not in this tree yet
    HARNESS_KEY = "dev.loopmath.harness"


def is_row(x: Any) -> bool:
    return isinstance(x, dict) and x.get("schema") == ROWS_SCHEMA


def is_doc(x: Any) -> bool:
    return isinstance(x, dict) and isinstance(x.get("run"), dict)


def _run(x: Any) -> dict:
    return x["run"] if is_doc(x) else {}


def _gate_ext(row: dict) -> dict:
    ext = row.get("ext") if isinstance(row.get("ext"), dict) else {}
    return ext.get(GATE_EXT) if isinstance(ext.get(GATE_EXT), dict) else {}


def task_of(x: Any) -> str | None:
    if is_row(x):
        return x.get("task_id")
    if is_doc(x):
        task = _run(x).get("task")
        return task.get("id") if isinstance(task, dict) else None
    return getattr(x, "task", None)


def digest_of(x: Any) -> str | None:
    if is_row(x):
        rec = x.get("harness")
        return rec.get("digest") if isinstance(rec, dict) else None
    if is_doc(x):
        ext = _run(x).get("ext") if isinstance(_run(x).get("ext"), dict) else {}
        for key in (HARNESS_KEY, "harness"):
            rec = ext.get(key)
            if isinstance(rec, dict) and rec.get("digest"):
                return rec["digest"]
        return None
    return getattr(x, "harness", None)


def arm_of(x: Any) -> str | None:
    if is_row(x):
        return x.get("arm") or _gate_ext(x).get("arm")
    if is_doc(x):
        labels = _run(x).get("labels")
        if isinstance(labels, dict) and labels.get("arm"):
            return labels["arm"]
        ext = _run(x).get("ext") if isinstance(_run(x).get("ext"), dict) else {}
        pair = ext.get(PAIR_EXT)
        return pair.get("arm") if isinstance(pair, dict) else None
    return getattr(x, "arm", None)


def name_of(x: Any) -> str:
    """The run's id for a message: the document's or the row's own, else its task and arm."""
    if is_doc(x) and _run(x).get("id"):
        return str(_run(x)["id"])
    if is_row(x):
        tool = x.get("tool") if isinstance(x.get("tool"), dict) else {}
        rid = x.get("id") or x.get("run_id") or tool.get("run_id")
        if rid:
            return str(rid)
    rid = getattr(x, "run", None)
    if rid:
        return str(rid)
    arm = arm_of(x)
    return f"{task_of(x)}" + (f" ({arm})" if arm else "")


def config_of(x: Any) -> str | None:
    if is_doc(x):
        cfg = _run(x).get("configuration")
        return cfg.get("id") if isinstance(cfg, dict) else None
    return None


def usd_of(x: Any) -> float | None:
    """The run's dollars, or None when not known (never 0 for unknown)."""
    if is_row(x):
        cost = x.get("cost") if isinstance(x.get("cost"), dict) else {}
        if cost.get("complete") is False:
            return None  # a partial sum is not the run's cost
        usd = cost.get("usd")
        return float(usd) if isinstance(usd, (int, float)) and not isinstance(usd, bool) else None
    if is_doc(x):
        from ..store.runs import run_cost

        return run_cost(x)["usd"]
    usd = getattr(x, "usd", None)
    return float(usd) if isinstance(usd, (int, float)) else None


def _score_name(rule: str) -> str | None:
    if isinstance(rule, str) and rule.startswith("score:"):
        return rule.split(":")[1]
    return None


def scored(x: Any, rule: Any, observe: Callable | None = None) -> bool:
    """Whether the run has an outcome under `rule`. Unscored runs are left out, never zeros."""
    if observe is not None:
        return observe(x, rule).outcome is not None
    if not (is_row(x) or is_doc(x)):
        return getattr(x, "outcome", None) is not None
    name = _score_name(rule)
    if is_row(x):
        out = x.get("outcome")
        if not isinstance(out, dict):
            return False
        if name is None:
            return isinstance(out.get("accepted"), bool)
        if isinstance(out.get("scores"), dict):
            value = out["scores"].get(name)
        elif (out.get("name") or "judge") == name:
            value = out.get("score")
        else:
            value = None
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name is None:
        return True  # an acceptance rule on a stored run: compare reads it; count it as offered
    from ..compare import last_score_signal

    return last_score_signal(x, name) is not None


def scale_of(x: Any, rule: Any) -> str | None:
    """The scale a scored run's outcome is on (`linear` unless it says), None under an acceptance rule.

    harness2 scores are fractions from 0 to 1 (`scale: "fraction"`); other tools write points or any
    linear score. The gate works in the runs' own unit and prints it.
    """
    name = _score_name(rule)
    if name is None:
        return None
    if is_row(x):
        out = x.get("outcome") if isinstance(x.get("outcome"), dict) else {}
        return out.get("scale") or "linear"
    if is_doc(x):
        from ..compare import last_score_signal

        sig = last_score_signal(x, name)  # the signal compare reads the score from, not the first one
        return (sig.get("scale") if sig else None) or "linear"
    return getattr(x, "scale", None) or "linear"


def parse_selector(sel: str) -> tuple[str, str]:
    """`digest:sha256:<hex>` or `sha256:<hex>`, `cfg_...`, or `arm:<name>` or `arm=<name>`, as (kind, value);
    ValueError otherwise. A digest may be a prefix; `select` resolves it against the digests present."""
    if sel.startswith("digest:"):
        sel = sel[len("digest:"):]
        if not sel.startswith("sha256:"):
            raise ValueError(f"unknown selector 'digest:{sel}': give {SELECTOR_FORMS}")
    if sel.startswith("sha256:") and len(sel) > len("sha256:"):
        return "digest", sel
    if sel.startswith("cfg_"):
        return "config", sel
    for lead in ("arm:", "arm="):
        if sel.startswith(lead) and sel[len(lead):]:
            return "arm", sel[len(lead):]
    raise ValueError(f"unknown selector {sel!r}: give {SELECTOR_FORMS}")


READERS = {"digest": digest_of, "config": config_of, "arm": arm_of}


def resolve(runs: Iterable[Any], sel: str) -> tuple[str, str]:
    """The selector's (kind, value), a digest prefix made whole when it matches exactly one digest of `runs`.
    A prefix of more than one digest refuses, listing them; one of none is left as given (it selects no run)."""
    kind, value = parse_selector(sel)
    if kind != "digest":
        return kind, value
    digests = sorted({d for d in (digest_of(x) for x in runs) if isinstance(d, str)})
    if value in digests:
        return kind, value
    hits = [d for d in digests if d.startswith(value)]
    if len(hits) > 1:
        raise ValueError(f"selector {sel!r} matches {len(hits)} digests: {', '.join(hits)}; give more of one")
    return kind, (hits[0] if hits else value)


def select(runs: Iterable[Any], sel: str) -> list[Any]:
    runs = list(runs)
    kind, value = resolve(runs, sel)
    return [x for x in runs if READERS[kind](x) == value]


def _shown(kind: str, value: str) -> str:
    if kind == "arm":
        return f"arm:{value}"
    return value[:19] + "..." if kind == "digest" and len(value) > 19 else value


def present(runs: Iterable[Any]) -> str:
    """What the runs offer to select, for a selector that matches none: arms, digests and configurations,
    each with its run count."""
    runs, parts = list(runs), []
    for kind, label in (("arm", "arms"), ("digest", "digests"), ("config", "configurations")):
        seen: dict[str, int] = {}
        for v in (READERS[kind](x) for x in runs):
            if isinstance(v, str) and v:
                seen[v] = seen.get(v, 0) + 1
        if seen:
            names = [f"{_shown(kind, v)} ({n} run{'' if n == 1 else 's'})" for v, n in sorted(seen.items())]
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts) if parts else "no arm, digest or configuration is recorded on these runs"


def read_rows(path: Path | str) -> list[dict]:
    """Every row of a harness-runs file. One bad line refuses the whole file, naming the line."""
    try:
        from loopmath import harness_runs  # the format's owner, when present
    except ImportError:
        harness_runs = None
    if harness_runs is not None:
        return harness_runs.read(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{i}: not JSON ({exc.msg})") from None
            if not isinstance(row, dict) or row.get("schema") != ROWS_SCHEMA:
                got = row.get("schema") if isinstance(row, dict) else type(row).__name__
                raise ValueError(f"{path}:{i}: schema {got!r} is not {ROWS_SCHEMA}")
            if not isinstance(row.get("task_id"), str) or not row["task_id"]:
                raise ValueError(f"{path}:{i}: task_id must be a non-empty string")
            rows.append(row)
    return rows
