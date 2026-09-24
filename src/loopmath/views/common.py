"""Page shell, CSS variables, number formatting, interval bars, the graph component.

Every view (spec 06) is one self-contained HTML file: inline CSS and JS only, no
server, no network. The page embeds its data object as
`<script type="application/json" id="data">` and renders from that object and
nothing else, so `--json` and the page show the same numbers.

The JS and CSS live in `views/assets/` and are concatenated inline here. The
graph component has two inputs:

- a workflow graph (`workflow_graph()`): pieces and artifacts of one
  configuration, annotated with predictions (plans, posterior) or with a run's
  realized attempts, gate results and artifact versions (`run_workflow_graph()`);
- a run graph (`run_graph()`): today's `graph/html_data` object for an OCP
  document, drawn with today's swimlanes, force and cost-curve layouts.

Owner: lane 12. Spec: design/0.1/06-views.md, 03-interfaces.md section 8.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..output import HTML_DEFAULT, home as store_home

ASSET_DIR = "assets"
TOKEN_STREAMS = (("in", "input_tokens"), ("cache_read", "cached_input_tokens"),
                 ("cache_write", "cache_creation_tokens"), ("out", "output_tokens"))
TIER_RANK = {"verified": 0, "reported": 1, "heuristic": 2, "asserted": 3}
_DATA_RE = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)


# ---------------------------------------------------------------- assets and page shell
def asset(name: str) -> str:
    """One file from `views/assets/`, read from package data."""
    return resources.files("loopmath.views").joinpath(ASSET_DIR, name).read_text(encoding="utf-8")


def legacy_layouts_js() -> str:
    """Today's swimlanes, force and cost-curve layouts, unchanged (graph/html_*.py)."""
    from ..graph.html_costcurve import COSTCURVE_JS
    from ..graph.html_force import FORCE_JS
    from ..graph.html_swimlanes import SWIMLANES_JS

    return "\n".join((SWIMLANES_JS, FORCE_JS, COSTCURVE_JS))


_LEGACY_CSS_PREFIXES = ("svg text", ".lanelbl", ".axlbl", ".axis", ".grid ", ".grid{", ".lanebg", ".node",
                        ".nlbl", ".hit", ".edge", ".e-", ".art", ".small", ".alink", ".hide-heur", ".seg", ".env")


def legacy_layouts_css() -> str:
    """The SVG rules of today's viewer stylesheet (graph/html_css.py), one rule per line there."""
    from ..graph.html_css import CSS

    keep = [line for line in CSS.splitlines() if line.strip().startswith(_LEGACY_CSS_PREFIXES)]
    return "\n".join(".lg " + line.strip() for line in keep)


def graph_js() -> str:
    """The shared graph component: `LM.Graph` (workflow graphs) and `LM.RunGraph` (run graphs)."""
    return legacy_layouts_js() + "\n" + asset("graph.js")


def graph_css() -> str:
    return asset("graph.css") + "\n" + legacy_layouts_css()


def embed_json(data: Any) -> str:
    """JSON for a `<script type="application/json">` block: still valid JSON, never closes the block."""
    try:
        text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=_default)
    except ValueError:
        text = json.dumps(_finite(data), ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=_default)
    return (text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def extract_data(page: str) -> dict:
    """The data object embedded in a view page (the inverse of `page()`)."""
    match = _DATA_RE.search(page)
    if not match:
        raise ValueError("no embedded data block")
    return json.loads(match.group(1))


def page(*, title: str, data: Any, body: str, scripts: Iterable[str], styles: Iterable[str] = ()) -> str:
    """One offline HTML document: base CSS plus `styles`, the data block, then `scripts` in order."""
    from html import escape

    css = "\n".join([asset("base.css"), *styles])
    js = "\n;\n".join([asset("common.js"), *scripts])
    for part in (css, js):
        if "</script" in part.lower() or "</style" in part.lower():
            raise ValueError("inline asset closes its own block")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{escape(title)}</title>\n<style>\n{css}\n</style>\n</head>\n<body>\n"
        f"<div id=\"app\" class=\"app\">\n{body}\n</div>\n"
        f"<script type=\"application/json\" id=\"data\">{embed_json(data)}</script>\n"
        f"<script>\n{js}\n</script>\n</body>\n</html>\n"
    )


def graph_page(graph: Mapping[str, Any], *, title: str, data: Mapping[str, Any] | None = None) -> str:
    """A standalone page for one workflow graph (lane 4's `workflows show --html`).

    `data` is the object to embed; the graph is read from its `graph` key. The default is
    `{"schema": "loopmath.view.workflow/1", "graph": graph}`.
    """
    obj = dict(data) if data is not None else {"schema": "loopmath.view.workflow/1", "graph": dict(graph)}
    obj.setdefault("graph", dict(graph))
    body = ('<header class="top"><h1><span class="k">loopmath</span> workflow</h1>'
            '<p class="lede" id="lede"></p></header><section class="panel"><div id="graph"></div></section>')
    boot = ("(() => { const D = LM.data(), g = D.graph || {};"
            " document.getElementById('lede').textContent = (g.label || g.config || 'workflow') +"
            " ': each box is a piece of the workflow, each diamond an artifact it reads or writes.';"
            " LM.Graph.render(document.getElementById('graph'), g, {}); })();")
    return page(title=title, data=obj, body=body, scripts=[graph_js(), boot], styles=[graph_css()])


def html_target(arg: str | None, command: str, home_override: str | None = None) -> Path | None:
    """Resolve `--html [PATH]` (spec 02 section 1). None when `--html` was not given."""
    from datetime import datetime

    if arg is None:
        return None
    if arg != HTML_DEFAULT:
        return Path(arg).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return store_home(home_override) / "views" / f"{command}-{stamp}.html"


def write_page(path: Path, text: str, *, json_mode: bool = False) -> Path:
    """Atomic write (temp file in the same folder, fsync, rename), then print the path.

    With `--json` the path goes to stderr, so stdout holds exactly one JSON object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(tmp, 0o666 & ~mask)  # mkstemp makes 0600; a view is an ordinary file
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(str(path), file=sys.stderr if json_mode else sys.stdout)
    return path


def _default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def plain(value: Any) -> Any:
    """Records (anything with `to_dict`), tuples and paths as plain JSON values; non-finite floats as None."""
    return _finite(json.loads(json.dumps(value, default=_default, allow_nan=True)))


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


# ---------------------------------------------------------------- number formatting (terminal summaries)
def num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def fmt_usd(value: Any) -> str:
    v = num(value)
    if v is None:
        return "n/a"
    if abs(v) >= 1000:
        return f"${v:,.0f}"
    return f"${v:.3f}" if 0 < abs(v) < 0.1 else f"${v:.2f}"


def fmt_tokens(value: Any) -> str:
    v = num(value)
    if v is None:
        return "n/a"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def fmt_money(usd: Any, tokens: Any) -> str:
    """Dollars with tokens beside them: `$1.95 (351k tokens)`."""
    return f"{fmt_usd(usd)} ({fmt_tokens(tokens)} tokens)"


def fmt_pct(value: Any, digits: int = 0) -> str:
    v = num(value)
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


# D107: the same words in every view (assets/common.js TAIL_NOTE); the JSON is unchanged.
TAIL_NOTE = "the average is pulled up by rare very large outcomes"


def pulled_up(iv: Mapping[str, Any] | None) -> bool:
    """A mean above its interval's upper end: after D98 and D106, always a heavy tail (D107)."""
    if not iv:
        return False
    mean, hi = num(iv.get("mean")), num(iv.get("hi"))
    return mean is not None and hi is not None and mean > hi


def fmt_interval(iv: Mapping[str, Any] | None, fmt=fmt_usd) -> str:
    """`mean (lo to hi)`, with the D107 note when the mean is above the upper end."""
    if not iv:
        return "n/a"
    text = f"{fmt(iv.get('mean'))} ({fmt(iv.get('lo'))} to {fmt(iv.get('hi'))})"
    return f"{text}; {TAIL_NOTE}" if pulled_up(iv) else text


# ---------------------------------------------------------------- workflows in either shape
def as_dict(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    return to_dict() if callable(to_dict) else value


def model_name(model: Any) -> str | None:
    """A setting's model: a canonical id string (types) or an OCP modelRef `{raw, id, ...}`."""
    if isinstance(model, Mapping):
        return model.get("id") or model.get("raw")
    return model if isinstance(model, str) else None


def _catalog_workflow(ref: str) -> dict | None:
    try:
        from ..workflows.format import catalog

        wf = catalog().get(ref)
    except Exception:  # lane 4's catalog may not have landed; the ref is shown as given
        return None
    return as_dict(wf) if wf is not None else None


def normalize_workflow(workflow: Any, settings: Mapping[str, Any] | None = None) -> dict:
    """One shape for a workflow given as `types.Workflow`, its dict, OCP 2.3, or an OCP `{ref, version}`.

    Returns `{id, version, title, pieces: [{id, role, width}], artifacts: [{id, kind}],
    edges: [[from, to]], gates: [{id, after, rule, on_fail}], budget_rounds, ref}`.
    """
    wf = as_dict(workflow) or {}
    ref = None
    if "ref" in wf and "pieces" not in wf:
        ref = wf.get("ref")
        wf = _catalog_workflow(str(ref)) or {"id": ref, "version": wf.get("version"), "pieces": [
            {"id": key, "role": ""} for key in (settings or {})]}
    pieces = []
    for p in wf.get("pieces") or []:
        p = as_dict(p)
        if isinstance(p, str):
            p = {"id": p}
        pieces.append({"id": str(p.get("id")), "role": str(p.get("role") or ""), "width": int(p.get("width") or 1)})
    artifacts = []
    for a in wf.get("artifacts") or []:
        a = as_dict(a)
        artifacts.append({"id": str(a), "kind": None} if not isinstance(a, Mapping)
                         else {"id": str(a.get("id")), "kind": a.get("kind")})
    edges = [[str(e[0]), str(e[1])] for e in (wf.get("edges") or []) if isinstance(e, (list, tuple)) and len(e) == 2]
    control = as_dict(wf.get("control")) or {}
    gates = []
    repair = control.get("repair") or {}
    for g in control.get("gates") or []:
        g = as_dict(g)
        if isinstance(g, Mapping):
            gates.append({"id": str(g.get("id")), "after": str(g.get("after")), "rule": g.get("rule") or "",
                          "on_fail": g.get("on_fail")})
        else:  # OCP: a gate is the piece id whose output is a verdict; repair maps it to the piece that reruns
            gates.append({"id": str(g), "after": str(g), "rule": "", "on_fail": repair.get(g)})
    budget = control.get("budget_rounds", control.get("budget"))
    return {"id": wf.get("id"), "version": wf.get("version"), "title": wf.get("title") or "", "pieces": pieces,
            "artifacts": artifacts, "edges": edges, "gates": gates,
            "budget_rounds": int(budget) if isinstance(budget, (int, float)) else None, "ref": ref}


def config_label(config: Mapping[str, Any]) -> str:
    """'implement_review: claude-opus-5-5/high, gpt-6-astra/xhigh', like `Configuration.label()`."""
    wf = normalize_workflow(config.get("workflow"), config.get("settings"))
    settings = config.get("settings") or {}
    parts = []
    for piece in wf["pieces"]:
        s = as_dict(settings.get(piece["id"]))
        if isinstance(s, Mapping):
            parts.append(f"{model_name(s.get('model'))}/{s.get('effort', 'default')}")
    name = wf.get("id") or "workflow"
    return f"{name}: " + ", ".join(parts)


def workflow_graph(config: Any, prediction: Any = None) -> dict:
    """The workflow graph of one configuration, with per-piece predictions when given.

    Shape (spec 03 section 8, as in the view fixtures): `{config, label, nodes, edges, gates}`,
    nodes `{id, kind: "piece", role, setting, prediction}` or `{id, kind: "artifact"}`,
    edges `{from, to}`, gates `{id, after, rule, on_fail}`. Extra keys: `workflow`, `title`,
    `budget_rounds`, and `width` or `artifact_kind` when they carry information.
    """
    cfg = as_dict(config) or {}
    pred = as_dict(prediction) or {}
    settings = cfg.get("settings") or {}
    wf = normalize_workflow(cfg.get("workflow"), settings)
    per_piece = pred.get("per_piece") or {}
    nodes: list[dict] = []
    for piece in wf["pieces"]:
        node: dict[str, Any] = {"id": piece["id"], "kind": "piece", "role": piece["role"],
                                "setting": as_dict(settings.get(piece["id"]))}
        if piece["id"] in per_piece:
            node["prediction"] = as_dict(per_piece[piece["id"]])
        if piece["width"] != 1:
            node["width"] = piece["width"]
        nodes.append(node)
    for art in wf["artifacts"]:
        node = {"id": art["id"], "kind": "artifact"}
        if art.get("kind"):
            node["artifact_kind"] = art["kind"]
        nodes.append(node)
    return {"config": cfg.get("id"), "label": config_label(cfg), "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in wf["edges"]], "gates": wf["gates"],
            "workflow": wf.get("id") or wf.get("ref"), "title": wf["title"], "budget_rounds": wf["budget_rounds"]}


# ---------------------------------------------------------------- OCP run documents
def attempt_cost(attempt: Mapping[str, Any]) -> dict:
    """Dollars and the four token streams of one attempt; None where not recorded."""
    cost = attempt.get("cost") if isinstance(attempt.get("cost"), Mapping) else {}
    streams = {key: num(cost.get(field)) for key, field in TOKEN_STREAMS}
    known = [v for v in streams.values() if v is not None]
    return {"usd": num(cost.get("usd")), "tokens": sum(known) if known else None, "streams": streams}


def _vertex_of(attempt: Mapping[str, Any], node_vertex: Mapping[str, str], pieces: list[str]) -> str | None:
    vertex = attempt.get("vertex") or node_vertex.get(str(attempt.get("node")))
    if vertex:
        return str(vertex)
    return pieces[0] if len(pieces) == 1 else None


def prediction_from_receipt(receipt: Any) -> dict | None:
    """The predicted side of a receipt in either shape: spec 03 `Receipt.before` or OCP `run.receipt.before`."""
    receipt = as_dict(receipt)
    if not isinstance(receipt, Mapping):
        return None
    before = receipt.get("before") or receipt.get("predicted")
    if isinstance(before, Mapping) and isinstance(before.get("predicted"), Mapping):
        before = before["predicted"]
    return dict(before) if isinstance(before, Mapping) else None


def parse_ts(value: Any) -> datetime | None:
    """An ISO 8601 timestamp as an aware datetime (local time when it has no offset); None if absent or bad."""
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.astimezone()


def ts_key(value: Any) -> tuple:
    """Sort key by instant, not by text (producers write different offsets); unparsed values sort first."""
    ts = parse_ts(value)
    return (1, ts.timestamp(), "") if ts is not None else (0, 0.0, str(value or ""))


def run_signals(doc: Mapping[str, Any], appended: Iterable[Mapping[str, Any]] = ()) -> list[dict]:
    """`run.signals` plus signals appended before `finish`, deduplicated by id, in time order."""
    seen: dict[str, dict] = {}
    for sig in list((doc.get("run") or {}).get("signals") or []) + list(appended):
        if isinstance(sig, Mapping):
            seen.setdefault(str(sig.get("id") or f"_{len(seen)}"), dict(sig))
    return sorted(seen.values(), key=lambda s: ts_key(s.get("observed_at")))


def run_workflow_graph(doc: Mapping[str, Any], prediction: Mapping[str, Any] | None = None,
                       signals: Iterable[Mapping[str, Any]] | None = None) -> dict | None:
    """The run's workflow graph with its realized attempts per piece and round.

    Each piece node gains `realized: {attempts, usd, tokens, streams, rounds}`; each gate gains
    `results: [{round, value, attempt}]`; each artifact node gains `versions`. Attempts that no
    piece claims are listed under `unplaced`. None when the run has no configuration.
    """
    run = doc.get("run") or {}
    cfg = run.get("configuration")
    if not isinstance(cfg, Mapping):
        return None
    graph = workflow_graph(cfg, prediction or prediction_from_receipt(run.get("receipt")))
    graph["run"] = run.get("id")
    pieces = [n["id"] for n in graph["nodes"] if n["kind"] == "piece"]
    node_vertex = {str(n.get("id")): n.get("vertex") for n in doc.get("nodes") or [] if n.get("vertex")}
    by_piece: dict[str, list[dict]] = {p: [] for p in pieces}
    unplaced: list[dict] = []
    for att in doc.get("attempts") or []:
        cost = attempt_cost(att)
        outcome = att.get("outcome") if isinstance(att.get("outcome"), Mapping) else {}
        item = {"id": att.get("id"), "round": att.get("round") or 1, "status": att.get("status"),
                "model": model_name(att.get("model")), "effort": att.get("effort"), "harness": att.get("harness"),
                "started_at": att.get("started_at"), "ended_at": att.get("ended_at"),
                "usd": cost["usd"], "tokens": cost["tokens"], "streams": cost["streams"],
                "result": outcome.get("result")}
        vertex = _vertex_of(att, node_vertex, pieces)
        (by_piece[vertex] if vertex in by_piece else unplaced).append(item)
    for node in graph["nodes"]:
        if node["kind"] != "piece":
            continue
        atts = sorted(by_piece[node["id"]], key=lambda a: (a["round"], ts_key(a["started_at"])))
        usd = [a["usd"] for a in atts if a["usd"] is not None]
        tok = [a["tokens"] for a in atts if a["tokens"] is not None]
        streams = {key: sum(a["streams"][key] or 0 for a in atts) for key, _ in TOKEN_STREAMS}
        node["realized"] = {"attempts": atts, "usd": sum(usd) if usd else None, "tokens": sum(tok) if tok else None,
                            "streams": streams, "rounds": max((a["round"] for a in atts), default=0)}
    attempt_piece = {a["id"]: p for p, atts in by_piece.items() for a in atts}
    attempt_round = {a["id"]: a["round"] for atts in by_piece.values() for a in atts}
    verdicts = [s for s in (run_signals(doc) if signals is None else signals) if s.get("kind") == "verdict"]
    for gate in graph["gates"]:
        results = [{"round": attempt_round.get(s.get("at_attempt")), "value": s.get("value"),
                    "attempt": s.get("at_attempt"), "name": s.get("name")}
                   for s in verdicts if attempt_piece.get(s.get("at_attempt")) == gate["after"]]
        if not results:  # no verdict signal: read the gate attempts' own outcomes
            results = [{"round": a["round"], "value": a["result"], "attempt": a["id"]}
                       for a in by_piece.get(gate["after"], []) if a["result"] in ("accept", "reject", "pass", "fail")]
        gate["results"] = sorted(results, key=lambda r: r["round"] or 0)
    versions: dict[str, list[dict]] = {}
    for art in doc.get("artifacts") or []:
        if art.get("vertex"):
            versions.setdefault(str(art["vertex"]), []).append(
                {"id": art.get("id"), "version": art.get("version") or 1, "supersedes": art.get("supersedes"),
                 "producer": art.get("producer"), "path": art.get("path")})
    for node in graph["nodes"]:
        if node["kind"] == "artifact" and node["id"] in versions:
            node["versions"] = sorted(versions[node["id"]], key=lambda v: v["version"])
    graph["unplaced"] = unplaced
    return graph


def run_graph(doc: Mapping[str, Any]) -> tuple[dict | None, str | None]:
    """Today's `graph/html_data` object for one OCP document (v0.2 or later), or (None, reason)."""
    from ..graph.html_data import build_html_data
    from ..ingest.ocp_convert import from_ocp

    try:
        data = build_html_data(from_ocp(copy.deepcopy(dict(doc))))
    except Exception as exc:  # the attempt timeline is optional; the view shows the reason instead
        return None, f"{type(exc).__name__}: {exc}"[:300]
    return data, None
