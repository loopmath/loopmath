"""Session discovery and loopmath-native pricing for labeler runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .labeler_common import COST_BOUND, LabelerError

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def find_session_files(ref: str) -> list[Path]:
    """The log files a session reference names: an existing path, else every file under
    the two log roots whose name carries the reference's uuid (a run id `cc_<uuid>` or
    `cx_<uuid>`, or a bare uuid)."""
    from .. import ingest

    p = Path(ref).expanduser()
    if p.is_file():
        return [p]
    m = _UUID_RE.search(ref)
    if not m:
        return []
    uuid = m.group(0).lower()
    hits: list[Path] = []
    for root in (ingest.CLAUDE_CODE_ROOT, ingest.CODEX_ROOT):
        if root.is_dir():
            hits.extend(q for q in sorted(root.rglob("*.jsonl")) if uuid in q.name.lower())
    return hits


def price_sessions(refs: list[str]) -> list[dict]:
    """loopmath's own price of each labeler session: the file found for the reference,
    parsed by `ingest`, graded and priced the way `loopmath analyze` does. Each result
    carries `usd`, `wall_s`, `tokens`, `model` and `run_id`, or `reason` when the
    session was not found, not parsed or not priced; nothing is estimated."""
    from .. import grade as grade_mod, ingest, price as price_mod

    table = price_mod.load_prices(None)
    out: list[dict] = []
    for ref in refs:
        files = find_session_files(ref)
        if not files:
            out.append({"ref": ref, "path": None, "usd": None, "reason": "no log file found for the reference under the log roots"})
            continue
        if len(files) > 1:
            out.append({"ref": ref, "path": [str(f) for f in files], "usd": None, "reason": f"{len(files)} log files carry the reference; ambiguous"})
            continue
        path = files[0]
        records, diag = ingest.parse_all(path, use_cache=False)
        if not records:
            out.append({"ref": ref, "path": str(path), "usd": None, "reason": "the file parsed to no run record (" + ", ".join(f"{k} {v}" for k, v in sorted(diag.get("skip_reasons", {}).items())) + ")"})
            continue
        records, _cov = grade_mod.grade_all(records)
        records, _warn = price_mod.price_all(records, table)
        rec, match_reason = _matching_session_record(ref, path, records)
        if rec is None:
            out.append({"ref": ref, "path": str(path), "usd": None, "records_in_file": len(records), "reason": match_reason})
            continue
        entry = {"ref": ref, "path": str(path), "run_id": rec.get("run_id"), "model": rec.get("model"), "effort": rec.get("effort"), "wall_s": rec.get("wall_s"), "tokens": rec.get("tokens"), "usd": rec.get("usd")}
        if len(records) > 1:
            entry["records_in_file"] = len(records)
        if rec.get("usd") is None:
            entry["reason"] = "loopmath did not price the session: " + str(price_mod.price_run(rec, table).get("reason") or "unpriced model or token block")
        out.append(entry)
    return out


def _matching_session_record(ref: str, path: Path, records: list[dict]) -> tuple[dict | None, str | None]:
    """Select one parsed run using identifiers recorded by the invocation or log.

    Exact recorded identifiers take precedence over UUID and rollout-path matches. A
    level that matches more than one record is ambiguous and is never weakened by
    selecting the first record.
    """
    identifier_keys = ("run_id", "session_id", "thread_id", "id")
    exact = [rec for rec in records if any(str(rec.get(key)) == ref for key in identifier_keys if rec.get(key) is not None)]
    levels: list[tuple[str, list[dict]]] = [("recorded identifier", exact)]
    match = _UUID_RE.search(ref)
    if match:
        uuid = match.group(0).lower()
        by_uuid = [
            rec
            for rec in records
            if any(uuid in str(rec.get(key)).lower() for key in identifier_keys if rec.get(key) is not None)
        ]
        levels.append(("session id", by_uuid))
    ref_path = Path(ref).expanduser()
    if ref_path.is_file():
        wanted = ref_path.resolve()
        by_path = []
        for rec in records:
            session_path = rec.get("session_path")
            if isinstance(session_path, str) and Path(session_path).expanduser().resolve() == wanted:
                by_path.append(rec)
        levels.append(("rollout path", by_path))
    for how, matches in levels:
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"{len(records)} records parsed from the file and {len(matches)} match the invocation by {how}; ambiguous"
    return None, f"{len(records)} records parsed from the file and none matches the invocation by recorded identifier, session id or rollout path"


def cost_figures(scores: dict, items: list[dict], sessions: list[dict], bound: float = COST_BOUND) -> dict:
    """The labeler's cost against the cost of the sessions the dataset labels, the bound
    compared, and the per-dollar and per-second figures. Every unpriced session on
    either side is counted; a labeled total with an unpriced part is still given, marked
    as a lower bound, but the ratio and the bound verdict are `None` whenever any session
    on either side is unpriced (a partial denominator would make the verdict false)."""
    nodes = [it for it in items if it.get("item") == "node"]
    labeled_usd = 0.0
    unpriced_nodes = 0
    for it in nodes:
        usd = it["features"]["skeleton"].get("usd")
        if isinstance(usd, (int, float)):
            labeled_usd += float(usd)
        else:
            unpriced_nodes += 1
    priced = [s for s in sessions if isinstance(s.get("usd"), (int, float))]
    unpriced = [s for s in sessions if not isinstance(s.get("usd"), (int, float))]
    labeler_usd = sum(float(s["usd"]) for s in priced) if sessions and not unpriced else None
    labeler_wall = [float(s["wall_s"]) for s in sessions if isinstance(s.get("wall_s"), (int, float))]
    wall_s = sum(labeler_wall) if sessions and len(labeler_wall) == len(sessions) else None
    out: dict = {
        "labeled_sessions": len(nodes),
        "labeled_sessions_unpriced": unpriced_nodes,
        "labeled_usd": labeled_usd,
        "labeled_usd_is_lower_bound": unpriced_nodes > 0,
        "labeler_sessions": len(sessions),
        "labeler_sessions_unpriced": len(unpriced),
        "labeler_usd": labeler_usd,
        "labeler_wall_s": wall_s,
        "bound": bound,
    }
    if not sessions:
        out["ratio"] = None
        out["ratio_reason"] = "no labeler session was given, so its cost is unknown"
    elif labeler_usd is None:
        out["ratio"] = None
        out["ratio_reason"] = f"{len(unpriced)} of {len(sessions)} labeler sessions could not be priced: " + "; ".join(f"{s['ref']}: {s.get('reason')}" for s in unpriced)
    elif unpriced_nodes:
        out["ratio"] = None
        out["ratio_reason"] = f"{unpriced_nodes} of {len(nodes)} labeled sessions are unpriced, so their total cost is only a lower bound and the ratio is not measured"
    elif labeled_usd <= 0:
        out["ratio"] = None
        out["ratio_reason"] = "the labeled sessions have no priced cost to compare against"
    else:
        out["ratio"] = labeler_usd / labeled_usd
        out["within_bound"] = out["ratio"] < bound
    correct = scores["role"]["correct"]
    out["role_correct"] = correct
    out["role_correct_per_usd"] = (correct / labeler_usd) if labeler_usd else None
    out["role_correct_per_second"] = (correct / wall_s) if wall_s else None
    if labeler_usd is None:
        out["per_usd_reason"] = "the labeler's cost is not fully known (see ratio_reason)"
    elif labeler_usd == 0:
        out["per_usd_reason"] = "the labeler's priced cost is zero, so a per-dollar figure is undefined"
    if wall_s is None:
        out["per_second_reason"] = "the wall clock of every labeler session is not known"
    return out


def session_id_from_events(path: Path) -> str | None:
    """The thread or session id in a `codex exec --json` event stream: the first uuid
    under a `thread_id` or `session_id` key on any event line."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            for key in ("thread_id", "session_id"):
                v = obj.get(key)
                if isinstance(v, str) and _UUID_RE.fullmatch(v):
                    return v
            for v in obj.values():
                if isinstance(v, dict):
                    for key in ("thread_id", "session_id", "id"):
                        w = v.get(key)
                        if isinstance(w, str) and _UUID_RE.fullmatch(w):
                            return w
    return None


def claude_json_result(path: Path) -> tuple[str | None, str | None]:
    """`(session_id, result text)` of a `claude -p --output-format json` answer."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise LabelerError(f"{path}: not the JSON object claude -p --output-format json writes ({e})") from e
    if isinstance(obj, list):
        obj = next((o for o in obj if isinstance(o, dict) and o.get("type") == "result"), obj[-1] if obj else {})
    if not isinstance(obj, dict):
        raise LabelerError(f"{path}: not a JSON object")
    sid = obj.get("session_id")
    result = obj.get("result")
    return (sid if isinstance(sid, str) else None), (result if isinstance(result, str) else None)


def find_new_session(newer_than: Path, cwd: str, harness: str) -> list[Path]:
    """Log files under the harness's root modified after `newer_than` whose head names
    `cwd` as the session's cwd: the fallback when a call's output gave no id."""
    from .. import ingest
    from .dataset import transcript_cwd

    root = ingest.CODEX_ROOT if harness == "codex" else ingest.CLAUDE_CODE_ROOT
    since = newer_than.stat().st_mtime
    hits = []
    if root.is_dir():
        for q in sorted(root.rglob("*.jsonl")):
            try:
                if q.stat().st_mtime < since:
                    continue
            except OSError:
                continue
            if transcript_cwd(q, harness) == cwd:
                hits.append(q)
    return hits
