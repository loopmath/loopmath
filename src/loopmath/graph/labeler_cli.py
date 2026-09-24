"""Command-line entry point for the labeler compatibility facade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .labeler_common import BATCH_SIZE, BOUNDARY_TOLERANCE_S, PROMPT_VERSION, PROMPT_VERSIONS, LabelerError, batches, load_dataset
from .labeler_parse import load_predictions, parse_response, write_predictions
from .labeler_pricing import claude_json_result, find_new_session, price_sessions, session_id_from_events
from .labeler_prompt import (
    _build_prompt_snapshot,
    _candidate_summary,
    _prompt_metadata,
    candidate_parents,
)
from .labeler_report import report_lines
from .labeler_score import score
from .labeler_pricing import cost_figures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m loopmath.graph.labeler", description="labeler prompts, answer parsing, scoring and pricing (no model call happens here)")
    sub = p.add_subparsers(dest="verb", required=True)
    b = sub.add_parser("batches", help="print the batch count and the ids per batch")
    b.add_argument("--dataset", required=True)
    b.add_argument("--size", type=int, default=BATCH_SIZE)
    b.add_argument("--count", action="store_true", help="print only the batch count")
    pr = sub.add_parser("prompt", help="write the prompt of one batch")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--batch", type=int, required=True)
    pr.add_argument("--size", type=int, default=BATCH_SIZE)
    pr.add_argument("--out", required=True)
    pr.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    pa = sub.add_parser("parse", help="parse one batch's answer into prediction items")
    pa.add_argument("--dataset", required=True)
    pa.add_argument("--batch", type=int, required=True)
    pa.add_argument("--size", type=int, default=BATCH_SIZE)
    pa.add_argument("--response", required=True, help="the answer text (or the claude -p JSON with --claude-json)")
    pa.add_argument("--claude-json", action="store_true")
    pa.add_argument("--model", required=True)
    pa.add_argument("--effort", required=True)
    pa.add_argument("--out", required=True, help="predictions jsonl, appended")
    pa.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    se = sub.add_parser("session", help="print the session id a call left behind")
    se.add_argument("--events", help="codex exec --json event stream")
    se.add_argument("--claude-json", help="claude -p --output-format json output")
    fs = sub.add_parser("find-session", help="find the log file a call left under the harness root")
    fs.add_argument("--newer-than", required=True)
    fs.add_argument("--cwd", required=True)
    fs.add_argument("--harness", choices=("codex", "claude-code"), required=True)
    sc = sub.add_parser("score", help="score predictions against the dataset and price the labeler's sessions")
    sc.add_argument("--dataset", required=True)
    sc.add_argument("--predictions", required=True)
    sc.add_argument("--session", action="append", default=[], help="a labeler session (run id, uuid or log path); repeatable")
    sc.add_argument("--tolerance", type=float, default=BOUNDARY_TOLERANCE_S)
    sc.add_argument("--out", help="write the scores as JSON here")
    args = p.parse_args(argv)
    try:
        return _run(args)
    except LabelerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _run(args) -> int:
    if args.verb == "batches":
        nodes, edges, c = load_dataset(args.dataset)
        bs = batches(nodes, args.size)
        if args.count:
            print(len(bs))
            return 0
        print(f"dataset {args.dataset}: node items {c['items_node']}, edge items {c['items_edge']}, batch size {args.size}, batches {len(bs)}")
        candidates = candidate_parents(nodes, edges)
        metadata = _prompt_metadata(nodes, edges, PROMPT_VERSION, 0)
        print("  " + _candidate_summary(metadata))
        for i, batch in enumerate(bs):
            print(f"  batch {i}: {len(batch)} nodes, {len(candidates)} parent candidates: {', '.join(it['id'] for it in batch)}")
        return 0
    if args.verb in ("prompt", "parse"):
        nodes, edges, _c = load_dataset(args.dataset)
        bs = batches(nodes, args.size)
        if not 0 <= args.batch < len(bs):
            raise LabelerError(f"batch {args.batch} does not exist; the dataset has {len(bs)} batches of up to {args.size}")
        batch = bs[args.batch]
        if args.verb == "prompt":
            snapshot = _build_prompt_snapshot(
                batch, version=args.prompt_version, all_nodes=nodes, edges=edges
            )
            metadata = snapshot.metadata
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(snapshot.text, encoding="utf-8")
            candidate_count = metadata["parent_candidates"]["included"]
            print(f"wrote prompt {args.prompt_version} for batch {args.batch} ({len(batch)} nodes, {candidate_count} parent candidates) to {out}; {_candidate_summary(metadata)}")
            return 0
        metadata = _prompt_metadata(nodes, edges, args.prompt_version, len(batch), batch=batch)
        rp = Path(args.response)
        if not rp.is_file():
            raise LabelerError(f"response {rp} does not exist")
        if args.claude_json:
            sid, text = claude_json_result(rp)
            if text is None:
                raise LabelerError(f"{rp}: the claude JSON carries no 'result' text")
        else:
            text = rp.read_text(encoding="utf-8")
        preds, c, warnings = parse_response(
            text,
            batch,
            model=args.model,
            effort=args.effort,
            batch_index=args.batch,
            version=args.prompt_version,
            prompt_metadata=metadata,
        )
        write_predictions(preds, Path(args.out), append=True)
        print(f"batch {args.batch}: asked {c['nodes_asked']}, answered {c['nodes_answered']}, unanswered {c['nodes_unanswered']}; " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k not in ("nodes_asked", "nodes_answered", "nodes_unanswered")))
        print("  " + _candidate_summary(metadata))
        for w in warnings:
            print(f"  {w}")
        print(f"appended {len(preds)} predictions to {args.out}")
        if c["nodes_answered"] == 0:
            raise LabelerError(f"batch {args.batch}: the answer named none of the {c['nodes_asked']} nodes asked about")
        return 0
    if args.verb == "session":
        if args.events:
            sid = session_id_from_events(Path(args.events))
            if sid is None:
                raise LabelerError(f"{args.events}: no thread_id or session_id uuid on any event line")
        elif args.claude_json:
            sid, _text = claude_json_result(Path(args.claude_json))
            if sid is None:
                raise LabelerError(f"{args.claude_json}: the claude JSON carries no session_id")
        else:
            raise LabelerError("session: give --events or --claude-json")
        print(sid)
        return 0
    if args.verb == "find-session":
        hits = find_new_session(Path(args.newer_than), args.cwd, args.harness)
        if len(hits) != 1:
            raise LabelerError(f"{len(hits)} {args.harness} log files newer than {args.newer_than} have cwd {args.cwd}" + (": " + ", ".join(str(h) for h in hits) if hits else ""))
        print(hits[0])
        return 0
    if args.verb == "score":
        nodes, edges, _c = load_dataset(args.dataset)
        preds, pc, warnings = load_predictions(args.predictions)
        for w in warnings:
            print(f"  {w}")
        items = nodes + edges
        scores = score(items, preds, tolerance_s=args.tolerance)
        sessions = price_sessions(args.session)
        cost = cost_figures(scores, items, sessions)
        cost["sessions"] = sessions
        print("\n".join(report_lines(scores, cost)))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"dataset": args.dataset, "predictions": args.predictions, "scores": scores, "cost": cost}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote scores to {out}")
        return 0
    raise LabelerError(f"unknown verb {args.verb}")
