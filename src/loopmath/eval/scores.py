"""`eval scores import | check | add | export`: the study folder and its score tables.

`args.eval_command` is `scores import`, `scores check`, `scores add` or `scores export`.

- `import` builds or extends a study folder (`runs.jsonl`, `scores.jsonl`, `pairs.jsonl`, `items.jsonl`)
  from OCP run documents, harness-runs rows, CSV files of readings and pairs, and an items file. A row
  whose id is already in the study with the same content changes nothing; so does a run the source says
  less about than the study does. An id seen before with other content is refused by id. Any refused row
  refuses the whole import: the report lists every refusal and nothing is written.
- `check` validates a study folder against the rubric and the judges file, lists every problem by name,
  reports arms with mixed settings and prints the design table (what the design can identify).
- `add` appends one reading or one pair.
- `export` prints, per run, the OCP signals and preferences of the study as JSON Lines. It writes no file.

Repeats are the default: two readings of one target, component and judge with different calls are both
read. A correction is explicit, through `replaces`.

This module reads OCP and harness-runs data only through `adapters/ocp.py`, which it loads when a
command needs it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..output import EXIT_OK, EXIT_USER, emit_json
from . import contracts as C

TABLES = {"runs": (C.RUNS_FILE, C.SCHEMA_RUNS), "scores": (C.SCORES_FILE, C.SCHEMA_SCORES),
          "pairs": (C.PAIRS_FILE, C.SCHEMA_PAIRS), "items": (C.ITEMS_FILE, C.SCHEMA_ITEMS)}
NOUN = {"runs": "run", "scores": "reading", "pairs": "pair", "items": "item"}
COUNTED = {"runs": "runs", "scores": "readings", "pairs": "pairs", "items": "items"}
SCORE_COLUMNS = ("run", "step", "component", "judge", "call", "value", "item", "of", "observed_at")
PAIR_COLUMNS = ("a", "b", "step", "component", "judge", "call", "shown_first", "verdict", "observed_at")
MORE_COLUMNS = ("id", "replaces", "assignment", "rubric", "cost_usd")
NEEDS = {
    "run_vs_input": "2 or more runs per input",
    "sigma": "2 or more calls by the judge on the same target and component",
    "tau": "repeats by the judge, and 2 or more judges on the same targets",
    "judge_input": "2 or more runs per input scored by the judge, on 10 or more inputs",
    "binary_status": "repeats by the anchor, or 2 or more judges on the same targets",
    "family": "judges from 2 or more families, each scoring outputs from 2 or more families",
    "reference_family": "a reference whose author's family differs from the judge's own",
    "halo": "panel and per_component calls by the same model, effort and prompt, or repeats of panel calls",
    "position": "pairs judged in both orders, or shown_first balanced across pairs",
    "sigma_pair": "repeats of pairs by the judge",
    "tie_width": "repeats of pairs by the judge, and ties",
}
_INT = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def run(args: argparse.Namespace) -> int:
    handlers = {"scores import": import_, "scores check": check, "scores add": add, "scores export": export}
    handler = handlers.get(getattr(args, "eval_command", ""))
    if handler is None:
        from .commands import not_built

        return not_built(args)
    return handler(args)


def _usage(command: str, where: str, problem: str, fix: str, code: str = "usage") -> C.ContractError:
    return C.ContractError(f"eval {command}", where, problem, fix, code)


def _load(folder: Path) -> dict[str, list[dict]]:
    return {name: C.read_lines(folder / file, schema, missing_ok=True) for name, (file, schema) in TABLES.items()}


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _digest(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def parse_value(text: str, comp: dict | None = None) -> Any:
    """A score typed on a command line or in a CSV cell, as the value a reading holds.

    On a `levels` component the text that names a level gives that level. Otherwise a whole number, then a
    number, then the text itself, which the scale check refuses where a number is needed."""
    if comp is not None and comp["scale"]["kind"] == "levels":
        for level in comp["scale"]["levels"]:
            if str(level) == text:
                return level
    try:
        if _INT.fullmatch(text):
            return int(text)
        if _FLOAT.fullmatch(text) and math.isfinite(float(text)):
            return float(text)
    except ValueError:
        pass
    return text


def _finish(args: argparse.Namespace, schema: str, report: dict, lines: list[str], errors: list[str] = ()) -> int:
    if args.json:
        emit_json(schema, report)
    else:
        if lines:
            print("\n".join(lines))
        for line in errors:
            print(f"error: {line}", file=sys.stderr)
    return EXIT_OK if report.get("ok", True) else EXIT_USER


def read_rows(rows: list[dict], file: str) -> list[dict]:
    """The rows that are read once corrections are applied, without refusing: a `replaces` that names
    nothing is ignored, and when the corrections form a loop every row is returned. `eval scores check`
    names both problems; here they must not hide a second one."""
    ids = {row["id"] for row in rows}
    rows = [{**row, "replaces": None} if row.get("replaces") is not None and row["replaces"] not in ids else row
            for row in rows]
    try:
        return C.live(rows, file)[0]
    except C.ContractError:
        return list(rows)


def call_problems(table: str, rows: list[dict], rubric: dict | None = None) -> list[dict]:
    """The calls that hold more than a call can, among `rows` of `scores` or `pairs` (the rows that are read).

    A call is one judge looking once. It gives one reading per component (or one line per item, or one
    count), and one verdict per component compared. A second row is a repeat and needs its own call, or
    is a correction and names the row it corrects. Each problem is `{"call", "component", "ids", "problem",
    "fix", "code"}`, with `ids` the rows that clash, in file order. A pair with no component is on the
    overall component of its step when `rubric` is given."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    out: list[dict] = []

    def problem(call: str, comp: Any, clash: list[dict], text: str, fix: str, code: str) -> None:
        out.append({"call": call, "component": comp, "ids": [row["id"] for row in clash], "problem": text,
                    "fix": fix, "code": code})

    if table == "pairs":
        for row in rows:
            comp = row["component"]
            if comp is None and rubric is not None:
                comp = C.overall_component(rubric, row["step"])
            groups[(row["call"], comp)].append(row)
        for (call, comp), group in groups.items():
            if len(group) > 1:
                on = "" if comp is None else f' for the component "{comp}"'
                problem(call, comp, group, f'two pairs of one call{on} ("{group[0]["id"]}" and "{group[1]["id"]}")',
                        "give a repeat its own call, or name the pair it corrects in replaces", "call_twice")
        return out
    for row in rows:
        groups[(row["call"], row["component"])].append(row)
    for (call, comp), group in groups.items():
        item_rows = [row for row in group if "item" in row]
        others = [row for row in group if "item" not in row]
        twice = [item for item, n in Counter(row["item"] for row in item_rows).items() if n > 1]
        if item_rows and others:
            problem(call, comp, group, f'the call has item lines and a count line for the component "{comp}" '
                    f'("{item_rows[0]["id"]}" and "{others[0]["id"]}")',
                    "write item lines or one count line, never both", "items_and_count")
        elif len(others) > 1:
            problem(call, comp, others, f'two readings of one call for the component "{comp}" '
                    f'("{others[0]["id"]}" and "{others[1]["id"]}")',
                    "give a repeat its own call, or name the reading it corrects in replaces", "call_twice")
        elif twice:
            problem(call, comp, [row for row in item_rows if row["item"] == twice[0]],
                    f'two readings of one call for the item "{twice[0]}"',
                    "give a repeat its own call, or name the reading it corrects in replaces", "call_twice")
    return out


# ---------------------------------------------------------------------------------------------------
# eval scores import

def _csv(path: str) -> tuple[str, list[str], list[tuple[int, dict]], str]:
    """`(file name, header, [(line, {column: cell})], the file's modification time)` of a CSV file."""
    file = Path(path)
    name = file.name
    try:
        text = file.read_text(encoding="utf-8-sig")
        modified = _dt.datetime.fromtimestamp(file.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except FileNotFoundError:
        raise C.ContractError(name, "", "no such file", f"check the path {file}", "no_file") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise C.ContractError(name, "", f"cannot be read ({type(exc).__name__})", "give a UTF-8 CSV file",
                              "unreadable") from None
    reader = csv.reader(io.StringIO(text, newline=""))
    header: list[str] | None = None
    records = []
    try:
        for cells in reader:
            if header is None:
                header = [cell.strip() for cell in cells]
                continue
            if any(cell.strip() for cell in cells):
                records.append((reader.line_num, {column: cells[n].strip() for n, column in enumerate(header)
                                                  if n < len(cells)}))
    except csv.Error as exc:
        raise C.ContractError(name, f"line {reader.line_num}", f"not valid CSV ({exc})", "fix the line",
                              "bad_csv") from None
    if not header or not any(header):
        raise C.ContractError(name, "", "the file has no header line", "name the columns on the first line",
                              "no_header")
    twice = [column for column, n in Counter(header).items() if n > 1 and column]
    if twice:
        raise C.ContractError(name, "line 1", f'the column "{twice[0]}" appears twice', "give every column once",
                              "duplicate")
    return name, header, records, modified


def _need_columns(name: str, header: list[str], any_of: tuple[str, ...], columns: tuple[str, ...],
                  rubric: dict | None) -> None:
    for group in any_of:
        if not any(column in header for column in group.split("|")):
            shown = " or ".join(f'"{column}"' for column in group.split("|"))
            raise C.ContractError(name, "line 1", f"no column {shown}", f"give the columns {', '.join(columns)}",
                                  "no_column")
    if rubric is None and "rubric" not in header:
        raise C.ContractError(name, "", "the rows name no rubric", "give --rubric, or a rubric column with id@version",
                              "no_rubric")


def _other(side: str) -> str:
    return "b" if side == "a" else "a"


def _join_score(raw: dict, plan: dict, used: set) -> str | None:
    """Fill a form row from the assignment that asked for it. Returns why it cannot be joined, or None."""
    if "assignment" in raw:
        found = [a for a in plan["assignments"] if a["id"] == raw["assignment"] and a["ask"] == "score"]
        if not found:
            return f'no score assignment "{raw["assignment"]}" in the plan'
    else:
        found = [a for a in plan["assignments"] if a["ask"] == "score" and a["target"]["run"] == raw.get("run")
                 and a["target"]["step"] == raw.get("step", C.FINAL) and raw.get("component") in a["components"]
                 and raw.get("judge", a["reviewer"]) == a["reviewer"]]
        if not found:
            return "no assignment of the plan asks for this reading"
    free = [a for a in found if (a["id"], raw.get("component"), raw.get("item")) not in used]
    a = (free or found)[0]
    for column, want in (("run", a["target"]["run"]), ("step", a["target"]["step"]), ("judge", a["reviewer"])):
        if raw.setdefault(column, want) != want:
            return f'{column} "{raw[column]}" does not match the assignment "{a["id"]}", which has "{want}"'
    if raw.get("component") not in a["components"]:
        return f'the assignment "{a["id"]}" does not ask for the component "{raw.get("component", "")}"'
    raw["assignment"] = a["id"]
    used.add((a["id"], raw.get("component"), raw.get("item")))
    return None


def _join_pair(raw: dict, plan: dict, used: set) -> str | None:
    """Fill a pair row of a form from the assignment that asked for it. Returns why it cannot be, or None."""
    if "assignment" in raw:
        found = [a for a in plan["assignments"] if a["id"] == raw["assignment"] and a["ask"] == "pair"]
        if not found:
            return f'no pair assignment "{raw["assignment"]}" in the plan'
    else:
        found = [a for a in plan["assignments"] if a["ask"] == "pair"
                 and {a["a"]["run"], a["b"]["run"]} == {raw.get("a"), raw.get("b")}
                 and a["a"]["step"] == raw.get("step", C.FINAL) and raw.get("judge", a["reviewer"]) == a["reviewer"]
                 and raw.get("component", a["components"][0]) in a["components"]]
        if not found:
            return "no assignment of the plan asks for this pair"
    free = [a for a in found if a["id"] not in used]
    a = (free or found)[0]
    flipped = raw.get("a") == a["b"]["run"] and raw.get("b") == a["a"]["run"]
    sides = ("b", "a") if flipped else ("a", "b")
    for column, side in zip(("a", "b"), sides):
        if raw.setdefault(column, a[side]["run"]) != a[side]["run"]:
            return f'{column} "{raw[column]}" does not match the assignment "{a["id"]}"'
    for column, want in (("step", a["a"]["step"]), ("judge", a["reviewer"])):
        if raw.setdefault(column, want) != want:
            return f'{column} "{raw[column]}" does not match the assignment "{a["id"]}", which has "{want}"'
    raw.setdefault("shown_first", _other(a["shown_first"]) if flipped else a["shown_first"])
    if "component" not in raw and len(a["components"]) == 1:
        raw["component"] = a["components"][0]
    elif "component" in raw and raw["component"] not in a["components"]:
        return f'the assignment "{a["id"]}" does not ask for the component "{raw["component"]}"'
    raw["assignment"] = a["id"]
    used.add(a["id"])
    return None


def _table_csv(path: str, table: str, rubric: dict | None, plan: dict | None) -> dict:
    """Read a CSV of readings (`table` is `scores`) or of pairs.

    Returns `{"rows", "refused", "not_used", "ignored", "file_time"}`. `rows` holds `(raw row, source)` and
    `file_time` the ids of the rows whose time is the file's. A row with an empty value or verdict is a blank
    line of a form: it is listed in `not_used`. Without an `id` column a row's id is made from its content, so the same file gives
    the same ids. Without a `call`, a reading is its own call; item lines of one target, component and judge
    share one. Without `observed_at`, the file's modification time is used."""
    name, header, records, modified = _csv(path)
    pairs = table == "pairs"
    columns = PAIR_COLUMNS if pairs else SCORE_COLUMNS
    _need_columns(name, header, ("verdict|choice",) if pairs else ("component", "value"), columns, rubric)
    known = columns + MORE_COLUMNS + (("choice",) if pairs else ())
    out: dict = {"rows": [], "refused": [], "not_used": [], "file_time": set(),
                 "ignored": [column for column in header if column and column not in known]}
    text_columns = [column for column in known if column not in ("value", "of", "cost_usd", "choice", "verdict")]
    seen: Counter = Counter()
    used: set = set()
    for line, record in records:
        source = f"{name} line {line}"
        answer = record.get("verdict") or record.get("choice", "") if pairs else record.get("value", "")
        if answer == "":
            out["not_used"].append({"source": source, "kind": "row", "id": record.get("id") or None,
                                    "reason": "no verdict" if pairs else "no value"})
            continue
        raw: dict = {column: record[column] for column in text_columns if record.get(column, "") != ""}
        problem = None
        if plan is not None:
            problem = _join_pair(raw, plan, used) if pairs else _join_score(raw, plan, used)
        if pairs:
            raw["verdict"] = record.get("verdict", "")
            choice, first = record.get("choice", "").lower(), raw.get("shown_first")
            if not raw["verdict"] and choice == "tie":
                raw["verdict"] = "tie"
            elif not raw["verdict"] and choice in ("left", "right") and first in ("a", "b"):
                raw["verdict"] = first if choice == "left" else _other(first)
            elif not raw["verdict"]:
                problem = problem or (f'the choice "{record.get("choice", "")}" is not left, right or tie, or the '
                                      "row has no shown_first to read left and right by")
        else:
            comp = None if rubric is None else C.component(rubric, raw.get("component", ""))
            raw["value"] = parse_value(record["value"], comp)
            if record.get("of", "") != "":
                raw["of"] = parse_value(record["of"])
        if record.get("cost_usd", "") != "":
            raw["cost_usd"] = parse_value(record["cost_usd"])
        if rubric is not None:
            raw.setdefault("rubric", C.rubric_ref(rubric))
        key = (*(raw.get(column, "") for column in columns if column not in ("value", "of", "observed_at")),
               record.get("of", ""), answer, record.get("observed_at", ""))
        seen[key] += 1
        if "id" not in raw:
            raw["id"] = ("p-" if pairs else "r-") + _digest(*key, seen[key])
        if "call" not in raw and "item" in raw:
            raw["call"] = "c-" + _digest(raw.get("run", ""), raw.get("step", ""), raw.get("component", ""),
                                         raw.get("judge", ""), raw.get("assignment", ""),
                                         record.get("observed_at", ""))
        raw.setdefault("call", raw["id"])
        if "observed_at" not in raw:
            raw["observed_at"] = modified
            out["file_time"].add(raw["id"])
        if problem is not None:
            out["refused"].append({"table": COUNTED[table], "id": raw["id"], "source": source, "reason": problem})
        else:
            out["rows"].append((raw, source))
    return out


def import_(args: argparse.Namespace) -> int:
    folder = Path(args.out)
    if not any(getattr(args, name) for name in ("ocp", "rows", "csv", "pairs_csv", "items")):
        raise _usage("scores import", "", "no source given", "give --ocp, --rows, --csv, --pairs-csv or --items")
    if args.labels and not (args.csv or args.pairs_csv):
        raise _usage("scores import", "--labels", "a labels plan is joined to a filled form",
                     "give the form with --csv or --pairs-csv")
    if folder.exists() and not folder.is_dir():
        raise C.ContractError(str(folder), "", "not a folder", "name the study folder to build or extend", "no_file")
    rubric = C.read_document(args.rubric, C.SCHEMA_RUBRIC) if args.rubric else None
    plan = C.read_document(args.labels, C.SCHEMA_LABELS) if args.labels else None
    if args.pair_component and rubric is not None and C.component(rubric, args.pair_component) is None:
        raise _usage("scores import", "--pair-component", f'unknown component "{args.pair_component}"',
                     f"use one of {', '.join(comp['id'] for comp in rubric['components'])}", "unknown_component")
    tables = _load(folder)
    incoming: dict[str, list[tuple[dict, str]]] = {name: [] for name in TABLES}
    refused: list[dict] = []
    not_used: list[dict] = []
    ignored: dict[str, list[str]] = {}
    file_time: dict[str, set] = {name: set() for name in TABLES}
    if args.ocp or args.rows:
        from .adapters import ocp as adapter

        batches = []
        if args.ocp:
            documents = [(Path(path).name, C.load_json(path)) for path in args.ocp]
            batches.append(adapter.read_documents(documents, rubric=rubric, pair_component=args.pair_component,
                                                  inputs={row["run"]: row["input"] for row in tables["runs"]}))
        if args.rows:
            batches.append(adapter.read_rows(args.rows, rubric=rubric))
        for batch in batches:
            for name in ("runs", "scores", "pairs"):
                key = C.LINE_KEY[TABLES[name][1]]
                incoming[name] += [(row, batch.source.get((name, str(row.get(key))), ""))
                                   for row in getattr(batch, name)]
            not_used += batch.not_used
    for path, table in ((args.csv, "scores"), (args.pairs_csv, "pairs")):
        if path:
            read = _table_csv(path, table, rubric, plan)
            incoming[table] += read["rows"]
            file_time[table] |= read["file_time"]
            refused += read["refused"]
            not_used += read["not_used"]
            if read["ignored"]:
                ignored[Path(path).name] = read["ignored"]
    if args.items:
        incoming["items"] += [(row, Path(args.items).name) for row in C.read_lines(args.items, C.SCHEMA_ITEMS)]

    tallies: dict[str, Counter] = {}
    merged: dict[str, list[dict]] = {}
    added: dict[str, dict[str, str]] = {name: {} for name in TABLES}  # per table, new id to its source
    simulated_study = any(row["simulated"] for row in tables["runs"])
    for name, (file, schema) in TABLES.items():
        key, validate = C.LINE_KEY[schema], C.LINES[schema]
        rows = list(tables[name])
        have = {row[key]: row for row in rows}
        tally = Counter({"refused": sum(1 for entry in refused if entry["table"] == COUNTED[name])})
        for raw, source in incoming[name]:
            reason = None
            try:
                row = validate({**raw, "schema": schema}, source or file, f'{NOUN[name]} "{raw.get(key)}": ')
            except C.ContractError as exc:
                row, reason = None, str(exc)
            if row is not None and row[key] in have:
                old = have[row[key]]
                # a source may say less about a run than the study does (no cost, no refs), never something else
                if old == row or (name == "runs" and all(old.get(member) == value for member, value in row.items()
                                                         if value is not None)) or (
                        # a row with no time of its own took the file's time, which moves when the file is copied
                        row[key] in file_time[name] and {**old, "observed_at": None} == {**row, "observed_at": None}):
                    tally["unchanged"] += 1
                    continue
                reason = (f'the {NOUN[name]} id "{row[key]}" is already in the study with other content; give a '
                          f"changed {NOUN[name]} a new id" + ("" if name in ("runs", "items") else
                                                                ", and name the one it corrects in replaces"))
            elif row is not None and name == "runs" and not args.allow_simulated and row["simulated"]:
                reason = f'the run "{row[key]}" is simulated; --allow-simulated takes simulated runs'
            elif row is not None and name == "runs" and not args.allow_simulated and simulated_study:
                reason = (f'the study holds simulated runs and the run "{row[key]}" is not simulated; '
                          "--allow-simulated mixes the two")
            if reason is not None:
                tally["refused"] += 1
                refused.append({"table": COUNTED[name], "id": raw.get(key), "source": source, "reason": reason})
                continue
            have[row[key]] = row
            rows.append(row)
            added[name][row[key]] = source
            tally["new"] += 1
        tallies[name], merged[name] = tally, rows

    # one call gives one reading per component and one verdict per component compared: a new row that
    # shares a call with another row that is read, in the study or in this import, is refused by id
    for name in ("scores", "pairs"):
        if not added[name]:
            continue
        clashing: dict[str, str] = {}
        for found in call_problems(name, read_rows(merged[name], TABLES[name][0]), rubric):
            new = [ident for ident in found["ids"] if ident in added[name]]
            old = [ident for ident in found["ids"] if ident not in added[name]]
            first = old[0] if old else new[0]
            on = "" if found["component"] is None else f' for the component "{found["component"]}"'
            for ident in new if old else new[1:]:
                clashing.setdefault(ident, f'the {NOUN[name]} "{ident}" shares the call "{found["call"]}" with the '
                                           f'{NOUN[name]} "{first}"{on}; give a repeat its own call, or name the '
                                           f"{NOUN[name]} it corrects in replaces")
        for ident, reason in clashing.items():
            tallies[name]["new"] -= 1
            tallies[name]["refused"] += 1
            refused.append({"table": COUNTED[name], "id": ident, "source": added[name][ident], "reason": reason})
    counts = {COUNTED[name]: {word: tallies[name][word] for word in ("new", "unchanged", "refused")}
              for name in TABLES}

    written = []
    if not refused:
        for name, (file, schema) in TABLES.items():
            if tallies[name]["new"]:
                C.write_lines(folder / file, schema, merged[name])
                written.append(file)
    report = {"ok": not refused, "study": str(folder), **counts, "refused": refused, "not_used": not_used,
              "written": written}
    if ignored:
        report["ignored_columns"] = ignored
    lines = [f"study {folder}"]
    lines += [f"{name}: {c['new']} new, {c['unchanged']} unchanged, {c['refused']} refused"
              for name, c in counts.items()]
    lines += [f"not used: {entry['kind']} {entry['id']!r} ({entry['source']}): {entry['reason']}"
              for entry in not_used]
    lines += [f"ignored columns of {name}: {', '.join(columns)}" for name, columns in ignored.items()]
    lines.append(f"wrote {', '.join(written)}" if written else
                 "nothing written: a refused row refuses the import" if refused else "nothing new: no file changed")
    errors = [f"{entry['reason']}" if entry["reason"].startswith(entry["source"] or "\0") else
              f"{entry['source'] or 'import'}: {entry['reason']}" for entry in refused]
    return _finish(args, "loopmath.eval-scores-import/1", report, lines, errors)


# ---------------------------------------------------------------------------------------------------
# eval scores check

def _family(run: dict, step: str) -> str:
    for entry in run["steps"]:
        if entry["step"] == step:
            return entry["model"]["family"]
    return run["steps"][-1]["model"]["family"]


def design(rubric: dict, judges: dict | None, runs: list[dict], scores: list[dict], pairs: list[dict]) -> list[dict]:
    """The design table: per term of the model, the count the design gives and whether the need is met.

    Each row is `{"term", "judge", "dimension", "count", "met", "needs"}` plus the figures behind the count.
    `scores` and `pairs` are the rows that are read (corrections applied). Rows on an unknown run or dimension
    are left out here: `check_study` names them. Without a judges file the anchor, family, reference and
    halo rows are absent. Whether a met need gives an identified or a weak term is decided by `eval fit`."""
    dims = C.dimensions(rubric)
    kinds = {dim["id"]: dim["scale"]["kind"] for dim in dims}
    run_of = {row["run"]: row for row in runs}
    judge_rows = {row["id"]: row for row in judges["judges"]} if judges else {}
    anchors = list(judges["anchor"]) if judges else []
    calls: dict[tuple[str, str, str], set] = defaultdict(set)
    for row in scores:
        dim = C.dimension_id(row["step"], row["component"])
        if row["run"] in run_of and dim in kinds:
            calls[(row["judge"], dim, row["run"])].add(row["call"])
    pair_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pairs:
        comp = row["component"] or C.overall_component(rubric, row["step"])
        dim = C.dimension_id(row["step"], comp or "")
        if dim in kinds and row["a"] in run_of and row["b"] in run_of:
            pair_rows[(row["judge"], dim)].append(row)
    seen = {judge for judge, _, _ in calls} | {judge for judge, _ in pair_rows}
    order = [judge for judge in judge_rows if judge in seen] + sorted(seen - set(judge_rows))
    targets: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    scored_by: dict[tuple[str, str], set] = defaultdict(set)
    for (judge, dim, name), ids in calls.items():
        targets[(judge, dim)][name] = len(ids)
        scored_by[(dim, name)].add(judge)
    rows: list[dict] = []

    def add(term: str, count: int, met: bool, judge: str | None = None, dimension: str | None = None,
            **more: Any) -> None:
        rows.append({"term": term, "judge": judge, "dimension": dimension, "count": count, "met": bool(met),
                     "needs": NEEDS.get(term, ""), **more})

    per_input = Counter(row["input"] for row in runs)
    paired = sum(1 for n in per_input.values() if n >= 2)
    add("run_vs_input", paired, paired >= 1, inputs=len(per_input))
    for dim in kinds:
        for judge in order:
            by_run = targets.get((judge, dim))
            if not by_run:
                continue
            repeats = sum(1 for n in by_run.values() if n >= 2)
            shared = sum(1 for name in by_run if len(scored_by[(dim, name)]) >= 2)
            add("sigma", repeats, repeats >= 1, judge, dim, targets=len(by_run))
            add("tau", shared, repeats >= 1 and shared >= 1, judge, dim, repeats=repeats)
            inputs = Counter(run_of[name]["input"] for name in by_run)
            twice = sum(1 for n in inputs.values() if n >= 2)
            add("judge_input", twice, twice >= 10, judge, dim, inputs=len(inputs))
        if judges is not None:
            anchored = {name for judge in anchors for name in targets.get((judge, dim), {})}
            floor = judges["min_anchor_targets"]
            state = ("relative" if len(anchored) < floor else "weak" if len(anchored) < C.WEAK_ANCHOR_BELOW
                     else "anchored")
            rows.append({"term": "anchor", "judge": None, "dimension": dim, "count": len(anchored),
                         "met": len(anchored) >= floor, "state": state,
                         "needs": f"{floor} targets scored by an anchor judge, {C.WEAK_ANCHOR_BELOW} for a firm "
                                  "anchor"})
        if kinds[dim] == "binary":
            names = {name for (judge, d, name) in calls if d == dim}
            known = sum(1 for name in names if len(scored_by[(dim, name)]) >= 2
                        or any(targets.get((judge, dim), {}).get(name, 0) >= 2 for judge in anchors))
            add("binary_status", known, known >= 1, None, dim, targets=len(names))
    if judges is not None:
        families: dict[str, set] = defaultdict(set)
        for judge, dim, name in calls:
            families[judge].add(_family(run_of[name], dim.split("/", 1)[0]))
        for (judge, dim), group in pair_rows.items():
            for row in group:
                families[judge].update(_family(run_of[row[side]], row["step"]) for side in ("a", "b"))
        models = {judge: row for judge, row in judge_rows.items() if row["kind"] == "model" and families.get(judge)}
        wide = {row["model"]["family"] for judge, row in models.items() if len(families[judge]) >= 2}
        add("family", len(wide), len(wide) >= 2, output_families=len({f for fs in families.values() for f in fs}))
        for judge, row in models.items():
            author = (row["reference"].get("author") or {}).get("model") if row["reference"]["shown"] else None
            if author is not None:
                same = author["family"] == row["model"]["family"]
                add("reference_family", 0 if same else 1, not same, judge,
                    state="confounded" if same else "separate")
        groups: dict[tuple, list[str]] = defaultdict(list)
        for judge, row in models.items():
            groups[(row["model"]["id"], row.get("effort"), row.get("prompt"))].append(judge)
        for members in groups.values():
            panel = [judge for judge in members if judge_rows[judge]["mode"] == "panel"]
            if not panel:
                continue
            per_target: dict[tuple[str, str, str], set] = defaultdict(set)
            for (judge, dim, name), ids in calls.items():
                if judge in panel:
                    per_target[(judge, dim.split("/", 1)[0], name)] |= ids
            repeats = sum(1 for ids in per_target.values() if len(ids) >= 2)
            add("halo", repeats, len(panel) < len(members) or repeats >= 1, ", ".join(members),
                modes=sorted({judge_rows[judge]["mode"] for judge in members}))
    dim_order = list(kinds)
    for judge, dim in sorted(pair_rows, key=lambda key: (dim_order.index(key[1]), order.index(key[0]))):
        group = pair_rows[(judge, dim)]
        shown: dict[tuple, set] = defaultdict(set)
        ids: dict[tuple, set] = defaultdict(set)
        first: Counter = Counter()
        for row in group:
            key = tuple(sorted((row["a"], row["b"])))
            ids[key].add(row["call"])
            if row["shown_first"] is not None:
                shown[key].add(row[row["shown_first"]])
                first[row["shown_first"]] += 1
        both = sum(1 for names in shown.values() if len(names) == 2)
        repeats = sum(1 for calls_ in ids.values() if len(calls_) >= 2)
        ties = sum(1 for row in group if row["verdict"] == "tie")
        add("position", both, both >= 1 or (first["a"] >= 1 and first["b"] >= 1), judge, dim, first_a=first["a"],
            first_b=first["b"])
        add("sigma_pair", repeats, repeats >= 1, judge, dim, pairs=len(ids))
        add("tie_width", ties, repeats >= 1 and ties >= 1, judge, dim, repeats=repeats)
    return rows


def check_study(study: C.Study, rubric: dict, judges: dict | None = None, *,
                require_assignment: bool = False) -> dict:
    """Check a study against a normalised rubric and judges file. Returns the report `eval scores check` prints.

    `problems` lists every refusal, each `{"file", "where", "problem", "fix", "code", "message"}`; `ok` is
    true when there is none. Nothing is raised for a problem in the data, so one run names them all."""
    found: list[C.ContractError] = []

    def bad(file: str, where: str, problem: str, fix: str, code: str) -> None:
        found.append(C.ContractError(file, where, problem, fix, code))

    ref = C.rubric_ref(rubric)
    runs = {row["run"]: row for row in study.runs}
    comps = {comp["id"]: comp for comp in rubric["components"]}
    judge_rows = None if judges is None else {row["id"]: row for row in judges["judges"]}
    items = {row["id"]: row for row in study.items}
    steps = rubric["steps"]

    def shared(row: dict, file: str, place: str, comp_id: str | None) -> dict | None:
        """The checks a reading and a pair have in common. Returns the component when it is known."""
        if row["rubric"] != ref:
            bad(file, f"{place}: rubric", f'unknown rubric version "{row["rubric"]}"',
                f'this check reads "{ref}": give the rubric the row was scored under', "unknown_rubric")
        if judge_rows is not None and row["judge"] not in judge_rows:
            bad(file, f"{place}: judge", f'unknown judge "{row["judge"]}"',
                f"use one of {', '.join(judge_rows)}, or add the judge to the judges file", "unknown_judge")
        if row["step"] not in steps:
            bad(file, f"{place}: step", f'unknown step "{row["step"]}"', f"use one of {', '.join(steps)}",
                "unknown_step")
        if comp_id is None:
            bad(file, f"{place}: component", f'the pair names no component and step "{row["step"]}" has no overall '
                "component", "name the component compared", "unknown_component")
            return None
        comp = comps.get(comp_id)
        if comp is None:
            bad(file, f"{place}: component", f'unknown component "{comp_id}"', f"use one of {', '.join(comps)}",
                "unknown_component")
        elif row["step"] in steps and row["step"] not in comp["applies_to"]:
            bad(file, f"{place}: step", f'the component "{comp_id}" is not scored at step "{row["step"]}"',
                f"it applies to {', '.join(comp['applies_to'])}", "unknown_step")
        if (require_assignment and judge_rows is not None and "assignment" not in row
                and judge_rows.get(row["judge"], {}).get("kind") == "person"):
            bad(file, place, f'the person "{row["judge"]}" gave it with no assignment',
                "name the labels next assignment that asked for it", "no_assignment")
        return comp

    for row in study.scores:
        place = f'id "{row["id"]}"'
        comp = shared(row, C.SCORES_FILE, place, row["component"])
        if comp is not None:
            try:
                C.check_reading(row, comp, C.SCORES_FILE, f"{place}: ")
            except C.ContractError as exc:
                found.append(exc)
        run_row = runs.get(row["run"])
        if run_row is None:
            bad(C.SCORES_FILE, f"{place}: run", f'no run "{row["run"]}" in {C.RUNS_FILE}',
                "import the run, or fix the reading", "no_run")
        if "item" in row:
            item = items.get(row["item"])
            if item is None:
                bad(C.SCORES_FILE, f"{place}: item", f'unknown item "{row["item"]}"',
                    f"add it to {C.ITEMS_FILE} (eval scores import --items)", "unknown_item")
            elif item["component"] != row["component"]:
                bad(C.SCORES_FILE, f"{place}: item", f'the item "{row["item"]}" belongs to the component '
                    f'"{item["component"]}"', "name an item of the component read", "unknown_item")
            elif run_row is not None and item["input"] != run_row["input"]:
                bad(C.SCORES_FILE, f"{place}: item", f'the item "{row["item"]}" is written for the input '
                    f'"{item["input"]}" and the run is on "{run_row["input"]}"', "name an item of the run's input",
                    "unknown_item")
    for row in study.pairs:
        place = f'id "{row["id"]}"'
        shared(row, C.PAIRS_FILE, place, row["component"] or C.overall_component(rubric, row["step"]))
        missing = [side for side in ("a", "b") if row[side] not in runs]
        for side in missing:
            bad(C.PAIRS_FILE, f"{place}: {side}", f'no run "{row[side]}" in {C.RUNS_FILE}',
                "import the run, or fix the pair", "no_run")
        if not missing and runs[row["a"]]["input"] != runs[row["b"]]["input"]:
            bad(C.PAIRS_FILE, place, f'the runs "{row["a"]}" and "{row["b"]}" have different inputs '
                f'("{runs[row["a"]]["input"]}" and "{runs[row["b"]]["input"]}")', "compare two runs of one input",
                "different_inputs")
    for row in study.items:
        comp = comps.get(row["component"])
        if comp is None:
            bad(C.ITEMS_FILE, f'id "{row["id"]}": component', f'unknown component "{row["component"]}"',
                f"use one of {', '.join(comps)}", "unknown_component")
        elif comp["scale"]["kind"] != "items":
            bad(C.ITEMS_FILE, f'id "{row["id"]}": component', f'the component "{row["component"]}" has a '
                f'"{comp["scale"]["kind"]}" scale', 'name a component with the scale kind "items"', "not_items")

    read: dict[str, list[dict]] = {}
    not_read: list[dict] = []
    for name, rows, file in (("scores", study.scores, C.SCORES_FILE), ("pairs", study.pairs, C.PAIRS_FILE)):
        ids = {row["id"] for row in rows}
        dangling = {row["id"] for row in rows if row.get("replaces") is not None and row["replaces"] not in ids}
        for row in rows:
            if row["id"] in dangling:
                bad(file, f'id "{row["id"]}": replaces', f'"{row["replaces"]}" names no {NOUN[name]}',
                    f"name the id of the {NOUN[name]} it corrects", "replaces_unknown")
        rows = [{**row, "replaces": None} if row["id"] in dangling else row for row in rows]
        try:
            read[name], dropped = C.live(rows, file)
        except C.ContractError as exc:
            found.append(exc)
            read[name], dropped = list(rows), []
        not_read += [{"table": COUNTED[name], **entry} for entry in dropped]

    for name, file in (("scores", C.SCORES_FILE), ("pairs", C.PAIRS_FILE)):
        for entry in call_problems(name, read[name], rubric):
            bad(file, f'call "{entry["call"]}"', entry["problem"], entry["fix"], entry["code"])
    spans: dict[str, set] = defaultdict(set)
    for row in read["scores"]:
        spans[row["call"]].add((row["judge"], row["run"], row["step"]))
    warnings = [f'{C.SCORES_FILE}: call "{call}": shared by {len(who)} pairs of judge and target; a call is one '
                "judge scoring one target" for call, who in spans.items() if len(who) > 1]

    arms: dict[str, dict[tuple, int]] = defaultdict(dict)
    for row in study.runs:
        setting = tuple((entry["step"], entry["model"]["id"], entry.get("effort"), entry.get("variant"))
                        for entry in row["steps"])
        arms[row["arm"]][setting] = arms[row["arm"]].get(setting, 0) + 1
    mixed = [{"arm": arm, "runs": sum(groups.values()),
              "settings": [{"runs": n, "steps": [list(entry) for entry in setting]} for setting, n in groups.items()]}
             for arm, groups in arms.items() if len(groups) > 1]

    return {"ok": not found, "rubric": ref,
            "counts": {"runs": len(study.runs), "readings": len(study.scores), "readings_read": len(read["scores"]),
                       "pairs": len(study.pairs), "pairs_read": len(read["pairs"]), "items": len(study.items)},
            "problems": [{"file": exc.file, "where": exc.where, "problem": exc.problem, "fix": exc.fix,
                          "code": exc.code, "message": str(exc)} for exc in found],
            "warnings": warnings, "mixed_arm": mixed, "not_read": not_read,
            "design": design(rubric, judges, study.runs, read["scores"], read["pairs"])}


def _design_lines(rows: list[dict]) -> list[str]:
    table = [("term", "judge", "dimension", "count", "met", "needs")]
    for row in rows:
        state = row.get("state")
        table.append((row["term"], row["judge"] or "-", row["dimension"] or "-", str(row["count"]),
                      state if state else "yes" if row["met"] else "no", row["needs"]))
    widths = [max(len(line[n]) for line in table) for n in range(5)]
    return ["  ".join([*(cell.ljust(widths[n]) for n, cell in enumerate(line[:5])), line[5]]) for line in table]


def check(args: argparse.Namespace) -> int:
    study = C.read_study(args.study)
    rubric = C.read_document(args.rubric, C.SCHEMA_RUBRIC)
    judges = C.read_document(args.judges, C.SCHEMA_JUDGES) if args.judges else None
    if args.require_assignment and judges is None:
        raise _usage("scores check", "--require-assignment", "the judges file says who is a person",
                     "give --judges with it")
    report = {"study": str(study.folder), **check_study(study, rubric, judges,
                                                        require_assignment=args.require_assignment)}
    counts = report["counts"]
    lines = [f"study {study.folder}: {counts['runs']} runs, {counts['readings']} readings "
             f"({counts['readings_read']} read), {counts['pairs']} pairs ({counts['pairs_read']} read), "
             f"{counts['items']} items", f"rubric {report['rubric']}"]
    lines += [f"warning: {text}" for text in report["warnings"]]
    lines += [f"mixed_arm: {entry['arm']}: {len(entry['settings'])} settings over {entry['runs']} runs"
              for entry in report["mixed_arm"]] or ["mixed_arm: none"]
    lines += ["", "design (what it can identify):", *_design_lines(report["design"]), ""]
    lines.append("ok" if report["ok"] else f"{len(report['problems'])} problems")
    return _finish(args, "loopmath.eval-scores-check/1", report, lines,
                   [entry["message"] for entry in report["problems"]])


# ---------------------------------------------------------------------------------------------------
# eval scores add

def _free_id(prefix: str, taken: set, width: int = 4) -> str:
    n = len(taken) + 1
    while f"{prefix}{n:0{width}d}" in taken:
        n += 1
    return f"{prefix}{n:0{width}d}"


def add(args: argparse.Namespace) -> int:
    study = C.read_study(args.study)
    rubric = C.read_document(args.rubric, C.SCHEMA_RUBRIC)
    pair = args.pair is not None
    if pair == (args.run is not None):
        raise _usage("scores add", "", "give --run for a reading or --pair for a pair, one of the two",
                     "name the run scored, or the two runs compared")
    given = {name: getattr(args, name) is not None for name in ("value", "item", "of", "verdict", "shown_first")}
    wrong = [name for name in (("value", "item", "of") if pair else ("verdict", "shown_first")) if given[name]]
    if wrong:
        raise _usage("scores add", f"--{wrong[0].replace('_', '-')}",
                     f"not an option of a {'pair' if pair else 'reading'}", "remove it")
    table = "pairs" if pair else "scores"
    file, schema = TABLES[table]
    rows = study.pairs if pair else study.scores
    runs = {row["run"]: row for row in study.runs}
    for name in args.pair if pair else [args.run]:
        if name not in runs:
            raise _usage("scores add", "--pair" if pair else "--run", f'no run "{name}" in {C.RUNS_FILE}',
                         "name a run of the study, or import it first", "no_run")
    comp_id = args.component or (C.overall_component(rubric, args.step) if pair else None)
    comp = None if comp_id is None else C.component(rubric, comp_id)
    if comp_id is None:
        raise _usage("scores add", "--component", "missing" if not pair else
                     f'step "{args.step}" has no overall component', "name the rubric component", "unknown_component")
    if comp is None:
        raise _usage("scores add", "--component", f'unknown component "{comp_id}"',
                     f"use one of {', '.join(c['id'] for c in rubric['components'])}", "unknown_component")
    if args.step not in comp["applies_to"]:
        raise _usage("scores add", "--step", f'the component "{comp_id}" is not scored at step "{args.step}"',
                     f"it applies to {', '.join(comp['applies_to'])}", "unknown_step")
    taken = {row["id"] for row in rows}
    ident = args.id or _free_id("p" if pair else "s", taken, 3 if pair else 4)
    if ident in taken:
        raise _usage("scores add", "--id", f'the {NOUN[table]} id "{ident}" is already in {file}',
                     "give a new id, and name the one it corrects in --replaces", "duplicate")
    if args.replaces is not None and args.replaces not in taken:
        raise _usage("scores add", "--replaces", f'"{args.replaces}" names no {NOUN[table]} of {file}',
                     f"name the id of the {NOUN[table]} it corrects", "replaces_unknown")
    raw: dict = {"schema": schema, "id": ident, "step": args.step, "judge": args.judge,
                 "call": args.call or f"c-{ident}", "rubric": C.rubric_ref(rubric),
                 "observed_at": args.observed_at or _now(), "replaces": args.replaces, "assignment": args.assignment}
    if pair:
        if args.verdict is None:
            raise _usage("scores add", "--verdict", "missing", "give a, b or tie")
        if runs[args.pair[0]]["input"] != runs[args.pair[1]]["input"]:
            raise _usage("scores add", "--pair", f'the runs "{args.pair[0]}" and "{args.pair[1]}" have different '
                         "inputs", "compare two runs of one input", "different_inputs")
        raw.update(a=args.pair[0], b=args.pair[1], component=args.component, verdict=args.verdict,
                   shown_first=args.shown_first)
        row = C.validate_pair(raw, file, "")
    else:
        if args.value is None:
            raise _usage("scores add", "--value", "missing", "give the raw score on the component's scale")
        raw.update(run=args.run, component=comp_id, value=parse_value(args.value, comp), item=args.item, of=args.of)
        row = C.validate_score(raw, file, "")
        C.check_reading(row, comp, file, "")
        if args.item is not None:
            item = {entry["id"]: entry for entry in study.items}.get(args.item)
            if item is None or item["component"] != comp_id or item["input"] != runs[args.run]["input"]:
                raise _usage("scores add", "--item", f'no item "{args.item}" of the component "{comp_id}" for the '
                             f'input "{runs[args.run]["input"]}" in {C.ITEMS_FILE}',
                             "name an item of the run's input, or import the items first", "unknown_item")
    for found in call_problems(table, read_rows([*rows, row], file), rubric):
        if ident in found["ids"]:
            other = next(name for name in found["ids"] if name != ident)
            raise _usage("scores add", "--call", f'the call "{row["call"]}" already has the {NOUN[table]} "{other}" '
                         f'for the component "{comp_id}"', f"give a repeat its own call, or name the {NOUN[table]} "
                         "it corrects in --replaces", "call_twice")
    C.write_lines(study.folder / file, schema, [*rows, row])
    report = {"ok": True, "study": str(study.folder), "added": NOUN[table], "file": file,
              "row": {key: value for key, value in row.items() if value is not None}}
    return _finish(args, "loopmath.eval-scores-add/1", report, [f"added {NOUN[table]} {ident} to {file}"])


# ---------------------------------------------------------------------------------------------------
# eval scores export

def export(args: argparse.Namespace) -> int:
    from .adapters import ocp as adapter

    study = C.read_study(args.study)
    known = {row["run"] for row in study.runs}
    for name in args.run or []:
        if name not in known:
            raise _usage("scores export", "--run", f'no run "{name}" in {C.RUNS_FILE}', "name a run of the study",
                         "no_run")
    persons: list[str] = []
    if (study.folder / "judges.json").is_file():  # `eval simulate` writes one beside the tables
        judges = C.read_document(study.folder / "judges.json", C.SCHEMA_JUDGES)
        persons = [row["id"] for row in judges["judges"] if row["kind"] == "person"]
    entries, not_used = adapter.export(study.runs, study.scores, study.pairs, only=args.run, persons=persons)
    if args.run is None:
        entries = [entry for entry in entries if entry["signals"] or entry["preferences"]]
    if args.json:
        emit_json("loopmath.eval-scores-export/1", {"ok": True, "study": str(study.folder), "runs": entries,
                                                    "not_used": not_used})
        return EXIT_OK
    for entry in entries:
        print(json.dumps(entry, ensure_ascii=False, allow_nan=False))
    for entry in not_used:
        print(f"not used: {entry['kind']} {entry['id']!r}: {entry['reason']}", file=sys.stderr)
    return EXIT_OK
