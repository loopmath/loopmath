"""The agent skill doc, `loopmath harness skill`: how an agent drives the tune CLI for a person.

Generated from `schema.FIELDS`, `events.TYPES` and `contracts.EXIT`, so the doc cannot drift from the config file,
the flags or the event stream. `markdown()` is the doc; `data()` is the same doc as data (`--format json`).
"""

from __future__ import annotations

from typing import Any

from . import CONFIG_SCHEMA, EVENTS_SCHEMA
from .contracts import EXIT
from .events import TYPES
from .schema import EDITORS, FIELDS, TABLES, Field, flag, table_fields

__all__ = ["SCHEMA", "data", "markdown"]

SCHEMA = "loopmath.harness-skill/0.1"

WHEN = ("You have a task you will run again and again (a kind of change in one repo), and a way to score a run: a "
        "command that checks the work (tests, a script that prints a number) or the tester judge, which writes "
        "acceptance tests once per task. The tune proposes edits to the repo's Claude Code harness (CLAUDE.md, "
        "skills, agents, settings hooks), measures each edit in a screening design of task runs, and delivers "
        "the kept edits only when they beat the base.")

NOT_WHEN = ("Not for a one-off change, and not without a scorer: with nothing to score the runs, the plan refuses. "
            "Every run spends model calls, so the person agrees to the spend first.")

FLOW = (
    ("init", "loopmath harness init", "Write loopmath-harness.toml in the repo. Set task.prompt (or "
     "task.prompt_file), score.command when there is a check, and auth.token_file (a `claude setup-token` token, "
     "mode 600). `loopmath harness config set KEY VALUE` edits one key and keeps the comments."),
    ("plan", "loopmath harness plan --json", "The dry run, with no model call. Show the person the sessions, the "
     "cost point and range, the worst-case time, the warnings and any failed preflight check."),
    ("ask", "", "Ask the person to agree to the spend: up to budget.usd, with the plan's point and range. Do not go "
     "on without a yes."),
    ("run", "loopmath harness run --detach --yes --json", "Only after the yes. It prints the tune id and returns; "
     "the engine runs on in its own session."),
    ("watch", "loopmath harness watch ID --json", "Follow the events (one JSON object per line) to tune.end, or "
     "`loopmath harness status ID --json` for a snapshot. The exit code is the tune's."),
    ("show", "loopmath harness show ID --effects", "What was delivered and why, and the effects table. Report it "
     "plainly, including a base delivery."),
    ("apply", "loopmath harness apply ID --dry-run", "Show the person which files would change. Only after they "
     "say yes, run `loopmath harness apply ID`. It writes the files and commits nothing."),
)

RULES = (
    "Never run a tune before the person agrees to its spend, and never apply a harness before they agree to it.",
    "Never raise budget.usd, budget.runs, budget.wall or budget.set_usd unasked.",
    "When a tune ends no-signal (exit 4), say plainly that the measurements could not separate an edit from the "
    "base, so the base is delivered. That is the rule working, not a failure to retry.",
    "A base delivery (exit 0, status base) means the tuned build did not beat the base. Do not apply edits by hand "
    "around it.",
    "A paused tune (exit 6) hit a usage limit or an auth error. Tell the person; `loopmath harness resume ID` "
    "continues it once they say so.",
    "To stop a tune, run `loopmath harness stop ID` (running sessions finish) or `--now` (they are killed).",
    "Read the event stream and the CLI's JSON; never read session transcripts or the token file.",
)

COMMANDS = (
    ("init [--out PATH] [--force] [flags]", "write the config file with one comment per field"),
    ("plan [flags]", "the dry run: stages, sessions, cost point and range, time, df, warnings, preflight"),
    ("run [flags] [--detach] [--yes]", "start a tune and print its id"),
    ("status [ID] [--line]", "a snapshot folded from the event stream"),
    ("watch [ID] [--json] [--from SEQ]", "follow the events until tune.end"),
    ("stop [ID] [--now]", "write the stop file"),
    ("resume ID [--detach]", "continue a paused or interrupted tune"),
    ("show [ID] [--diff] [--effects] [--ledger]", "the delivered harness and why"),
    ("apply [ID] [--to REPO] [--work] [--dry-run]", "write the delivered harness into the repo; commits nothing"),
    ("list", "tunes, newest first"),
    ("config get|set|unset KEY [VALUE]", "read or edit the config file in place, keeping comments"),
    ("schema --json", "the config's JSON Schema"),
    ("skill [--format md|json]", "this doc"),
    ("mod install|uninstall|path|validate [--print]", "the Claude Code mod (/harness); install prints each claude "
     "command and asks before running it"),
    ("herdr link|unlink|path [--print]", "the herdr plugin; link prints the herdr command and asks before running it"),
)

EVENT_NOTES = {
    "tune.start": "the config digest and the plan",
    "stage.start": "a stage begins, with its planned session count",
    "stage.end": "a stage ends, with its counts",
    "session.start": "a model session starts: role, model, pid",
    "session.end": "it ends: exit, wall time, cost, tokens",
    "run.score": "an executor run's score",
    "edit.proposed": "an editor proposed an edit",
    "edit.refused": "an edit broke a rule",
    "edit.kept": "an edit enters the design",
    "design": "the probe design: matrix, run order, residual df",
    "fit": "the effect of each edit, and which are kept",
    "build": "the kept set built into one harness",
    "deliver": "the delivery choice, the rule and why",
    "signal": "a line about the measurements (ceiling, floor, no separation)",
    "budget": "spend against a cap",
    "stop.requested": "the stop file was seen",
    "error": "an error",
    "tune.paused": "a usage limit or an auth error paused the tune",
    "tune.resume": "resume continued it",
    "tune.end": "the end status, the delivered harness and the totals",
}

EXIT_MEANING = {
    "delivered": "the tuned harness is delivered",
    "base": "the base is delivered (by the delivery rule)",
    "error": "an error",
    "failed": "the tune failed",
    "usage": "a usage or config error",
    "refused": "refused by the budget or the preflight, before any session",
    "budget": "a cap ended the tune",
    "no-signal": "not enough signal: the base is delivered",
    "stopped": "stopped",
    "paused": "paused; resume continues it",
}


def _default(f: Field) -> Any:
    return list(f.default) if isinstance(f.default, list) else f.default


def data() -> dict:
    """The doc as data."""
    fields = []
    for f in FIELDS:
        fields.append({"key": f.key, "type": f.type, "default": _default(f), "flag": flag(f), "help": f.help,
                       **({"choices": list(f.choices)} if f.choices else {}),
                       **({"note": f.note} if f.note else {})})
    return {"schema": SCHEMA, "name": "loopmath-harness", "config_schema": CONFIG_SCHEMA,
            "events_schema": EVENTS_SCHEMA, "when": WHEN, "not_when": NOT_WHEN,
            "flow": [{"step": s, "command": c, "what": w} for s, c, w in FLOW],
            "rules": list(RULES),
            "commands": [{"command": f"loopmath harness {c}", "what": w} for c, w in COMMANDS],
            "fields": fields, "editors": [dict(e) for e in EDITORS],
            "events": [{"type": t, "data": list(keys), "what": EVENT_NOTES.get(t, "")} for t, keys in TYPES.items()],
            "exit": [{"status": k, "code": v, "what": EXIT_MEANING.get(k, "")} for k, v in EXIT.items()]}


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, list):
        text = "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value) + "]"
    elif isinstance(value, str):
        text = f'"{value}"'
    else:
        text = str(value)
    return text.replace("|", "\\|")


def markdown() -> str:
    """The doc as markdown."""
    d = data()
    out = ["# loopmath harness", "", f"Tune a Claude Code harness on a repeatable task. Config schema "
           f"`{CONFIG_SCHEMA}`, events `{EVENTS_SCHEMA}`.", "", "## When to use it", "", WHEN, "", NOT_WHEN, "",
           "## The flow", ""]
    for i, step in enumerate(d["flow"], start=1):
        cmd = f"`{step['command']}`: " if step["command"] else ""
        out.append(f"{i}. {cmd}{step['what']}")
    out += ["", "## Rules", ""] + [f"- {r}" for r in RULES]
    out += ["", "## Commands", "", "Every command takes `--json`.", "", "| Command | Does |", "|---|---|"]
    out += ["| `" + c["command"].replace("|", "\\|") + f"` | {c['what']} |" for c in d["commands"]]
    out += ["", "## Config fields", "",
            "Every field has a flag. A flag wins over the file, and the file over the default. `--set KEY=VALUE` "
            "takes any dotted key and is repeatable. `--editors N` sets the number of editors; `--editor-model M` "
            "and the other `--editor-*` flags set every editor; `--set editors.2.model=M` sets one."]
    for table in TABLES:
        title = "[[editors]]" if table == "editors" else f"[{table}]"
        out += ["", f"### {title}", "", "| Key | Default | Flag | What |", "|---|---|---|---|"]
        for f in table_fields(table):
            what = f.help + (f" ({' / '.join(map(str, f.choices))})" if f.choices else "")
            out.append(f"| `{f.key}` | `{_cell(_default(f))}` | `{flag(f)}` | {what.replace('|', '/')} |")
        if table == "editors":
            names = ", ".join(e["name"] for e in EDITORS)
            out += ["", f"The defaults are {len(EDITORS)} editors: {names}."]
    out += ["", "## Events", "", "`loopmath harness watch ID --json` prints one event per line: "
            '`{"v", "seq", "ts", "tune", "type", "stage", "label", "data"}`.', "",
            "| Type | Data keys | What |", "|---|---|---|"]
    out += [f"| `{e['type']}` | {', '.join(e['data']) or 'none'} | {e['what']} |" for e in d["events"]]
    out += ["", "## Exit codes", "", "| Code | Status | What |", "|---|---|---|"]
    out += [f"| {e['code']} | {e['status']} | {e['what']} |" for e in d["exit"]]
    return "\n".join(out) + "\n"
