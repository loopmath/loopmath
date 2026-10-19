"""A check: one paired A/B of two arms, kept at `checks/<chk>.json` in the store.

`loopmath check-change start` writes it with its pairs: a task each (from `--tasks`, or from the pair's first
run with `--n`) and a random arm order. `run record --check CHK --arm base|cand` (and `run start`) claims the
earliest pair without that arm (`claim`), so the run gets the pair's task and opens or joins the pair's slate,
then tags the run and records it in the check (`commit`). The planned arm order balances which arm runs first;
a run recorded out of it is kept, with a warning, and the pair keeps the order its runs came in (`ran`), which
`status` marks (`out_of_order`). `status` reads the pairs' runs back.

loopmath never creates the worktrees or starts the agents: `next_view` prints those steps, and the blinded
referee's, as the record-run skill describes them.
"""

from __future__ import annotations

import json
import random
import re
import shlex
from pathlib import Path
from typing import Any

from ..store import runs as R
from ..store.commands import UserError
from ..store.home import NotFound, Store, StoreError
from ..store.ids import new_id, now_iso
from ..store.lock import atomic_write_json
from . import CHECK_EXT, PAIR_EXT

CHECK_SCHEMA = "loopmath.check/1"
ARMS = ("base", "cand")
CHK_RE = re.compile(r"^chk_[0-9A-Za-z]+$")
# the task flags a claimed pair with a task refuses: the pair's task file replaces them
TASK_FLAGS = (("--task-file", "task_file"), ("--type", "task_type"), ("--repo", "repo"), ("--subtype", "subtype"),
              ("--title", "title"), ("--feature", "feature"), ("--base-commit", "base_commit"))
TASK_VIEW = ("id", "type", "repo", "title", "base_commit")
REFEREE = ("a fresh agent of another model family (config referee.model) sees the two diffs as A and B in random "
           "order and names the better one, or a tie; it is told nothing about which arm is which")


# ---------------------------------------------------------------- the record
def checks_dir(store: Store) -> Path:
    return Path(store.home) / "checks"


def check_path(store: Store, chk: str) -> Path:
    if not isinstance(chk, str) or not CHK_RE.match(chk):
        raise UserError(f"a check id is chk_ and letters or digits, not {chk!r}")
    return checks_dir(store) / f"{chk}.json"


def load(store: Store, chk: str) -> dict[str, Any]:
    path = check_path(store, chk)
    if not path.is_file():
        raise NotFound(f"no check {chk} in {store.home}; loopmath check-change start makes one")
    try:
        check = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StoreError(f"{path}: not JSON: {exc}") from None
    if not isinstance(check, dict) or check.get("schema") != CHECK_SCHEMA:
        raise StoreError(f"{path}: not a {CHECK_SCHEMA} record")
    return check


def save(store: Store, check: dict[str, Any]) -> Path:
    checks_dir(store).mkdir(parents=True, exist_ok=True)
    return atomic_write_json(check_path(store, check["check"]), check)


def latest(store: Store) -> str:
    """The newest check: ids are ULIDs, so they sort by creation time."""
    found = sorted(p.stem for p in checks_dir(store).glob("chk_*.json")) if checks_dir(store).is_dir() else []
    if not found:
        raise NotFound(f"no checks in {store.home}; loopmath check-change start makes one")
    return found[-1]


PATH_LIKE = ("/", "./", "../", "~")  # an arm that starts so is a path, never a label


def arm_spec(value: str, flag: str = "the arm") -> dict[str, Any]:
    """An arm: a free label, or a path to the harness tree it runs (its CLAUDE.md, skills, hooks). A tree gets
    its harness record when `loopmath.harness` is present. A value that looks like a path (`PATH_LIKE`, `.`,
    `..`) must be a folder; a label may hold a slash inside, as a branch name does."""
    if not value or not value.strip():
        raise UserError("an arm is a label or a path to its harness tree")
    arm: dict[str, Any] = {"label": value}
    path = Path(value).expanduser()
    if value.startswith(PATH_LIKE) or value in (".", ".."):
        if not path.exists():  # a missing path exits 2, as a missing --tasks or gate --runs file does
            raise NotFound(f"{flag} {value}: no folder there; give the arm's harness tree, or a label that does "
                           f"not start with / . or ~")
        if not path.is_dir():
            raise UserError(f"{flag} {value}: not a folder; an arm's harness tree is a folder")
    if path.is_dir():
        arm["path"] = str(path.resolve())
        try:
            from ..harness import record_from_tree  # type: ignore[attr-defined]
        except ImportError:
            pass
        else:
            rec = record_from_tree(path)
            if isinstance(rec, dict) and isinstance(rec.get("digest"), str):
                arm["harness"] = rec["digest"]
    return arm


def read_tasks(path: str) -> list[dict[str, Any]]:
    """One task per line as JSON, as `plan --backlog` reads them; blank lines are skipped."""
    p = Path(path).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NotFound(f"no tasks file {path}") from None
    tasks = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except ValueError as exc:
            raise UserError(f"{path}:{i}: not JSON: {exc}") from None
        if not isinstance(task, dict) or not task.get("type") or not task.get("repo"):
            raise UserError(f'{path}:{i}: a task is an object with a type and a repo, such as '
                            '{"type": "bug_fix", "repo": "acme/api", "title": "..."}')
        tasks.append(task)
    if not tasks:
        raise UserError(f"{path}: no tasks")
    return tasks


def new_check(store: Store, *, base: dict[str, Any], cand: dict[str, Any], tasks: list[dict[str, Any]] | None,
              n: int | None, rule: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Write a new check with its pairs. Each pair's arm order is drawn from the check id and the seed."""
    chk = new_id("chk")
    count = len(tasks) if tasks is not None else int(n or 0)
    if count < 1:
        raise UserError("a check needs at least one pair: --tasks FILE.jsonl or --n N")
    rng = random.Random(f"{chk}:{settings.get('seed', 0)}")
    width = max(2, len(str(count)))
    pairs = []
    for i in range(count):
        task = None
        if tasks is not None:
            task = dict(tasks[i])
            task.setdefault("id", new_id("tsk"))  # both runs of the pair share it: a slate holds one task
        order = list(ARMS)
        rng.shuffle(order)
        pairs.append({"pair": f"p{i + 1:0{width}d}", "task": task, "order": order, "slate": None,
                      "runs": {arm: None for arm in ARMS}})
    check = {"schema": CHECK_SCHEMA, "check": chk, "created_at": now_iso(), "base": base, "cand": cand,
             "rule": rule, "settings": settings, "pairs": pairs}
    store.ensure()
    save(store, check)
    return check


def pair_of(check: dict[str, Any], key: str) -> dict[str, Any]:
    for p in check["pairs"]:
        if p["pair"] == key:
            return p
    raise NotFound(f"check {check['check']} has no pair {key} (pairs {check['pairs'][0]['pair']} to "
                   f"{check['pairs'][-1]['pair']})")


def next_pair(check: dict[str, Any]) -> dict[str, Any] | None:
    """The earliest pair still missing a run."""
    return next((p for p in check["pairs"] if not all(p["runs"].values())), None)


# ---------------------------------------------------------------- recording into a check
def claim(store: Store, args: Any) -> dict[str, Any]:
    """Before the run opens: choose the pair and point `args` at its task file and slate. Nothing is written
    to the check yet; `commit` does that once the run exists."""
    chk, arm = args.check, getattr(args, "arm", None)
    if arm not in ARMS:
        raise UserError(f"--check {chk} needs --arm base or --arm cand")
    if getattr(args, "slate", None) or getattr(args, "new_slate", False):
        raise UserError("--check gives the run its pair's slate; drop --slate and --new-slate")
    check = load(store, chk)
    if getattr(args, "pair", None):
        pair = pair_of(check, args.pair)
        if pair["runs"][arm]:
            raise UserError(f"pair {pair['pair']} of {chk} already has its {arm} run {pair['runs'][arm]}")
    else:
        pair = next((p for p in check["pairs"] if not p["runs"][arm]), None)
        if pair is None:
            raise UserError(f"every pair of {chk} has its {arm} run; start a check with more tasks")
    if pair["task"] is not None:
        given = [flag for flag, attr in TASK_FLAGS if getattr(args, attr, None)]
        if given:
            raise UserError(f"pair {pair['pair']} of {chk} has its task; drop {', '.join(given)}")
        task_dir = checks_dir(store) / chk
        task_dir.mkdir(parents=True, exist_ok=True)
        args.task_file = str(atomic_write_json(task_dir / f"{pair['pair']}.task.json", pair["task"]))
    args.slate, args.new_slate = (pair["slate"], False) if pair["slate"] else (None, True)
    return {"check": chk, "arm": arm, "pair": pair["pair"]}


def commit(store: Store, claimed: dict[str, Any], run: str) -> str | None:
    """After the run opens: tag it with its check and pair key, and record it in the check with the order the
    pair's runs came in. Returns a warning when this run is its pair's first and the plan has the opposite arm
    first: the run is kept all the same."""
    chk, arm, key = claimed["check"], claimed["arm"], claimed["pair"]
    store.tag_run(run, ext={CHECK_EXT: dict(claimed), PAIR_EXT: {"pair": key, "arm": arm, "experiment": chk}})
    doc = store.run_doc(run)
    with store.lock():
        check = load(store, chk)
        pair = pair_of(check, key)
        if pair["runs"][arm] not in (None, run):
            raise StoreError(f"pair {key} of {chk} got its {arm} run {pair['runs'][arm]} while {run} was opened; "
                             f"record {run} again with --pair, or leave it out")
        pair["runs"][arm] = run
        ran = pair.get("ran") or [a for a in ARMS if pair["runs"][a] and a != arm]  # checks before `ran`
        if arm not in ran:
            ran.append(arm)
        pair["ran"] = ran
        pair["slate"] = ((doc.get("run") or {}).get("slate") or {}).get("id") or pair["slate"]
        if pair["task"] is None:  # --n: the first run's task is the pair's
            pair["task"] = (doc.get("run") or {}).get("task")
        save(store, check)
    order = pair["order"]
    if ran == [arm] and order[0] != arm:
        return (f"pair {key} of {chk} is planned {' then '.join(order)}, which balances which arm runs first; "
                f"this {arm} run comes first, out of that order. It is recorded and counts, and status marks the "
                f"pair")
    return None


# ---------------------------------------------------------------- reading a check back
def docs(store: Store, check: dict[str, Any]) -> tuple[list[dict], list[dict], list[str]]:
    """(finished base runs, finished candidate runs, open run ids) of the check's pairs."""
    base, cand, still_open = [], [], []
    for p in check["pairs"]:
        for arm in ARMS:
            run = p["runs"][arm]
            if not run:
                continue
            try:
                doc = store.run_doc(run)
            except NotFound:
                continue
            if not R.is_finished(doc):
                still_open.append(run)
                continue
            (base if arm == "base" else cand).append(doc)
    return base, cand, still_open


def out_of_order(check: dict[str, Any]) -> list[dict[str, Any]]:
    """The pairs whose runs came in another order than planned: [{"pair", "planned", "ran"}]. A check written
    before `ran` was kept has no such mark."""
    return [{"pair": p["pair"], "planned": list(p["order"]), "ran": list(p["ran"])} for p in check["pairs"]
            if p.get("ran") and list(p["ran"]) != list(p["order"])[:len(p["ran"])]]


def preferences(check: dict[str, Any], docs_: list[dict]) -> dict[str, int]:
    """Blinded referee preferences over the check's pairs: the latest per slate, by arm. They are counted
    beside the comparison and never enter its verdict."""
    arm_of = {p["runs"][a]: a for p in check["pairs"] for a in ARMS if p["runs"][a]}
    slates = {p["slate"] for p in check["pairs"] if p["slate"]}
    latest_by_slate: dict[str, dict] = {}
    for d in docs_:
        for pref in (d.get("run") or {}).get("preferences") or []:
            judge = pref.get("judge") or {}
            if not isinstance(pref, dict) or pref.get("slate") not in slates:
                continue
            if judge.get("kind") != "referee" or not judge.get("blinded"):
                continue
            seen = latest_by_slate.get(pref["slate"])
            if seen is None or str(pref.get("observed_at") or "") > str(seen.get("observed_at") or ""):
                latest_by_slate[pref["slate"]] = pref
    out = {"base": 0, "cand": 0, "tie": 0}
    for pref in latest_by_slate.values():
        won = pref.get("winner")
        out["tie" if won in (None, "tie") else arm_of.get(won, "tie")] += 1
    return out


def task_view(task: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if task is None else {k: task.get(k) for k in TASK_VIEW}


def next_view(check: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    """The next pair to run: its task, arm order, worktrees, `run record` lines and the referee step."""
    chk, key, task = check["check"], pair["pair"], pair["task"]
    start = (task or {}).get("base_commit") or "HEAD"
    worktrees = {arm: f"../{chk}-{key}-{arm}" for arm in ARMS}
    todo = [arm for arm in pair["order"] if not pair["runs"][arm]]
    commands = [f"git worktree add {worktrees[arm]} {start}" for arm in todo]
    notes = [f"{arm}: put the {arm} harness from {check[arm]['path']} in {worktrees[arm]}"
             for arm in todo if check[arm].get("path")]
    for i, arm in enumerate(todo):
        task_flags = " --type T --repo R" if task is None and i == 0 else ""
        harness = (f" --harness {shlex.quote(check[arm]['path'])}"  # the run records the arm's harness record
                   if check[arm].get("path") and check[arm].get("harness") else "")
        commands.append(f"loopmath run record --check {chk} --arm {arm} --workflow WORKFLOW --source designed"
                        f"{task_flags} --cwd {worktrees[arm]}{harness} --since START --verified tests=pass|fail")
    commands.append(f"loopmath outcome --slate {pair['slate'] or 'SLT'} --prefer RUN|tie --judge referee --blinded")
    if task is None:
        notes.append("the pair's first run sets its task from --type and --repo; the second takes it")
    return {"pair": key, "task": task_view(task), "order": list(pair["order"]), "todo": todo,
            "worktrees": worktrees, "commands": commands, "referee": REFEREE, "notes": notes}
