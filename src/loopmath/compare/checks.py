"""A check: one paired A/B of two arms, kept at `checks/<chk>.json` in the store.

`loopmath check-change start` writes it with its pairs: a task each (from `--tasks`, or from the pair's first
run with `--n`) and a random arm order. `run record --check CHK --arm base|cand` (and `run start`) claims the
earliest pair without that arm (`claim`), so the run gets the pair's task and opens or joins the pair's slate,
then tags the run and records it in the check (`commit`). The planned arm order balances which arm runs first.
The pair keeps the order its runs were recorded in (`ran`), and its runs' start times say which arm started
first: when the planned second arm did, the pair is kept with a warning, and `status` marks it
(`out_of_order`). Two arms run at the same time, the planned first a little earlier, are in order whichever
finishes and is recorded first. `status` reads the pairs' runs back.

loopmath never creates the worktrees or starts the agents: `next_view` prints those steps, and the blinded
referee's, as the record-run skill describes them.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
import re
import shlex
from pathlib import Path
from typing import Any

from ..store import runs as R
from ..store.commands import UserError, parse_set
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


# ---------------------------------------------------------------- an arm's setting
def arm_set(items: list[str] | None, flag: str) -> dict[str, str] | None:
    """`--base-set` or `--cand-set`, as `run record --set` takes it: {piece: "harness:model:effort"}, or None."""
    if not items:
        return None
    try:
        given = parse_set(items)
    except UserError as exc:
        raise UserError(str(exc).replace("--set", flag, 1)) from None
    return {piece: f"{s.harness}:{s.model}:{s.effort}" for piece, s in given.items()}


def _fixed(arm: dict[str, Any]) -> dict[str, Any]:
    return parse_set([f"{piece}={spec}" for piece, spec in (arm.get("set") or {}).items()])


def _same(a: Any, b: Any) -> bool:
    """One harness, model and effort, model aliases (`sonnet-5`, `claude-sonnet-5`) counting as one."""
    from ..belief.forest import canonical_model_id

    return (a.harness.lower() == b.harness.lower() and canonical_model_id(a.model) == canonical_model_id(b.model)
            and a.effort.lower() == b.effort.lower())


def arm_settings(store: Store, args: Any) -> None:
    """`run start` or `run record` into an arm with a set: the run takes the arm's set. A `--set` naming one of
    the arm's pieces with another harness, model or effort is refused, and so are --config, --rec and --choice,
    which bring their own configuration. All before anything is written."""
    chk, arm = getattr(args, "check", None), getattr(args, "arm", None)
    if not chk or arm not in ARMS:
        return
    spec = load(store, chk)[arm].get("set")
    if not spec:
        return
    shown = " ".join(f"{piece}={s}" for piece, s in spec.items())
    for flag, attr in (("--config", "config"), ("--rec", "rec"), ("--choice", "choice")):
        if getattr(args, attr, None):
            raise UserError(f"{flag}: arm {arm} of {chk} runs {shown}, which fixes the run's configuration; give "
                            f"--workflow and leave the setting to the arm")
    fixed = _fixed({"set": spec})
    for piece, s in parse_set(getattr(args, "set", None) or []).items():
        if piece in fixed and not _same(s, fixed[piece]):
            raise UserError(f"--set {piece}={s.harness}:{s.model}:{s.effort}: arm {arm} of {chk} runs "
                            f"{piece}={spec[piece]}; drop --set, or record the run into an arm that runs it")
    args.set = [f"{piece}={s}" for piece, s in spec.items()] + [
        item for item in getattr(args, "set", None) or [] if item.split("=", 1)[0].strip() not in spec]


def arm_models(check: dict[str, Any], docs: list[dict[str, Any]]) -> tuple[dict[str, list[str]], list[dict]]:
    """Per arm, the models its finished runs ran (one spelling each); and each attempt on a piece of its arm's
    set that ran another harness, model or effort. Effort counts only when the arm names one (not `default`)
    and the attempt has one. A mismatch is shown, never dropped: the comparison still counts the run."""
    from ..belief.forest import canonical_model_id

    models: dict[str, set[str]] = {arm: set() for arm in ARMS}
    mismatch: list[dict[str, Any]] = []
    fixed = {arm: _fixed(check[arm]) for arm in ARMS}
    for doc in docs:
        tag = ((doc.get("run") or {}).get("ext") or {}).get(CHECK_EXT) or {}
        arm = tag.get("arm")
        if arm not in ARMS:
            continue
        for a in doc.get("attempts") or []:
            if not isinstance(a, dict):
                continue
            m = a.get("model")
            ran_model = (m.get("raw") or m.get("id")) if isinstance(m, dict) else m
            if ran_model:
                models[arm].add(canonical_model_id(str(ran_model)))
            piece = a.get("node") or a.get("piece")  # a stored attempt names its piece as `node` (store/runs.py:281)
            want = fixed[arm].get(piece)
            if want is None or not ran_model:
                continue
            effort = a.get("effort") if want.effort.lower() != "default" and a.get("effort") else want.effort
            ran = type(want)(harness=a.get("harness") or want.harness, model=str(ran_model), effort=str(effort))
            if not _same(ran, want):
                mismatch.append({"pair": tag.get("pair"), "arm": arm, "piece": piece,
                                 "declared": check[arm]["set"][piece],
                                 "ran": f"{ran.harness}:{ran.model}:{ran.effort}"})
    return {arm: sorted(v) for arm, v in models.items()}, sorted(mismatch, key=lambda x: (x["pair"] or "", x["arm"]))


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
        raise UserError(f"--check {chk} needs --arm base or --arm cand"
                        + ("; A and B are a screen's arms: record them with --screen SCR" if arm in ("A", "B") else ""))
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
    swapped = out_of_order({"pairs": [pair]}, run_starts(store, {"pairs": [pair]}))
    if swapped:
        first = swapped[0]["ran"][0]
        how = "started first" if swapped[0]["by"] == "start" else "was recorded first, and a start time is missing"
        return (f"pair {key} of {chk} is planned {' then '.join(pair['order'])}, which balances which arm runs "
                f"first; its {first} run {how}, out of that order. Both runs are recorded and count, and status "
                f"marks the pair")
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


def _start(ts: Any) -> float | None:
    """A run's start time in seconds, or None when it is not a time."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def run_starts(store: Store, check: dict[str, Any]) -> dict[str, Any]:
    """Each recorded run's start time (`run.started_at`), finished or open, by run id."""
    out: dict[str, Any] = {}
    for p in check["pairs"]:
        for arm in ARMS:
            run = p["runs"][arm]
            if run:
                try:
                    out[run] = (store.run_doc(run).get("run") or {}).get("started_at")
                except NotFound:
                    continue
    return out


def out_of_order(check: dict[str, Any], starts: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The pairs whose planned second arm started first: [{"pair", "planned", "ran", "by"}], `ran` the arms in
    the order their runs started (`by` "start"; a tie keeps the plan). A pair is judged once both its runs are
    recorded. When a start time is missing (or no `starts` is given), the order the runs were recorded in stands
    in (`by` "record"), as before 0.3.1. A check written before `ran` was kept is judged on start times only."""
    out = []
    for p in check["pairs"]:
        if not all(p["runs"][a] for a in ARMS):
            continue
        plan = list(p["order"])
        times = {a: _start((starts or {}).get(p["runs"][a])) for a in ARMS}
        if None not in times.values():
            began = sorted(plan, key=lambda a: (times[a], plan.index(a)))
            if began != plan:
                out.append({"pair": p["pair"], "planned": plan, "ran": began, "by": "start"})
        elif p.get("ran") and list(p["ran"]) != plan:
            out.append({"pair": p["pair"], "planned": plan, "ran": list(p["ran"]), "by": "record"})
    return out


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
        sets = "".join(f" --set {shlex.quote(f'{piece}={spec}')}"  # the arm's set
                       for piece, spec in (check[arm].get("set") or {}).items())
        commands.append(f"loopmath run record --check {chk} --arm {arm} --workflow WORKFLOW{sets} --source designed"
                        f"{task_flags} --cwd {worktrees[arm]}{harness} --since START --verified tests=pass|fail")
    commands.append(f"loopmath outcome --slate {pair['slate'] or 'SLT'} --prefer RUN|tie --judge referee --blinded")
    if task is None:
        notes.append("the pair's first run sets its task from --type and --repo; the second takes it")
    return {"pair": key, "task": task_view(task), "order": list(pair["order"]), "todo": todo,
            "worktrees": worktrees, "commands": commands, "referee": REFEREE, "notes": notes}
