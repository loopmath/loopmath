"""A screen in the store: `screens/<scr>.json`, its arm trees under `screens/<scr>/trees/`.

`screen start` writes it with its pairs: a task each (from `--tasks`, or from the pair's first run with `--n`), a
fold-over row and a random arm order. `run record --screen SCR --arm A|B` (and `run start`) claims the earliest
pair without that arm, as `check-change` does: the run gets the pair's task and opens or joins the pair's slate
(`claim`), then is tagged and recorded in the screen (`commit`).
"""

from __future__ import annotations

import json
import random
import re
import shlex
from pathlib import Path
from typing import Any

from ..compare import PAIR_EXT
from ..compare.checks import TASK_FLAGS, task_view
from ..store import runs as R
from ..store.commands import UserError
from ..store.home import NotFound, Store, StoreError
from ..store.ids import new_id, now_iso
from ..store.lock import atomic_write_json
from . import ARMS, SCREEN_EXT, SCREEN_SCHEMA, design, edits as E

SCR_RE = re.compile(r"^scr_[0-9A-Za-z]+$")


def screens_dir(store: Store) -> Path:
    return Path(store.home) / "screens"


def screen_path(store: Store, scr: str) -> Path:
    if not isinstance(scr, str) or not SCR_RE.match(scr):
        raise UserError(f"a screen id is scr_ and letters or digits, not {scr!r}")
    return screens_dir(store) / f"{scr}.json"


def load(store: Store, scr: str) -> dict[str, Any]:
    path = screen_path(store, scr)
    if not path.is_file():
        raise NotFound(f"no screen {scr} in {store.home}; loopmath screen start makes one")
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StoreError(f"{path}: not JSON: {exc}") from None
    if not isinstance(rec, dict) or rec.get("schema") != SCREEN_SCHEMA:
        raise StoreError(f"{path}: not a {SCREEN_SCHEMA} record")
    return rec


def save(store: Store, rec: dict[str, Any]) -> Path:
    screens_dir(store).mkdir(parents=True, exist_ok=True)
    return atomic_write_json(screen_path(store, rec["screen"]), rec)


def latest(store: Store) -> str:
    found = sorted(p.stem for p in screens_dir(store).glob("scr_*.json")) if screens_dir(store).is_dir() else []
    if not found:
        raise NotFound(f"no screens in {store.home}; loopmath screen start makes one")
    return found[-1]


def names(rec: dict[str, Any]) -> list[str]:
    return [e["name"] for e in rec["edits"]]


def new_screen(store: Store, *, base: Path, edits: list[dict[str, Any]], tasks: list[dict[str, Any]] | None,
               n: int | None, rule: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Write a new screen with its pairs, each with a fold-over row and an arm order drawn from the seed."""
    scr = new_id("scr")
    count = len(tasks) if tasks is not None else int(n or 0)
    if count < 2:
        raise UserError("a screen needs at least 2 pairs: --tasks FILE.jsonl with 2 or more tasks, or --n N")
    seed = int(settings.get("seed") or 0)
    p = len(edits)
    even = count + count % 2  # an odd count drops the last row of the next even design
    rows = design.fold_over(p, even, seed=seed)[:count]
    how = design.plan(p, even) if count % 2 == 0 else {"kind": "balanced-random, one row dropped", "block": None,
                                                         "orthogonal": False}
    rng = random.Random(f"{scr}:{seed}")
    width = max(2, len(str(count)))
    pairs = []
    for i in range(count):
        task = None
        if tasks is not None:
            task = dict(tasks[i])
            task.setdefault("id", new_id("tsk"))
        order = list(ARMS)
        rng.shuffle(order)
        pairs.append({"pair": f"p{i + 1:0{width}d}", "task": task, "row": [int(x) for x in rows[i]], "order": order,
                      "slate": None, "runs": {arm: None for arm in ARMS}, "ran": []})
    rec = {"schema": SCREEN_SCHEMA, "screen": scr, "created_at": now_iso(),
           "base": {"path": str(Path(base).resolve()), "digest": E.digest(Path(base))},
           "edits": edits, "rule": rule, "settings": settings,
           "design": {"kind": how["kind"], "orthogonal": bool(how["orthogonal"]), "pairs": count, "edits": p},
           "pairs": pairs}
    store.ensure()
    save(store, rec)
    return rec


def check_unchanged(rec: dict[str, Any]) -> None:
    """The base and every edit hold what they held at `start`: a tree built now is the tree the screen plans."""
    base = Path(rec["base"]["path"])
    if not base.is_dir():
        raise NotFound(f"the screen's base {base} is gone")
    if E.digest(base) != rec["base"]["digest"]:
        raise UserError(f"the base {base} changed since screen start; restore it, or start a new screen")
    for e in rec["edits"]:
        path = Path(e["path"])
        if not path.is_dir() or E.digest(path) != e["digest"]:
            raise UserError(f"edit {e['name']} ({path}) changed since screen start; restore it, or start a new "
                            f"screen")


def tree(store: Store, rec: dict[str, Any], on: list[str]) -> Path:
    """The arm tree for the edits `on`, built once in the store."""
    return E.built(Path(rec["base"]["path"]), rec["edits"], on, screens_dir(store) / rec["screen"] / "trees")


def harness_digest(path: Path) -> str | None:
    try:
        from ..harness import record_from_tree
    except ImportError:  # pragma: no cover
        return None
    rec = record_from_tree(path)
    return rec.get("digest") if isinstance(rec, dict) else None


def pair_of(rec: dict[str, Any], key: str) -> dict[str, Any]:
    for p in rec["pairs"]:
        if p["pair"] == key:
            return p
    raise NotFound(f"screen {rec['screen']} has no pair {key} (pairs {rec['pairs'][0]['pair']} to "
                   f"{rec['pairs'][-1]['pair']})")


def next_pair(rec: dict[str, Any]) -> dict[str, Any] | None:
    return next((p for p in rec["pairs"] if not all(p["runs"].values())), None)


def on_in(rec: dict[str, Any], pair: dict[str, Any], arm: str) -> list[str]:
    a, b = design.arms(pair["row"], names(rec))
    return a if arm == "A" else b


# ---------------------------------------------------------------- recording into a screen
def claim(store: Store, args: Any) -> dict[str, Any]:
    """Before the run opens: choose the pair and point `args` at its task file and slate."""
    scr, arm = args.screen, getattr(args, "arm", None)
    if getattr(args, "check", None):
        raise UserError("--screen and --check each record the run into their own pair; give one of them")
    if arm not in ARMS:
        raise UserError(f"--screen {scr} needs --arm A or --arm B")
    if getattr(args, "slate", None) or getattr(args, "new_slate", False):
        raise UserError("--screen gives the run its pair's slate; drop --slate and --new-slate")
    rec = load(store, scr)
    if getattr(args, "pair", None):
        pair = pair_of(rec, args.pair)
        if pair["runs"][arm]:
            raise UserError(f"pair {pair['pair']} of {scr} already has its {arm} run {pair['runs'][arm]}")
    else:
        pair = next((p for p in rec["pairs"] if not p["runs"][arm]), None)
        if pair is None:
            raise UserError(f"every pair of {scr} has its {arm} run; start a screen with more tasks")
    if pair["task"] is not None:
        given = [flag for flag, attr in TASK_FLAGS if getattr(args, attr, None)]
        if given:
            raise UserError(f"pair {pair['pair']} of {scr} has its task; drop {', '.join(given)}")
        task_dir = screens_dir(store) / scr
        task_dir.mkdir(parents=True, exist_ok=True)
        args.task_file = str(atomic_write_json(task_dir / f"{pair['pair']}.task.json", pair["task"]))
    args.slate, args.new_slate = (pair["slate"], False) if pair["slate"] else (None, True)
    return {"screen": scr, "arm": arm, "pair": pair["pair"], "on": on_in(rec, pair, arm)}


def commit(store: Store, claimed: dict[str, Any], run: str) -> str | None:
    """After the run opens: tag it and record it in its pair. Returns a warning when the pair's planned second
    arm was recorded first (the run is kept)."""
    scr, arm, key = claimed["screen"], claimed["arm"], claimed["pair"]
    store.tag_run(run, ext={SCREEN_EXT: dict(claimed), PAIR_EXT: {"pair": key, "arm": arm, "experiment": scr}})
    doc = store.run_doc(run)
    with store.lock():
        rec = load(store, scr)
        pair = pair_of(rec, key)
        if pair["runs"][arm] not in (None, run):
            raise StoreError(f"pair {key} of {scr} got its {arm} run {pair['runs'][arm]} while {run} was opened")
        pair["runs"][arm] = run
        ran = pair.get("ran") or []
        if arm not in ran:
            ran.append(arm)
        pair["ran"] = ran
        pair["slate"] = ((doc.get("run") or {}).get("slate") or {}).get("id") or pair["slate"]
        if pair["task"] is None:
            pair["task"] = (doc.get("run") or {}).get("task")
        save(store, rec)
    if len(ran) == 1 and ran[0] != pair["order"][0]:
        return (f"pair {key} of {scr} is planned {' then '.join(pair['order'])}; its {arm} run came first. It is "
                f"recorded and counts")
    return None


# ---------------------------------------------------------------- reading a screen back
def finished(store: Store, rec: dict[str, Any]) -> tuple[list[tuple[dict, dict, dict]], list[str]]:
    """([(pair, doc A, doc B)] for the pairs with both runs finished, the open run ids)."""
    done, still_open = [], []
    for p in rec["pairs"]:
        docs = {}
        for arm in ARMS:
            run = p["runs"][arm]
            if not run:
                continue
            try:
                doc = store.run_doc(run)
            except NotFound:
                continue
            if R.is_finished(doc):
                docs[arm] = doc
            else:
                still_open.append(run)
        if len(docs) == 2:
            done.append((p, docs["A"], docs["B"]))
    return done, still_open


def next_view(store: Store, rec: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    """The next pair: its task, arm order, each arm's edits and built tree, and the `run record` lines."""
    check_unchanged(rec)
    scr, key, task = rec["screen"], pair["pair"], pair["task"]
    start = (task or {}).get("base_commit") or "HEAD"
    arms_view = {}
    for arm in ARMS:
        on = on_in(rec, pair, arm)
        path = tree(store, rec, on)
        arms_view[arm] = {"on": on, "tree": str(path), "harness": harness_digest(path),
                          "worktree": f"../{scr}-{key}-{arm}", "run": pair["runs"][arm]}
    todo = [arm for arm in pair["order"] if not pair["runs"][arm]]
    commands = []
    for i, arm in enumerate(todo):
        v = arms_view[arm]
        commands.append(f"git worktree add {v['worktree']} {start}")
        commands.append(f"cp -R {shlex.quote(v['tree'])}/. {v['worktree']}/")
        task_flags = " --type T --repo R" if task is None and i == 0 else ""
        commands.append(f"loopmath run record --screen {scr} --arm {arm} --workflow WORKFLOW --source designed"
                        f"{task_flags} --cwd {v['worktree']} --harness {shlex.quote(v['tree'])} --since START "
                        f"--verified tests=pass|fail")
    return {"pair": key, "task": task_view(task), "order": list(pair["order"]), "todo": todo, "row": pair["row"],
            "arms": arms_view, "commands": commands}
