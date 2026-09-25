"""Store layout and the Store class (design/0.1/03-interfaces.md, section 4).

```
$LOOPMATH_HOME/            default ~/.loopmath
  config.toml
  runs/<run>.ocp.json      one OCP v0.3 document per run; the source of truth
  runs/index.jsonl         one row per state change; the last row per run wins
  signals/<run>.jsonl      signals recorded before `finish` folds them in
  receipts/<rct>.json
  recs/<rec>.json
  fits/<fit>/ and fits/latest
  workflows/*.toml, priors/, views/, share/
  lock
```

Writers hold `store_lock` for every read-modify-write; readers never lock.
Other lanes use this class: lane 3 (`import_run`), lane 5 (`finished_docs`,
`latest_fit`), lane 6 (`config`, `spend`, `write_rec`), lane 8 and 12 (`runs`,
`run_doc`).
"""

from __future__ import annotations

import copy

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..output import home as default_home
from ..types import AcceptanceRule, Configuration, Signal, Task
from . import runs as R
from .config import Config
from .ids import new_id, now_iso, parse_ts
from .lock import (append_line, atomic_write_json, read_json, read_jsonl, store_lock)

SUBDIRS = ("runs", "signals", "receipts", "recs", "fits", "workflows", "priors", "views")


class StoreError(Exception):
    """A user error against the store (exit code 1)."""


class NotFound(StoreError):
    """A run, attempt, slate or record that does not exist (exit code 2)."""


class Conflict(StoreError):
    """The run file changed between read and write."""


class EndBeforeStart(StoreError):
    """`--ended-at` before the attempt's `started_at` (exit code 2)."""


class Store:
    def __init__(self, home: Path | str | None = None):
        self.home = Path(home).expanduser() if home else default_home()

    # ------------------------------------------------------------ layout
    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def index_path(self) -> Path:
        return self.runs_dir / "index.jsonl"

    @property
    def fits_dir(self) -> Path:
        return self.home / "fits"

    @property
    def views_dir(self) -> Path:
        return self.home / "views"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"

    def run_path(self, run: str) -> Path:
        self._check_id(run)
        return self.runs_dir / f"{run}.ocp.json"

    def signals_path(self, run: str) -> Path:
        self._check_id(run)
        return self.home / "signals" / f"{run}.jsonl"

    def receipt_path(self, rct: str) -> Path:
        self._check_id(rct)
        return self.home / "receipts" / f"{rct}.json"

    def rec_path(self, rec: str) -> Path:
        self._check_id(rec)
        return self.home / "recs" / f"{rec}.json"

    @staticmethod
    def _check_id(value: str) -> None:
        if not isinstance(value, str) or not R.valid_run_id(value):
            raise StoreError(f"not a valid id: {value!r}")

    def ensure(self) -> None:
        for sub in SUBDIRS:
            (self.home / sub).mkdir(parents=True, exist_ok=True)

    def lock(self, timeout: float | None = None):
        self.home.mkdir(parents=True, exist_ok=True)
        return store_lock(self.home, timeout)

    # ------------------------------------------------------------ reading
    def exists(self, run: str) -> bool:
        return self.run_path(run).is_file()

    def _read(self, run: str) -> dict[str, Any]:
        path = self.run_path(run)
        try:
            doc = read_json(path)
        except FileNotFoundError:
            raise NotFound(f"no run {run} in {self.home}") from None
        if not isinstance(doc, dict):
            raise StoreError(f"{path} does not hold a JSON object")
        return doc

    def pending_signals(self, run: str) -> list[dict[str, Any]]:
        return read_jsonl(self.signals_path(run))

    def run_doc(self, run: str) -> dict[str, Any]:
        """The run file, with signals still pending before `finish` merged into `run.signals`."""
        doc = self._read(run)
        pending = self.pending_signals(run)
        if pending:
            R.merge_signals(doc, pending)
        return doc

    def index_rows(self) -> dict[str, dict[str, Any]]:
        return R.latest_rows(read_jsonl(self.index_path))

    def runs(self, **filters: Any) -> Iterator[dict[str, Any]]:
        """Index rows, oldest first. Filters: any index field by equality, plus `since` (ISO time or datetime)."""
        since = filters.pop("since", None)
        since_dt = parse_ts(since) if isinstance(since, str) else since
        rows = sorted(self.index_rows().values(), key=lambda r: str(r.get("started_at") or ""))
        for row in rows:
            if any(v is not None and row.get(k) != v for k, v in filters.items()):
                continue
            if since_dt is not None:
                started = parse_ts(row.get("started_at"))
                if started is None or started < since_dt:
                    continue
            yield row

    def finished_docs(self) -> Iterator[dict[str, Any]]:
        """Every finished run document (for `fit`), read from the run files themselves so a missing or
        stale index never drops a run; open runs are never yielded. Unreadable files are skipped."""
        folder = self.home / "runs"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.ocp.json")):
            try:
                doc = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and R.is_finished(doc):
                yield doc

    def latest_fit(self) -> Path | None:
        link = self.fits_dir / "latest"
        if not link.exists():
            return None
        target = link.resolve()
        return target if target.is_dir() else None

    def config(self) -> Config:
        return Config.load(self.config_path)

    def spend(self, period: str) -> float:
        from .budget import spend

        return spend(self, period)["usd"]

    # ------------------------------------------------------------ writing: helpers
    def _write(self, doc: dict[str, Any]) -> None:
        run = doc["run"]["id"]
        R.bump(doc)
        atomic_write_json(self.run_path(run), doc)

    def _index(self, doc: dict[str, Any]) -> None:
        append_line(self.index_path, R.index_row(doc))

    def _update(self, run: str, fn) -> Any:
        """Read, apply `fn(doc)`, write; under the lock."""
        with self.lock():
            doc = self._read(run)
            result = fn(doc)
            self._write(doc)
            return result

    # ------------------------------------------------------------ writing: spec 03 section 4
    def new_run(self, task: Task, config: Configuration | dict[str, Any], *, source: str, rec: str | None,
                slate: str | None, rule: AcceptanceRule, base_commit: str | None,
                started_at: str | None = None, run_id: str | None = None) -> str:
        """Open a run: the OCP v0.3 skeleton, an index row, and the slate joined in every member's file.

        `config` is a spec 03 Configuration or an OCP 2.2 configuration dict.
        """
        cfg_ocp = R.configuration_to_ocp(config) if isinstance(config, Configuration) else dict(config)
        if not cfg_ocp.get("id") or not R.pieces_of(cfg_ocp):
            raise StoreError("the configuration has no id or no pieces")
        settings = cfg_ocp.get("settings") or {}
        missing = [p["id"] for p in R.pieces_of(cfg_ocp) if p["id"] not in settings]
        if missing:
            raise StoreError(f"no setting for piece(s) {', '.join(missing)}")
        if base_commit:
            task = Task.from_dict({**task.to_dict(), "base_commit": base_commit})
        run = run_id or new_id("run")
        started = started_at or now_iso()
        self.ensure()
        with self.lock():
            slate_obj = None
            members_docs: list[dict[str, Any]] = []
            if slate:
                members_docs = [self._read(r["run"]) for r in self.runs(slate=slate)]
                for other in members_docs:
                    other_task = (other.get("run") or {}).get("task") or {}
                    if other_task.get("id") != task.id:
                        raise StoreError(f"slate {slate} is for task {other_task.get('id')!r}, not {task.id!r}: a slate holds "
                                     "one task and task flags make a new one, so start every member from the same "
                                     "--task-file")
                    if (other_task.get("base_commit") or None) != (task.base_commit or None):
                        raise StoreError(f"slate {slate} starts from base commit {other_task.get('base_commit')!r}, "
                                         f"not {task.base_commit!r}")
                members = [(d.get("run") or {}).get("id") for d in members_docs] + [run]
                slate_obj = dict(((members_docs[0].get("run") or {}).get("slate") or {}) if members_docs else {})
                slate_obj.update({"id": slate, "members": members})
                if task.base_commit:
                    slate_obj["base_commit"] = task.base_commit
            doc = R.new_doc(run_id=run, task=task, cfg_ocp=cfg_ocp, source=source, rec=rec, rule=rule,
                            slate=slate_obj, started_at=started)
            if self.run_path(run).exists():
                raise StoreError(f"run {run} already exists")
            self._write(doc)
            self._index(doc)
            for other in members_docs:
                other["run"]["slate"] = dict(slate_obj)
                self._write(other)
        return run

    def add_attempt(self, run: str, attempt: dict[str, Any]) -> str:
        """`attempt`: piece, harness, model, effort, cwd, session, round, cause, started_at, session_from."""
        att = attempt.get("id") or new_id("att")

        def apply(doc: dict[str, Any]) -> str:
            self._require_open(doc)
            piece = attempt.get("piece")
            if R.node(doc, piece) is None:
                pieces = ", ".join(n.get("id") for n in doc.get("nodes") or [])
                raise StoreError(f"run {run} has no piece {piece!r} (pieces: {pieces})")
            if R.attempt(doc, att) is not None:
                raise StoreError(f"attempt {att} already exists")
            started = attempt.get("started_at") or now_iso()
            rec = R.attempt_record(doc, att_id=att, piece=piece, harness=attempt["harness"], model=attempt["model"],
                                   effort=attempt.get("effort"), cwd=attempt.get("cwd"), session=attempt.get("session"),
                                   round_=attempt.get("round") or 1, cause=attempt.get("cause") or "initial",
                                   started_at=started, session_from=attempt.get("session_from"))
            doc.setdefault("attempts", []).append(rec)
            R.node(doc, piece)["state"] = "working"
            R.add_event(doc, "attempt_started", at=started, node=piece, attempt=att)
            return att

        return self._update(run, apply)

    def end_attempt(self, run: str, att: str, status: str, ended_at: str | None) -> str:
        """Settle attempt `att`; returns the recorded end time (now when `ended_at` is None)."""

        def apply(doc: dict[str, Any]) -> str:
            self._require_open(doc)
            rec = R.attempt(doc, att)
            if rec is None:
                raise NotFound(f"run {run} has no attempt {att}")
            started, end = parse_ts(rec.get("started_at")), parse_ts(ended_at)
            if started is not None and end is not None and end < started:
                raise EndBeforeStart(f"--ended-at {ended_at} is before attempt {att} started ({rec['started_at']}); "
                                     "give its end, at or after the start")
            ended = ended_at or now_iso()
            rec["status"] = status
            rec["ended_at"] = ended
            rec["outcome"] = {"result": status, "evidence": "reported"}
            n = R.node(doc, rec.get("node"))
            if n is not None:
                n["state"] = status
            R.add_event(doc, "attempt_settled", at=ended, node=rec.get("node"), attempt=att, detail=status)
            return ended

        return self._update(run, apply)

    def add_artifact(self, run: str, artifact: dict[str, Any]) -> str:
        """`artifact`: kind, path, by (attempt id), read_by (attempt ids), supersedes."""
        art = artifact.get("id") or new_id("art")

        def apply(doc: dict[str, Any]) -> str:
            by = artifact["by"]
            known = {a.get("id") for a in doc.get("attempts") or [] if isinstance(a, dict)}
            for a in [by] + list(artifact.get("read_by") or []):
                if a not in known:
                    raise NotFound(f"run {run} has no attempt {a}")
            sup = artifact.get("supersedes")
            if sup and not any(isinstance(x, dict) and x.get("id") == sup for x in doc.get("artifacts") or []):
                raise NotFound(f"run {run} has no artifact {sup}")
            at = now_iso()
            rec = R.artifact_record(doc, art_id=art, kind=artifact["kind"], path=artifact["path"], by=by,
                                    read_by=list(artifact.get("read_by") or []), supersedes=sup, at=at)
            doc.setdefault("artifacts", []).append(rec)
            R.add_event(doc, "artifact_written", at=at, attempt=by, detail=artifact["kind"])
            return art

        return self._update(run, apply)

    def add_signal(self, run: str, signal: Signal) -> str:
        """Record a signal. Open run: appended to `signals/<run>.jsonl`. Finished run: a late signal, written
        into the run file with a `signal_observed` event, and the run's receipts flagged `rescore: true`.

        Returns "pending" or "late".
        """
        record = R.signal_record(signal)
        with self.lock():
            doc = self._read(run)
            if not R.is_finished(doc):
                append_line(self.signals_path(run), record)
                return "pending"
            R.merge_signals(doc, [record])
            R.add_event(doc, "signal_observed", at=signal.observed_at or None, detail=f"{signal.kind}:{signal.name}")
            self._write(doc)
            self.flag_rescore(run)
            return "late"

    def add_preference(self, slate: str, winner: str | None, judge: str, blinded: bool,
                       *, model: str | None = None, observed_at: str | None = None, tier: str = "reported") -> str:
        """Copy one preference into every member's file (OCP 2.7). `winner` None means a tie."""
        pref_id = new_id("prf")
        with self.lock():
            rows = list(self.runs(slate=slate))
            if not rows:
                raise NotFound(f"no runs in slate {slate}")
            members = [r["run"] for r in rows]
            if winner not in (None, "tie") and winner not in members:
                raise StoreError(f"{winner} is not a member of slate {slate} ({', '.join(members)})")
            judge_rec: dict[str, Any] = {"kind": judge, "blinded": bool(blinded)}
            if model:
                judge_rec["model"] = model
            pref = {"id": pref_id, "slate": slate, "winner": winner or "tie", "members": members,
                    "judge": judge_rec, "observed_at": observed_at or now_iso(), "tier": tier}
            for m in members:
                doc = self._read(m)
                R.add_preference(doc, pref)
                if blinded and isinstance(doc["run"].get("slate"), dict):
                    doc["run"]["slate"]["blinded"] = True
                R.add_event(doc, "signal_observed", at=pref["observed_at"], detail=f"preference:{pref['winner']}")
                self._write(doc)
        return pref_id

    def finish(self, run: str, doc: dict[str, Any], *, expect_rev: int | None = None,
               receipt: dict[str, Any] | None = None) -> None:
        """Write the validated, settled `doc` as finished, drop the pending signal file, index row, and the
        completed `receipt` under the same lock, so a late signal always finds the receipt it must flag.

        `doc` must already hold every pending signal: the caller folds them in before it computes
        evidence and the receipt. A signal appended since then (an open-run signal does not bump the
        rev) raises Conflict, as does a changed rev with `expect_rev`, so the caller recomputes.
        """
        with self.lock():
            current = self._read(run)
            if R.is_finished(current):
                raise StoreError(f"run {run} is already finished")
            if expect_rev is not None and R.rev(current) != expect_rev:
                raise Conflict(f"run {run} changed while it was being finished")
            have = {s.get("id") for s in (doc.get("run") or {}).get("signals") or [] if isinstance(s, dict)}
            if any(s.get("id") not in have for s in self.pending_signals(run)):
                raise Conflict(f"a signal for run {run} arrived while it was being finished")
            doc.setdefault("run", {}).setdefault("ended_at", now_iso())
            if not any(e.get("type") == "run_finished" for e in doc.get("events") or [] if isinstance(e, dict)):
                R.add_event(doc, "run_finished", at=doc["run"]["ended_at"])
            R.set_store_ext(doc, state=R.FINISHED, rev=R.rev(current))
            self._write(doc)
            self._index(doc)
            try:
                os.unlink(self.signals_path(run))
            except FileNotFoundError:
                pass
            if receipt is not None:
                self.write_receipt(receipt)

    def import_run(self, doc: dict[str, Any], *, finished: bool = True, check: bool = True) -> str:
        """Store a whole OCP v0.3 document (strict check, atomic write, index row under the lock).

        A document with the same run id replaces the stored one. `finished=True` marks it finished as is
        (no log matching; that is `run import --finish`). A document loopmath already finished (its store
        ext says so, for example one copied from another home) stays finished either way.
        """
        from .finish import validate

        run = (doc.get("run") or {}).get("id")
        if not isinstance(run, str) or not R.valid_run_id(run):
            raise StoreError(f"run.id {run!r} is not usable as a file name ([A-Za-z0-9._-], at most 200)")
        if check:
            result = validate(doc)
            if result["errors"]:
                raise ValidationFailed(result)
        doc = copy.deepcopy(doc)
        self.ensure()
        with self.lock():
            if finished or R.store_ext(doc).get("state") == R.FINISHED:
                doc.setdefault("run", {})
                if not doc["run"].get("ended_at"):
                    ends = [a.get("ended_at") for a in doc.get("attempts") or [] if isinstance(a, dict) and a.get("ended_at")]
                    doc["run"]["ended_at"] = max(ends, key=lambda t: parse_ts(t) or datetime.min.astimezone()) if ends else now_iso()
                if not any(isinstance(e, dict) and e.get("type") == "run_finished" for e in doc.get("events") or []):
                    R.add_event(doc, "run_finished", at=doc["run"]["ended_at"])
                R.set_store_ext(doc, state=R.FINISHED)
            else:
                R.set_store_ext(doc, state=R.OPEN)
            self._write(doc)
            self._index(doc)
        return run

    # ------------------------------------------------------------ receipts, recs, re-scoring
    def write_receipt(self, receipt: dict[str, Any]) -> Path:
        path = self.receipt_path(receipt["id"])
        atomic_write_json(path, receipt)
        return path

    def receipts(self) -> Iterator[dict[str, Any]]:
        folder = self.home / "receipts"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.json")):
            try:
                obj = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(obj, dict):
                yield obj

    def receipts_for(self, run: str) -> list[dict[str, Any]]:
        return [r for r in self.receipts() if r.get("run") == run]

    def flag_rescore(self, run: str) -> int:
        """Mark every receipt of `run` for re-scoring. Returns how many were flagged."""
        n = 0
        with self.lock():
            for r in self.receipts_for(run):
                if not r.get("rescore"):
                    r["rescore"] = True
                    self.write_receipt(r)
                    n += 1
        return n

    def rec(self, rec: str) -> dict[str, Any] | None:
        try:
            obj = read_json(self.rec_path(rec))
        except (FileNotFoundError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None

    def write_rec(self, rec: str, payload: dict[str, Any]) -> Path:
        return atomic_write_json(self.rec_path(rec), payload)

    def recs(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """(rec id, payload), newest file first."""
        folder = self.home / "recs"
        if not folder.is_dir():
            return
        paths = sorted(folder.glob("rec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                obj = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(obj, dict):
                yield path.stem, obj

    # ------------------------------------------------------------ index maintenance
    def reindex(self) -> dict[str, Any]:
        """Rewrite index.jsonl from the run files, one row per run."""
        from .lock import atomic_write_text
        import json

        rows: list[dict[str, Any]] = []
        bad: list[str] = []
        with self.lock():
            for path in sorted(self.runs_dir.glob("*.ocp.json")) if self.runs_dir.is_dir() else []:
                try:
                    doc = read_json(path)
                    rows.append(R.index_row(doc))
                except (OSError, ValueError, AttributeError):
                    bad.append(path.name)
            rows.sort(key=lambda r: str(r.get("started_at") or ""))
            text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
            atomic_write_text(self.index_path, text)
        return {"runs": len(rows), "unreadable": bad}

    # ------------------------------------------------------------ internals
    @staticmethod
    def _require_open(doc: dict[str, Any]) -> None:
        if R.is_finished(doc):
            raise StoreError(f"run {doc['run']['id']} is finished; its attempts can no longer change "
                             "(a commit made later can still be recorded with run artifact, D17)")


class ValidationFailed(StoreError):
    def __init__(self, result: dict[str, Any]):
        self.result = result
        first = "; ".join(f"{e.get('code', '')} {e.get('message', '')}".strip() for e in result["errors"][:3])
        super().__init__(f"OCP v0.3 check failed: {first}")
