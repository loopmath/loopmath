"""Adapter for bb's SQLite orchestration store.

One ``threads`` row becomes one bb node.  ``parent_thread_id`` and a
``source_thread_id`` whose ``origin_kind`` is ``fork`` become ``spawn`` edges;
both relations are explicit self-references in the store and therefore carry
the ``verified`` evidence tier.

bb delegates execution to Claude Code or Codex. Event rows identify that
vendor session with ``events.provider_thread_id`` in current stores and with
``data.providerThreadId`` in legacy rows. The first-class column wins on each
event, the JSON member is a fallback, and a launch edge is emitted only when
all populated events for a thread resolve to one UUID and the host independently
parses exactly one vendor session with that correlate. That combined evidence
supports the distinct vendor node and a heuristic correspondence edge; it does
not establish causal launch evidence.

bb's running token total is never used. Joined vendor usage and pricing arrive
through the adapter host-service boundary. This avoids treating bb's merged
cache counter as either a cache read or a cache write and avoids deriving cost
from a running snapshot.

The adapter reads only structural metadata from bb: identifiers, providers,
relations, timestamps, workspace paths, model/effort overrides, and the one
vendor UUID JSON member.  It never selects message content or event payloads.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import sys as _sys
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType as _ModuleType
from typing import Any, Iterable, Iterator

from .base import Adapter, Selection, Usage
from .bb_support import (
    SUPPORTED_PROVIDERS,
    _attempt,
    _attempt_id,
    _bb_join_issue,
    _bound_ms,
    _connect_read_only,
    _counts,
    _iso_from_ms,
    _normalized_uuid,
    _read_store,
    _RFC3339_RE,
    _thread_node_id,
    _UUID_RE,
    _workspace_group,
)
from .registry import register


DEFAULT_STORE = Path.home() / ".bb" / "bb.db"
EXT_NAMESPACE = "dev.dagr.adapter.bb"


class _BbModule(_ModuleType):
    def __setattr__(
        self,
        name: str,
        value: object,
        _support=_sys.modules[f"{__package__}.bb_support"],
        _bridged=frozenset(
            {
                "Any",
                "Iterable",
                "Path",
                "SUPPORTED_PROVIDERS",
                "_RFC3339_RE",
                "_UUID_RE",
                "_attempt_id",
                "_connect_read_only",
                "_normalized_uuid",
                "datetime",
                "defaultdict",
                "hashlib",
                "sqlite3",
                "timezone",
            }
        ),
    ) -> None:
        super().__setattr__(name, value)
        if name in _bridged:
            setattr(_support, name, value)


_sys.modules[__name__].__class__ = _BbModule
del _BbModule, _ModuleType, _sys


@register
class BbAdapter(Adapter):
    """Convert bb SQLite metadata and mechanically joined vendor sessions."""

    name = "bb"

    def discover(self) -> tuple[Path, ...]:
        return (DEFAULT_STORE,) if DEFAULT_STORE.is_file() else ()

    def fixture_selection(
        self, fixture_dir: Path, /
    ) -> AbstractContextManager[Selection]:
        """Materialize the synthetic SQL fixture for one gate invocation."""

        @contextmanager
        def materialized() -> Iterator[Selection]:
            fixture_sql = fixture_dir.expanduser().resolve() / "bb.sql"
            if not fixture_sql.is_file():
                raise FileNotFoundError(f"bb fixture SQL does not exist: {fixture_sql}")
            with TemporaryDirectory(prefix="loopmath-bb-fixture-") as temp_dir:
                db_path = Path(temp_dir) / "bb.db"
                with sqlite3.connect(db_path) as connection:
                    connection.executescript(fixture_sql.read_text(encoding="utf-8"))
                yield Selection(stores=(db_path,))

        return materialized()

    def sessions(self) -> tuple[dict[str, Any], ...]:
        descriptors: list[dict[str, Any]] = []
        for store in self.discover():
            threads, _queued = _read_store(store)
            for thread in threads:
                descriptors.append(
                    {
                        "id": thread["id"],
                        "provider": thread["provider"],
                        "workspace": thread["workspace"],
                        "created_at": _iso_from_ms(thread["created_at_ms"]),
                    }
                )
        return tuple(sorted(descriptors, key=lambda item: (item["created_at"], item["id"])))

    def emit(self, selection: Selection) -> dict[str, Any]:
        stores = selection.stores or self.discover()
        unique_stores: list[Path] = []
        seen_stores: set[Path] = set()
        for raw_store in stores:
            store = raw_store.expanduser().resolve()
            if store not in seen_stores:
                seen_stores.add(store)
                unique_stores.append(store)

        all_threads: list[dict[str, Any]] = []
        queued: list[tuple[str, str | None]] = []
        for store in sorted(unique_stores):
            store_threads, store_queued = _read_store(store)
            all_threads.extend(store_threads)
            queued.extend(store_queued)

        duplicate_ids = sorted(
            thread_id
            for thread_id, count in _counts(thread["id"] for thread in all_threads).items()
            if count > 1
        )
        if duplicate_ids:
            raise ValueError(
                "bb thread ids are not unique across the selected stores; "
                "emit one store at a time"
            )

        since_ms = _bound_ms(selection.since, "--since")
        until_ms = _bound_ms(selection.until, "--until")
        if since_ms is not None and until_ms is not None and since_ms > until_ms:
            raise ValueError("bb --since must not be later than --until")
        if selection.limit is not None and (
            isinstance(selection.limit, bool) or selection.limit < 1
        ):
            raise ValueError("bb limit must be a positive integer")

        wanted_ids = set(selection.session_ids)
        wanted_workspaces = set(selection.workspaces)
        selected = [
            thread
            for thread in all_threads
            if (not wanted_ids or thread["id"] in wanted_ids)
            and (not wanted_workspaces or thread["workspace"] in wanted_workspaces)
            and (since_ms is None or thread["created_at_ms"] >= since_ms)
            and (until_ms is None or thread["created_at_ms"] <= until_ms)
        ]
        selected.sort(key=lambda thread: (thread["created_at_ms"], thread["id"]))
        if selection.limit is not None:
            selected = selected[: selection.limit]

        selected_ids = {thread["id"] for thread in selected}
        vendor_owners: dict[tuple[str, str], list[str]] = defaultdict(list)
        for thread in all_threads:
            for session_id in thread["vendor_session_ids"]:
                vendor_owners[(thread["provider"], session_id)].append(thread["id"])

        groups: dict[str, dict[str, str]] = {}
        nodes: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        def add_attempt_events(attempt: dict[str, Any]) -> None:
            if "started_at" in attempt:
                events.append(
                    {
                        "at": attempt["started_at"],
                        "type": "attempt_started",
                        "node": attempt["node"],
                        "attempt": attempt["id"],
                    }
                )
            if "ended_at" in attempt:
                events.append(
                    {
                        "at": attempt["ended_at"],
                        "type": "attempt_settled",
                        "node": attempt["node"],
                        "attempt": attempt["id"],
                    }
                )

        for thread in selected:
            node_id = _thread_node_id(thread["id"])
            workspace = thread["workspace"]
            node: dict[str, Any] = {
                "id": node_id,
                "kind": "ops",
                "title": f"bb thread ({thread['provider']})",
                "state": "settled_unverified",
                "labels": {"harness": "bb", "provider": thread["provider"]},
            }
            if workspace:
                group_id = _workspace_group(workspace)
                groups[group_id] = {"id": group_id, "title": "bb workspace"}
                node["group"] = group_id
            nodes.append(node)
            attempt = _attempt(
                node_id,
                actor="bb",
                harness="bb",
                started_at=_iso_from_ms(thread["created_at_ms"]),
                wall_s=(thread["updated_at_ms"] - thread["created_at_ms"]) / 1000.0,
                model=thread["model"],
                model_tier="reported" if thread["model"] else None,
                effort=thread["effort"],
            )
            attempts.append(attempt)
            add_attempt_events(attempt)

        filtered_relations = 0
        for thread in selected:
            target = _thread_node_id(thread["id"])
            relations = []
            if thread["parent"]:
                relations.append(
                    (
                        thread["parent"],
                        "bb current threads.parent_thread_id self-reference",
                        None,
                    )
                )
            if thread["source"] and thread["origin_kind"] == "fork":
                relations.append(
                    (
                        thread["source"],
                        "bb threads.source_thread_id self-reference with origin_kind=fork",
                        "fork",
                    )
                )
            for source_id, evidence, native_relation in relations:
                if source_id not in selected_ids:
                    filtered_relations += 1
                    continue
                edge = {
                    "from": _thread_node_id(source_id),
                    "to": target,
                    "kind": "spawn",
                    "tier": "verified",
                    "from_attempt": _attempt_id(_thread_node_id(source_id)),
                    "to_attempt": _attempt_id(target),
                    "evidence": evidence,
                }
                if native_relation is not None:
                    edge["ext"] = {
                        EXT_NAMESPACE: {"native_relation": native_relation}
                    }
                edges.append(edge)

        joined = 0
        unjoined = 0
        unpriced = 0
        emitted_vendor_nodes: set[str] = set()
        for thread in selected:
            session_id = thread["vendor_session_id"]
            if _bb_join_issue(thread, vendor_owners) is not None:
                unjoined += 1
                continue
            if session_id is None:
                raise AssertionError("eligible bb join must carry a correlate")
            resolution = self.resolve_vendor_session(thread["provider"], session_id)
            if resolution.reason != "resolved":
                unjoined += 1
                continue
            session = resolution.session
            if session is None:
                raise AssertionError("resolved vendor session result must carry a session")

            vendor_node_id = session.run_id
            if vendor_node_id not in emitted_vendor_nodes:
                emitted_vendor_nodes.add(vendor_node_id)
                workspace = thread["workspace"]
                vendor_node: dict[str, Any] = {
                    "id": vendor_node_id,
                    "kind": "unknown",
                    "title": f"joined {thread['provider']} session",
                    "state": "settled_unverified",
                    "labels": {
                        "harness": thread["provider"],
                        "source": "vendor-session",
                    },
                }
                if workspace:
                    vendor_node["group"] = _workspace_group(workspace)
                nodes.append(vendor_node)
                vendor_attempt = _attempt(
                    vendor_node_id,
                    actor=thread["provider"],
                    harness=thread["provider"],
                    started_at=session.started_at,
                    wall_s=session.wall_s,
                    model=session.model,
                    model_tier=session.model_tier,
                    effort=session.effort,
                )
                vendor_attempt["session"] = session.correlate
                usage: Usage | None = session.usage
                pricing = self.price_usage(session.model, usage)
                cost = pricing.to_ocp()
                if cost is not None:
                    vendor_attempt["cost"] = cost
                if not pricing.priced:
                    unpriced += 1
                vendor_attempt["origin"] = {
                    "launched_by": _attempt_id(_thread_node_id(thread["id"])),
                    "workspace": _workspace_group(workspace) if workspace else None,
                    "external": False,
                    "how": "bb controller record joined to independently parsed vendor record",
                    "tier": "heuristic",
                    "evidence": (
                        "bb recorded one UUID and the host independently parsed "
                        "one matching vendor session"
                    ),
                }
                attempts.append(vendor_attempt)
                add_attempt_events(vendor_attempt)

            edges.append(
                {
                    "from": _thread_node_id(thread["id"]),
                    "to": vendor_node_id,
                    "kind": "launch",
                    "tier": "heuristic",
                    "from_attempt": _attempt_id(_thread_node_id(thread["id"])),
                    "to_attempt": _attempt_id(vendor_node_id),
                    "evidence": (
                        "bb recorded one UUID and the host independently parsed "
                        "one matching vendor session"
                    ),
                }
            )
            joined += 1

        nodes.sort(key=lambda node: node["id"])
        attempts.sort(key=lambda attempt: attempt["id"])
        edges.sort(
            key=lambda edge: (
                edge["kind"], edge["from"], edge["to"], edge["evidence"]
            )
        )
        events.sort(key=lambda event: (event["at"], event["type"], event["attempt"]))

        starts = [attempt["started_at"] for attempt in attempts if "started_at" in attempt]
        ends = [attempt["ended_at"] for attempt in attempts if "ended_at" in attempt]
        workspaces = sorted(
            {thread["workspace"] for thread in selected if thread["workspace"]}
        )
        run_seed = "\0".join(
            f"{thread['store']}\0{thread['id']}" for thread in selected
        ) or "\0".join(str(store) for store in sorted(unique_stores))
        run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:24]
        run: dict[str, Any] = {
            "id": f"dagr-adapter-bb:{run_id}",
            "title": "bb orchestration",
        }
        if len(workspaces) == 1:
            run["workspace"] = _workspace_group(workspaces[0])
        if starts:
            run["started_at"] = min(starts)
        if ends:
            run["ended_at"] = max(ends)

        selected_queue = sum(
            1
            for target, sender in queued
            if target in selected_ids or (sender is not None and sender in selected_ids)
        )
        ignored_snapshots = sum(thread["ignored_token_snapshots"] for thread in selected)
        producer: dict[str, Any] = {
            "name": "dagr-adapter-bb",
            "framework": "bb",
            "source_contract": "bb/sqlite",
            "capabilities": {
                "groups": True, "events": True, "artifacts": False, "edges_dep": False,
                "edges_spawn": True, "edges_launch": True, "edges_artifact": False,
                "cost_usd": True, "cost_tokens": True, "outcome_evidence": True,
            },
        }
        producer_version = self.producer_version()
        if producer_version is not None:
            producer["version"] = producer_version
        return {
            "ocp": "0.2",
            "producer": producer,
            "privacy": {
                "profile": "metadata_only",
                "note": "bb message and event content excluded; structural metadata only",
            },
            "run": run,
            "groups": [groups[group_id] for group_id in sorted(groups)],
            "nodes": nodes,
            "edges": edges,
            "attempts": attempts,
            "artifacts": [],
            "events": events,
            "ext": {
                EXT_NAMESPACE: {
                    "coverage": {
                        "selected_threads": len(selected),
                        "joined_vendor_sessions": joined,
                        "unjoined_vendor_sessions": unjoined,
                        "unpriced_vendor_sessions": unpriced,
                        "filtered_relations": filtered_relations,
                        "unrepresented_queued_messages": selected_queue,
                        "ignored_bb_token_snapshots": ignored_snapshots,
                    }
                }
            },
        }
