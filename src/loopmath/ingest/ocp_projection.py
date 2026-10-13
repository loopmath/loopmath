"""Extension-level projection accounting for the OCP reader."""

from __future__ import annotations

from collections import Counter, defaultdict

# The graph writer's keys, current first, then the pre-0.3 ones (W182).
_GRAPH_KEYS = ("dev.loopmath.graph", "dev.dagr.graph")
_ARTIFACT_KEYS = ("dev.loopmath.artifact", "dev.dagr.artifact")


def _projection_key(collection: str, record: dict):
    if collection in {"groups", "nodes", "attempts", "artifacts"}:
        return record.get("id")
    if collection == "edges":
        return (
            record.get("from"),
            record.get("to"),
            record.get("kind", "dep"),
            record.get("artifact"),
        )
    return (
        record.get("at"),
        record.get("type"),
        record.get("node"),
        record.get("attempt"),
    )


def _extension_loss_counters(document: dict, projected: dict) -> dict[str, int]:
    """Count foreign namespaces and changed members of modeled extensions."""
    counters: Counter = Counter()
    missing = object()

    def inspect(source, output, *, artifact_extension: bool = False) -> None:
        if isinstance(source, dict):
            source_ext = source.get("ext")
            output_ext = output.get("ext") if isinstance(output, dict) else None
            if isinstance(source_ext, dict):
                for namespace, value in source_ext.items():
                    if namespace in _GRAPH_KEYS:
                        counter = "ocp_projection_dagr_graph_members_unmodeled"
                        keys = _GRAPH_KEYS
                    elif namespace in _ARTIFACT_KEYS and artifact_extension:
                        counter = "ocp_projection_dagr_artifact_members_unmodeled"
                        keys = _ARTIFACT_KEYS
                    else:
                        counters["ocp_projection_extension_namespaces_unmodeled"] += 1
                        continue
                    # The writer re-emits a pre-0.3 key under its current name.
                    projected_value = missing
                    if isinstance(output_ext, dict):
                        projected_value = next(
                            (output_ext[k] for k in keys if k in output_ext), missing
                        )
                    if isinstance(value, dict):
                        counters[counter] += sum(
                            not isinstance(projected_value, dict)
                            or projected_value.get(key, missing) != member
                            for key, member in value.items()
                        )
                    elif projected_value != value:
                        counters[counter] += 1
            for key, nested in source.items():
                if key != "ext":
                    counterpart = output.get(key) if isinstance(output, dict) else None
                    inspect(nested, counterpart, artifact_extension=artifact_extension)
        elif isinstance(source, list):
            output_items = output if isinstance(output, list) else []
            for index, nested in enumerate(source):
                counterpart = output_items[index] if index < len(output_items) else None
                inspect(nested, counterpart, artifact_extension=artifact_extension)

    inspect({"ext": document.get("ext")}, {"ext": projected.get("ext")})
    for family in ("producer", "privacy", "run"):
        inspect(document.get(family), projected.get(family))
    for collection in ("groups", "nodes", "edges", "attempts", "artifacts", "events"):
        available: dict[object, list[dict]] = defaultdict(list)
        for record in projected.get(collection, []):
            if isinstance(record, dict):
                available[_projection_key(collection, record)].append(record)
        for record in document.get(collection, []):
            if not isinstance(record, dict):
                continue
            candidates = available.get(_projection_key(collection, record), [])
            exact = next(
                (index for index, candidate in enumerate(candidates) if candidate == record),
                0,
            )
            inspect(
                record,
                candidates.pop(exact) if candidates else None,
                artifact_extension=collection == "artifacts",
            )
    return {key: value for key, value in counters.items() if value}
