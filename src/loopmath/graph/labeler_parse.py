"""Model-answer parsing and prediction-file I/O for the labeler."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .dataset import ATTEMPT_LABEL_KEYS, FINE_ROLE_WORDS, ROLE_WORDS
from .labeler_common import PROMPT_VERSION, PREDICTION_KEYS, LabelerError
from .scan import epoch


def _find_json_object(text: str) -> dict | None:
    """The first JSON object in `text` carrying a `labels` key: the whole text, a fenced
    block, or the outermost braces."""
    candidates = [text.strip()]
    candidates += [m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)]
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and "labels" in obj:
            return obj
    return None


def _check(key: str, value):
    """`(value, None)` when `value` is a valid prediction for `key`, else `(None, reason)`."""
    if value is None:
        return None, None
    if key == "role":
        return (value, None) if isinstance(value, str) and value in ROLE_WORDS else (None, f"not one of {', '.join(ROLE_WORDS)}")
    if key == "fine_role":
        return (value, None) if isinstance(value, str) and value in FINE_ROLE_WORDS else (None, f"not one of {', '.join(FINE_ROLE_WORDS)}")
    if key == "parent":
        return (value, None) if isinstance(value, str) and value else (None, "not a non-empty id string")
    if key == "boundaries":
        if not isinstance(value, list):
            return None, "not a list"
        bad = [v for v in value if not isinstance(v, str) or epoch(v) is None]
        if bad:
            return None, f"{len(bad)} entries are not ISO 8601 times"
        return value, None
    if key in ATTEMPT_LABEL_KEYS:
        return (value, None) if isinstance(value, bool) else (None, "not true or false")
    raise ValueError(key)


def parse_response(
    text: str,
    batch: list[dict],
    *,
    model: str,
    effort: str,
    batch_index: int,
    version: str = PROMPT_VERSION,
    prompt_metadata: dict | None = None,
) -> tuple[list[dict], Counter, list[str]]:
    """Prediction items for one batch out of the model's answer text. Raises
    `LabelerError` when no JSON object with a `labels` list is found. Returns the
    items (one per answered node of the batch, in batch order), counts and warnings."""
    c: Counter = Counter()
    warnings: list[str] = []
    obj = _find_json_object(text)
    if obj is None:
        raise LabelerError(f"batch {batch_index}: the answer holds no JSON object with a 'labels' key (first 200 chars: {text.strip()[:200]!r})")
    labels = obj["labels"]
    if not isinstance(labels, list):
        raise LabelerError(f"batch {batch_index}: 'labels' is not a list")
    asked = [it["id"] for it in batch]
    seen: dict[str, dict] = {}
    for i, entry in enumerate(labels):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            c["entries_malformed"] += 1
            warnings.append(f"batch {batch_index} entry {i}: not an object with a string id; ignored")
            continue
        nid = entry["id"]
        if nid not in asked:
            c["entries_unknown_id"] += 1
            warnings.append(f"batch {batch_index} entry {i}: id {nid} was not asked about; ignored")
            continue
        if nid in seen:
            c["entries_repeated_id"] += 1
            warnings.append(f"batch {batch_index} entry {i}: id {nid} answered again; the first answer is kept")
            continue
        preds: dict = {}
        rejected: dict = {}
        for key in PREDICTION_KEYS:
            if key not in entry:
                c[f"keys_missing:{key}"] += 1
                preds[key] = None
                rejected[key] = {"value": None, "reason": "key is missing"}
                continue
            value, reason = _check(key, entry[key])
            if reason:
                c[f"values_rejected:{key}"] += 1
                rejected[key] = {"value": entry[key], "reason": reason}
                warnings.append(f"batch {batch_index} node {nid}: {key} {json.dumps(entry[key])[:80]} rejected: {reason}")
            preds[key] = value
        conf = entry.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
            if "confidence" in entry:
                c["values_rejected:confidence"] += 1
                rejected["confidence"] = {"value": conf, "reason": "not a number from 0 to 1"}
            else:
                c["keys_missing:confidence"] += 1
                rejected["confidence"] = {"value": None, "reason": "key is missing"}
            conf = None
        evidence = entry.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            if "evidence" in entry:
                c["values_rejected:evidence"] += 1
                rejected["evidence"] = {"value": evidence, "reason": "not a non-empty string"}
            else:
                c["keys_missing:evidence"] += 1
                rejected["evidence"] = {"value": None, "reason": "key is missing"}
            evidence = None
        extra = sorted(k for k in entry if k not in PREDICTION_KEYS and k not in ("id", "confidence", "evidence"))
        if extra:
            c["keys_unknown"] += len(extra)
        prediction = {
            "item": "prediction",
            "id": nid,
            "tier": "reported",
            "model": model,
            "effort": effort,
            "prompt_version": version,
            "batch": batch_index,
            "labels": preds,
            "rejected": rejected,
            "confidence": conf,
            "evidence": evidence,
        }
        if prompt_metadata is not None:
            prediction["prompt_metadata"] = prompt_metadata
        seen[nid] = prediction
    for nid in asked:
        if nid not in seen:
            c["nodes_unanswered"] += 1
            warnings.append(f"batch {batch_index}: node {nid} has no entry in the answer")
    c["nodes_asked"] = len(asked)
    c["nodes_answered"] = len(seen)
    return [seen[nid] for nid in asked if nid in seen], c, warnings


def load_predictions(path: str | Path) -> tuple[list[dict], Counter, list[str]]:
    """Prediction items of a jsonl file; a second item for one id is counted and the
    first kept."""
    p = Path(path)
    if not p.is_file():
        raise LabelerError(f"predictions {p} does not exist")
    out: list[dict] = []
    c: Counter = Counter()
    warnings: list[str] = []
    seen: set[str] = set()
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise LabelerError(f"predictions {p} line {i}: not JSON ({e})") from e
            if not isinstance(obj, dict) or obj.get("item") != "prediction" or not isinstance(obj.get("id"), str) or not isinstance(obj.get("labels"), dict):
                raise LabelerError(f"predictions {p} line {i}: not a prediction item")
            if obj["id"] in seen:
                c["predictions_repeated_id"] += 1
                warnings.append(f"predictions line {i}: id {obj['id']} predicted again; the first is kept")
                continue
            seen.add(obj["id"])
            out.append(obj)
    c["predictions"] = len(out)
    return out, c, warnings


def write_predictions(preds: list[dict], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, sort_keys=True, ensure_ascii=False) + "\n")
