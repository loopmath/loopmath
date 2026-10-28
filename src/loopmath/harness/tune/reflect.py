"""The reflector: one session that reads the effects of the edits and returns one lesson as JSON.

Its working folder holds harness/ (the delivered harness), edits/<id>/ (each kept edit's folder) and EFFECTS.md.
The reply is one JSON object,

    {"lesson": str, "applies_when": str, "evidence": str, "confidence": "low" | "medium" | "high",
     "edits": [{"slot": str, "effect": "positive" | "negative" | "unclear"}]}

or {"lesson": null} when nothing generalizes. `parse_lesson` reads the last such object in the reply, so prose or
a code fence around it does no harm.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .contracts import Host, Lesson, ReflectRequest, SessionRequest, label
from .edits import HarnessSurface, harness_files
from .prompts import CONFIDENCE, EFFECTS, reflector_prompt

__all__ = ["SessionReflector", "parse_lesson", "json_objects"]

LESSON_MAX = 2000


def json_objects(text: str) -> list[dict]:
    """Every JSON object in `text` that is not inside another one, in order."""
    dec = json.JSONDecoder()
    out: list[dict] = []
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end


def _str(value: Any, limit: int = 1000) -> str:
    return " ".join(str(value).split())[:limit] if isinstance(value, (str, int, float)) else ""


def parse_lesson(text: str) -> Lesson:
    """The reflector's reply as a Lesson. The last JSON object with a `lesson` key wins. `lesson` null, missing
    text or blank gives Lesson(lesson=None). An unknown confidence reads as low, an unknown effect as unclear,
    and a malformed `edits` entry is dropped. ValueError when the reply holds no such object."""
    found = [o for o in json_objects(text or "") if "lesson" in o]
    if not found:
        raise ValueError("the reflector's reply holds no JSON object with a lesson key")
    obj = found[-1]
    lesson = obj.get("lesson")
    if not isinstance(lesson, str) or not lesson.strip():
        return Lesson(lesson=None)
    conf = str(obj.get("confidence") or "").strip().lower()
    edits = []
    raw = obj.get("edits")
    for e in raw if isinstance(raw, list) else []:
        if not isinstance(e, dict) or not _str(e.get("slot"), 200):
            continue
        effect = str(e.get("effect") or "").strip().lower()
        edits.append({"slot": _str(e["slot"], 200), "effect": effect if effect in EFFECTS else "unclear"})
    return Lesson(lesson=" ".join(lesson.split())[:LESSON_MAX], applies_when=_str(obj.get("applies_when")),
                  evidence=_str(obj.get("evidence")), confidence=conf if conf in CONFIDENCE else "low",
                  edits=edits)


def _copy(src: Path, dst: Path) -> None:
    if dst.exists() or not Path(src).is_dir():
        return
    shutil.copytree(src, dst, symlinks=False)


class SessionReflector:
    """The `Reflector` protocol: one session through `host.session`. `surface` names the always-on files the
    prompt shows (default: CLAUDE.md)."""

    def __init__(self, surface: HarnessSurface | None = None):
        self.surface = surface

    def _label(self, req: ReflectRequest) -> str:
        """`reflect`, or `r<k>-reflect` in round k: the engine names the request's folder the same way."""
        return req.folder.name if req.folder is not None else label("reflect")

    def seed(self, folder: Path, req: ReflectRequest) -> None:
        """harness/, edits/<id>/ and EFFECTS.md in the session's folder; what is there already stays."""
        folder.mkdir(parents=True, exist_ok=True)
        _copy(Path(req.harness), folder / "harness")
        for e in req.kept:
            _copy(Path(e.path), folder / "edits" / e.id)
        effects = folder / "EFFECTS.md"
        if not effects.exists():
            effects.write_text(req.effects_table.rstrip() + "\n", encoding="utf-8")

    def prompt(self, req: ReflectRequest) -> str:
        always_on = self.surface.always_on if self.surface else ("CLAUDE.md",)
        try:
            files = sorted(harness_files(Path(req.harness)))
        except Exception:
            files = []
        texts = {}
        for rel in always_on:
            p = Path(req.harness) / rel
            if p.is_file():
                texts[rel] = p.read_text(encoding="utf-8", errors="replace")
        d = req.delivery
        outcome = f"{d.choice} ({d.label}), under the rule {d.rule}" + (f": {d.reason}" if d.reason else "")
        return reflector_prompt(
            task=req.task.prompt, effects_table=req.effects_table,
            kept=[{"id": e.id, "component": e.component, "title": e.title, "why": e.why} for e in req.kept],
            refused=[{"id": r.id, "reason": r.reason} for r in req.refused], files=files, always_on=texts,
            outcome=outcome)

    def reflect(self, req: ReflectRequest, host: Host) -> Lesson | None:
        lab = self._label(req)
        ws = host.workspace(lab, "reflector")
        self.seed(Path(ws.folder), req)
        end = host.session(SessionRequest(label=lab, role="reflector", stage="reflect", prompt=self.prompt(req),
                                          workspace=ws, limits=host.limits("reflector"),
                                          meta={"kept": [e.id for e in req.kept]}))
        if not end.ok:
            return None
        return parse_lesson(end.text)
