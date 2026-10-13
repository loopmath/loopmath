"""types.Control.rescue is one of three words or an OCP rescue object kept whole."""

from __future__ import annotations

import json

from loopmath.types import Control, Gate


def test_rescue_word_round_trips():
    control = Control(gates=(Gate("g1", "implement", "tests_pass"),), budget_rounds=2, rescue="person")
    back = Control.from_dict(json.loads(json.dumps(control.to_dict())))
    assert back == control and back.rescue == "person"
    assert Control().rescue == "redo_usual"


def test_rescue_object_is_kept_whole():
    rescue = {"kind": "workflow", "workflow": "plan_implement_review", "x.note": {"why": "fallback"}}
    control = Control.from_dict({"gates": [], "budget_rounds": 1, "rescue": rescue, "x.future": 3})
    assert control.rescue == rescue and control.extra == {"x.future": 3}
    assert control.to_dict()["rescue"] == rescue
    assert Control.from_dict(json.loads(json.dumps(control.to_dict()))) == control
