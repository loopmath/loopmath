"""types.PiecePrediction: `cost` is the full-run contribution, `cost_per_round` one execution."""

from __future__ import annotations

import json

from loopmath.types import Interval, Money, PiecePrediction


def _money(usd: float) -> Money:
    return Money(usd=Interval(usd, usd, usd), tokens=Interval(0.0, 0.0, 0.0))


def test_cost_per_round_round_trips():
    piece = PiecePrediction("implement", _money(1.5), Interval(0.5, 0.3, 0.7), Interval(1.5, 1.0, 2.0),
                            cost_per_round=_money(1.0))
    back = PiecePrediction.from_dict(json.loads(json.dumps(piece.to_dict())))
    assert back == piece and back.cost_per_round.usd.mean == 1.0 and back.cost.usd.mean == 1.5


def test_a_prediction_without_it_reads_as_none():
    old = PiecePrediction("plan", _money(1.0), None, Interval(1.0, 1.0, 1.0)).to_dict()
    del old["cost_per_round"]
    back = PiecePrediction.from_dict(old)
    assert back.cost_per_round is None and back == PiecePrediction("plan", _money(1.0), None, Interval(1.0, 1.0, 1.0))
