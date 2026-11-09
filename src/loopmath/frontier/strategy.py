"""The pick seam: a `Strategy` gives the next trials, a `TaskChooser` swaps only the task given the setting.

v1 ships one strategy, `kg+random`: kg_dollar over the open settings, then a task drawn by the declared task
weights, with replacement. A difficulty-weighted chooser is a new `TaskChooser` that returns its pick probability;
a joint (setting, task) KG or a frontier Thompson picker is a new `Strategy`. Neither touches `kg`, the CLI or the
state file. The strategy name is stored in the frontier file, and resuming under another name is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

if TYPE_CHECKING:
    from .cost import CostModel
    from .model import TaskPosterior


@dataclass
class Pick:
    setting: int
    task: int
    p_task: float               # the chooser's probability of this task given the setting, logged with the trial
    why: dict = field(default_factory=dict)   # e.g. {"kg": 0.012, "L": 10}; JSON-able, written to the trial


@dataclass
class StrategyConfig:
    """What a strategy factory gets. `weights`: T task weights summing to 1 (the declared ones, else equal).
    `spec`: the `MatchSpec` fields as a dict. `seed`: the frontier's seed. `picker`: extra `Picker` keywords."""

    n_settings: int
    weights: np.ndarray
    spec: dict
    seed: int
    picker: dict = field(default_factory=dict)


class TaskChooser(Protocol):
    def choose(self, setting: int, post: TaskPosterior, rng: np.random.Generator) -> tuple[int, float]:
        """The task for the next run of `setting`, and the probability this chooser gave it."""


class Strategy(Protocol):
    name: str

    def pick(self, post: TaskPosterior, cost: CostModel, open_: np.ndarray, k: int,
             rng: np.random.Generator) -> list[Pick]:
        """Up to `k` picks on distinct settings among `open_` (bool, shape S), best first. Empty when none is open."""


class RandomTasks:
    """Task drawn by the declared weights, with replacement; `p_task` is that weight."""

    def __init__(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, float)
        if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("task weights must be finite, nonnegative and have positive total")
        self.weights = weights / weights.sum()

    def choose(self, setting: int, post: TaskPosterior, rng: np.random.Generator) -> tuple[int, float]:
        task = int(rng.choice(len(self.weights), p=self.weights))
        return task, float(self.weights[task])


class KGThenTask:
    """kg_dollar for the setting (`kg.Picker`, top `k` distinct settings for `next --k`), then the chooser's task."""

    name = "kg+random"

    def __init__(self, picker: Any, chooser: TaskChooser) -> None:
        self.picker, self.chooser = picker, chooser

    def pick(self, post: TaskPosterior, cost: CostModel, open_: np.ndarray, k: int,
             rng: np.random.Generator) -> list[Pick]:
        from .kg import MatchSpec
        if k < 1:
            raise ValueError("k must be positive")
        available = np.asarray(open_, bool).copy()
        picks = []
        for _ in range(min(k, int(available.sum()))):
            s = self.picker.pick(post, getattr(self.picker, "spec", MatchSpec(kind="ci", eps_q=0.035)), available)
            t, probability = self.chooser.choose(s, post, rng)
            picks.append(Pick(int(s), int(t), float(probability), {"picker": self.picker.kind}))
            available[s] = False
        return picks


def _kg_random(cfg: StrategyConfig) -> Strategy:
    from .kg import Picker, MatchSpec

    picker = Picker("kg_dollar", cfg.n_settings, cfg.seed, **cfg.picker)
    picker.spec = MatchSpec(**cfg.spec)
    return KGThenTask(picker, RandomTasks(cfg.weights))


STRATEGIES: dict[str, Callable[[StrategyConfig], Strategy]] = {"kg+random": _kg_random}
