"""Synthetic banks with known parameters, for recovery checks of the task fit and the candidate posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .fit import Bank


@dataclass
class Truth:
    theta: np.ndarray
    b: np.ndarray
    a: np.ndarray
    u: np.ndarray
    p: np.ndarray


def simulate_bank(rng: np.random.Generator, *, arms: int, tasks: int, trials: int, model: str = "rasch",
                  tau: float = 1.5, slope_sd: float = 0.3, sigma_u: float = 0.0, theta_mean: float = 0.0,
                  theta_sd: float = 1.0) -> tuple[Truth, Bank]:
    """Arms x tasks counts of `trials` runs each, from the model the fit assumes."""
    theta = rng.normal(theta_mean, theta_sd, arms)
    b = rng.normal(0.0, tau, tasks)
    a = np.exp(rng.normal(0.0, slope_sd, tasks)) if model == "2pl" else np.ones(tasks)
    u = rng.normal(0.0, sigma_u, (arms, tasks)) if sigma_u > 0 else np.zeros((arms, tasks))
    p = expit(a[None, :] * (theta[:, None] - b[None, :]) + u)
    s = rng.binomial(trials, p)
    bank = Bank(arms=[f"arm{i:02d}" for i in range(arms)], tasks=[f"task{j:03d}" for j in range(tasks)],
                successes=s, trials=np.full_like(s, trials))
    return Truth(theta=theta, b=b, a=a, u=u, p=p), bank
