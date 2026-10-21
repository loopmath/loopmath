"""The history fit: each task's difficulty and slope from other arms' runs on it.

The model is item response theory on the logit scale. Arm a on task t accepts with chance
expit(a_t (theta_a - b_t) + u_at), where u_at ~ N(0, sigma_u^2) is the arm-by-task part no arm or task level
explains (integrated out per cell by Gauss-Hermite). Rasch fixes every slope a_t at 1; 2PL gives each task its
own, log a_t ~ N(0, slope_sd^2). Difficulty b_t ~ N(0, tau^2), ability theta_a ~ N(0, 3^2) (weak).

The mode is found by L-BFGS with an analytic gradient; standard errors come from the analytic Hessian there
(Laplace). tau is chosen by empirical Bayes: the Laplace marginal likelihood over a grid. Outcomes may be
fractional (a score in 0 to 1 counts as that share of one run), read as a quasi-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

TAUS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
THETA_SD = 3.0
NODES = 16
MODELS = ("rasch", "2pl")


def gauss_hermite(k: int = NODES) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights with sum(w f(x)) = E f(Z), Z ~ N(0, 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(k)
    return x, w / w.sum()


@dataclass(frozen=True)
class Bank:
    """Accepted runs (`successes`, may be fractional) and runs (`trials`) per arm and task, arms x tasks."""

    arms: list[str]
    tasks: list[str]
    successes: np.ndarray
    trials: np.ndarray

    def __post_init__(self):
        s, n = np.asarray(self.successes, float), np.asarray(self.trials, float)
        if s.shape != (len(self.arms), len(self.tasks)) or n.shape != s.shape:
            raise ValueError(f"successes and trials are arms x tasks, {len(self.arms)} x {len(self.tasks)}")
        if np.any(n < 0) or np.any(s < 0) or np.any(s > n + 1e-9):
            raise ValueError("successes must lie between 0 and the number of runs")
        object.__setattr__(self, "successes", s)
        object.__setattr__(self, "trials", n)

    @property
    def runs(self) -> int:
        return int(round(float(self.trials.sum())))


@dataclass
class TaskFit:
    """Per task: difficulty `b`, its standard error `b_se`, slope `a`; per history arm: ability `theta`.

    `theta_mean` and `theta_sd` are the spread of the history arms' abilities: a new arm's prior. `sigma_u` is
    the arm-by-task scale. `converged` and `grad_norm` record the optimizer's end state."""

    model: str
    tasks: list[str]
    b: np.ndarray
    b_se: np.ndarray
    a: np.ndarray
    theta: dict[str, float]
    theta_mean: float
    theta_sd: float
    tau: float
    sigma_u: float
    slope_sd: float
    converged: bool
    grad_norm: float
    history: dict[str, int] = field(default_factory=dict)  # per task: history runs (for display)

    def for_suite(self, tasks: list[str]) -> "TaskFit":
        """This fit on `tasks`, in that order. A task with no history gets the prior: b 0, its SE tau, slope 1."""
        idx = {t: j for j, t in enumerate(self.tasks)}
        b, se, a = [], [], []
        for t in tasks:
            j = idx.get(t)
            b.append(0.0 if j is None else float(self.b[j]))
            se.append(self.tau if j is None else float(self.b_se[j]))
            a.append(1.0 if j is None else float(self.a[j]))
        return replace(self, tasks=list(tasks), b=np.array(b), b_se=np.array(se), a=np.array(a),
                       history={t: self.history.get(t, 0) for t in tasks})

    def summary(self) -> dict:
        return {"model": self.model, "tasks": len(self.tasks), "arms": len(self.theta), "tau": round(self.tau, 3),
                "sigma_u": self.sigma_u, "slope_sd": self.slope_sd if self.model == "2pl" else None,
                "theta_mean": round(self.theta_mean, 3), "theta_sd": round(self.theta_sd, 3),
                "converged": bool(self.converged), "grad_norm": float(f"{self.grad_norm:.2e}")}


class _Objective:
    """Negative log posterior of (theta, b[, log a]) for one tau, with its gradient and Hessian."""

    def __init__(self, bank: Bank, model: str, tau: float, slope_sd: float, sigma_u: float):
        self.S, self.N = bank.successes, bank.trials
        self.mask = self.N > 0
        self.A, self.T = self.S.shape
        self.model, self.tau, self.slope_sd, self.sigma_u = model, tau, slope_sd, sigma_u
        self.x, self.w = gauss_hermite()
        self.dim = self.A + self.T * (2 if model == "2pl" else 1)

    def split(self, z):
        th, b = z[:self.A], z[self.A:self.A + self.T]
        al = z[self.A + self.T:] if self.model == "2pl" else np.zeros(self.T)
        return th, b, al

    def cells(self, eta):
        """Per cell: log-likelihood, its first and second derivative in eta."""
        S, N = self.S, self.N
        if self.sigma_u == 0:
            p = expit(eta)
            ll = S * log_expit(eta) + (N - S) * log_expit(-eta)
            return ll, S - N * p, -N * p * (1 - p)
        e = eta[..., None] + self.sigma_u * self.x
        lk = S[..., None] * log_expit(e) + (N - S)[..., None] * log_expit(-e)
        lw = lk + np.log(self.w)
        ll = logsumexp(lw, axis=-1)
        r = np.exp(lw - ll[..., None])
        p = expit(e)
        d1 = S[..., None] - N[..., None] * p
        g = np.sum(r * d1, axis=-1)
        h = np.sum(r * (d1 * d1 - N[..., None] * p * (1 - p)), axis=-1) - g * g
        return ll, g, h

    def value_grad(self, z):
        th, b, al = self.split(z)
        a = np.exp(al)
        eta = a[None, :] * (th[:, None] - b[None, :])
        ll, g, _ = self.cells(eta)
        ll, g = np.where(self.mask, ll, 0.0), np.where(self.mask, g, 0.0)
        lp = -0.5 * np.sum(th**2) / THETA_SD**2 - 0.5 * np.sum(b**2) / self.tau**2
        gth = (g * a[None, :]).sum(1) - th / THETA_SD**2
        gb = -(g * a[None, :]).sum(0) - b / self.tau**2
        parts = [gth, gb]
        if self.model == "2pl":
            lp += -0.5 * np.sum(al**2) / self.slope_sd**2
            parts.append((g * eta).sum(0) - al / self.slope_sd**2)
        return -(ll.sum() + lp), -np.concatenate(parts)

    def hessian(self, z):
        """The Hessian of the negative log posterior at z (dense)."""
        th, b, al = self.split(z)
        a = np.exp(al)
        eta = a[None, :] * (th[:, None] - b[None, :])
        _, g, h = self.cells(eta)
        g, h = np.where(self.mask, g, 0.0), np.where(self.mask, h, 0.0)
        A, T = self.A, self.T
        H = np.zeros((self.dim, self.dim))
        a2 = a[None, :] ** 2
        H[:A, :A] = np.diag((h * a2).sum(1))
        H[A:A + T, A:A + T] = np.diag((h * a2).sum(0))
        H[:A, A:A + T] = -h * a2
        H[A:A + T, :A] = H[:A, A:A + T].T
        if self.model == "2pl":
            k = A + T
            H[k:, k:] = np.diag((h * eta * eta + g * eta).sum(0))
            tha = h * a[None, :] * eta + g * a[None, :]
            H[:A, k:] = tha
            H[k:, :A] = tha.T
            ba = np.diag((-h * a[None, :] * eta - g * a[None, :]).sum(0))
            H[A:k, k:] = ba
            H[k:, A:k] = ba
        H = -H
        H[np.diag_indices(A)] += 1 / THETA_SD**2
        H[np.arange(A, A + T), np.arange(A, A + T)] += 1 / self.tau**2
        if self.model == "2pl":
            H[np.arange(A + T, self.dim), np.arange(A + T, self.dim)] += 1 / self.slope_sd**2
        return H

    def log_marginal(self, z, f):
        """Laplace: log p(data | tau) up to a constant shared by every tau."""
        H = self.hessian(z)
        try:  # Cholesky: slogdet raises spurious floating-point flags on this BLAS, and this checks H > 0
            logdet = 2 * float(np.sum(np.log(np.diag(np.linalg.cholesky(H)))))
        except np.linalg.LinAlgError:
            return -np.inf, H
        return -f - 0.5 * logdet - self.T * np.log(self.tau), H


def _mode(obj: _Objective, z0):
    res = minimize(obj.value_grad, z0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-8, "ftol": 1e-14})
    return res


def fit_bank(bank: Bank, *, model: str = "2pl", slope_sd: float = 0.3, sigma_u: float = 0.0,
             tau: float | None = None, taus: tuple[float, ...] = TAUS) -> TaskFit:
    """Fit the bank. `tau` None chooses it by empirical Bayes over `taus`."""
    if model not in MODELS:
        raise ValueError(f"model is one of {', '.join(MODELS)}, not {model!r}")
    if len(bank.arms) < 2:
        raise ValueError("a task fit needs runs of at least 2 arms")
    grid = (tau,) if tau is not None else taus
    best = None
    z = None
    for t in grid:
        obj = _Objective(bank, model, float(t), slope_sd, sigma_u)
        res = _mode(obj, np.zeros(obj.dim) if z is None else z)
        z = res.x
        lml, H = obj.log_marginal(res.x, res.fun) if len(grid) > 1 else (0.0, obj.hessian(res.x))
        if best is None or lml > best[0]:
            best = (lml, t, res, obj, H)
    _, t, res, obj, H = best
    th, b, al = obj.split(res.x)
    try:
        Linv = np.linalg.inv(np.linalg.cholesky(H))
        se = np.sqrt(np.clip(np.sum(Linv**2, axis=0), 1e-12, None))  # diag(H^-1) = column sums of L^-1 squared
    except np.linalg.LinAlgError:
        se = np.full(obj.dim, np.nan)
    th_se = se[:obj.A]
    grad = np.max(np.abs(obj.value_grad(res.x)[1]))
    A = obj.A
    spread = float(np.sqrt(max(np.var(th, ddof=1) if A > 1 else 0.0, 0.0) * (1 + 1 / A)))
    hist = {task: int(round(float(n))) for task, n in zip(bank.tasks, bank.trials.sum(0))}
    return TaskFit(model=model, tasks=list(bank.tasks), b=b.copy(), b_se=se[A:A + obj.T].copy(), a=np.exp(al),
                   theta={arm: float(v) for arm, v in zip(bank.arms, th)}, theta_mean=float(np.mean(th)),
                   theta_sd=max(spread, 0.3), tau=float(t), sigma_u=float(sigma_u), slope_sd=float(slope_sd),
                   converged=bool(res.success or grad < 1e-4) and bool(np.all(np.isfinite(th_se))),
                   grad_norm=float(grad), history=hist)
