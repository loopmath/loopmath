"""The dry run: resolved stages, session counts, cost point and range, time, residual df, warnings, preflight.
No model call, and nothing is written.

Sessions at the defaults: 15 executor runs (2 base + 12 probes + 1 final), 2 editor sessions, 1 test writer and 1
reflector. The prior per session is the published study's mean on its own tasks (`PRIORS`); a component on another
model has its prior scaled by the ratio of the two models' price-table rates at a 1 : 0.1 : 2 : 5 mix of input,
cache-read, cache-write and output tokens. Point = sum of count x prior. Low end: executor and judge priors at
0.5x, editors and reflector at 0.7x. High end: executor and judge at 3x (real repos run longer than the study's
tasks), and every editor and reflector session retried once. At the defaults that is 2.72 USD (1.72 to 6.38); with
`--judge-kind none` and a score command, 2.38 USD (1.55 to 5.36).

Time: waves of `run.width` sessions per stage. The worst case runs every wave at its sessions' `timeout_s` (plus
`score.timeout_s` for executor waves); the typical time needs wall priors per session, which are not set yet, so
`typical_s` is None and the plan says "typical: unknown".
"""

from __future__ import annotations

import math
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .contracts import Config, Plan, STAGES
from .design import MAX_EDITS, probe_count, residual_df
from .schema import parse_duration

__all__ = ["CLAUDE_FLOOR", "HERDR_FLOOR", "HIGH", "LOW", "PRIORS", "Prior", "TOKEN_MIX", "count_sessions",
           "make_plan", "preflight", "render", "session_high", "stage_order", "unit_cost"]

CLAUDE_FLOOR = (2, 1, 301)
HERDR_FLOOR = (0, 9, 3)
TOKEN_MIX = {"input": 1.0, "cache_read": 0.1, "cache_write": 2.0, "output": 5.0}


@dataclass(frozen=True)
class Prior:
    model: str
    usd: float
    source: str


PRIORS = {
    "executor": Prior("claude-haiku-5-5", 0.040, "study mean per Haiku 5.5 run: 21.04 USD over 520 runs"),
    "editor": Prior("claude-opus-5-5", 0.765, "study mean per Opus editor session: 61.12 USD over 80 sessions "
                    "(0.764), set to 0.765 so the plan reproduces the published figures"),
    "reflector": Prior("claude-opus-5-5", 0.25, "study mean per Opus reflector session"),
    "judge": Prior("claude-sonnet-5-5", 0.34, "study mean per task for the Sonnet test writer, its attempts included"),
}
LOW = {"executor": 0.5, "judge": 0.5, "editor": 0.7, "reflector": 0.7}
HIGH = {"executor": 3.0, "judge": 3.0, "editor": 2.0, "reflector": 2.0}  # 2.0: every session retried once


def _blended(rate: dict) -> float:
    return sum(TOKEN_MIX[k] * float(rate.get(k, 0.0)) for k in TOKEN_MIX)


def _scale(model: str, role: str, prices, warnings: list[str]) -> float:
    prior = PRIORS[role]
    if not model or model == prior.model:
        return 1.0
    new, old = (prices.rate(model), prices.rate(prior.model)) if prices is not None else (None, None)
    if new is None or old is None:
        missing = model if new is None else prior.model
        warnings.append(f"no price row for {missing}: {role} sessions on {model} are priced at the {prior.model} "
                        "prior, unscaled")
        return 1.0
    return _blended(new) / _blended(old)


def unit_cost(role: str, model: str = "", prices=None, units: dict | None = None) -> float:
    """One session's point cost in USD for `role` on `model`: the role's prior (or `units[role]`, a measured mean
    on the configured model, which the engine sets after the base runs) scaled by the price-table ratio."""
    if units and role in units:
        return float(units[role])
    return PRIORS[role].usd * _scale(model, role, prices, [])


def session_high(role: str, model: str = "", prices=None, units: dict | None = None) -> float:
    """One session's high estimate, `unit_cost` times the role's high multiplier: what the engine reserves against
    `budget.usd` before it starts the session."""
    return unit_cost(role, model, prices, units) * HIGH[role]


def stage_order(cfg: Config) -> list[str]:
    """The stages the config runs, in order. `reflect` moves to after `fit` with reflector.when = after-fit."""
    g = cfg.get
    on = {
        "judge": g("judge.kind") == "tester",
        "base": True,
        "signal": bool(g("signal.enabled")),
        "edit": True,
        "design": True,
        "probe": True,
        "fit": True,
        "build": True,
        "final": g("final.runs") > 0 and g("deliver.rule") != "fit",
        "check": g("deliver.rule") == "gate",
        "deliver": True,
        "reflect": bool(g("reflector.enabled")),
    }
    order = [s for s in STAGES if on[s]]
    if "reflect" in order and g("reflector.when") == "after-fit":
        order.remove("reflect")
        order.insert(order.index("fit") + 1, "reflect")
    return order


def design_edits(cfg: Config, edits: int | None = None) -> int:
    """Edits the design takes at the plan: the given count (edit.from folders), else every editor's max_edits
    summed, capped at screen.max_edits and the design's own limit."""
    g = cfg.get
    cap = min(g("screen.max_edits"), MAX_EDITS[g("screen.design")])
    if edits is not None:
        return min(edits, cap)
    if not g("edit.enabled"):
        return cap
    return min(sum(int(e.get("max_edits") or 0) for e in cfg.editors), cap)


def count_sessions(cfg: Config, edits: int | None = None) -> tuple[dict, dict, dict]:
    """(sessions at the plan, sessions with every retry, the parts): executor, editor, judge, reflector."""
    g = cfg.get
    m = design_edits(cfg, edits)
    probes = probe_count(g("screen.design"), m, g("screen.repeats"), g("screen.pairs"))
    rule = g("deliver.rule")
    finals = 0 if rule == "fit" or m == 0 else g("final.runs")
    rounds = g("loop.rounds")
    base_fresh = 0 if g("base.reuse") else g("base.runs")
    later_base = max(0, g("base.runs") - finals) * (rounds - 1)
    checks = 2 * g("check.max_pairs") if rule == "gate" else 0
    editors = len(cfg.editors) if g("edit.enabled") else 0
    executor = base_fresh + later_base + rounds * (probes + finals)
    reflector = rounds if g("reflector.enabled") else 0
    kind = g("judge.kind")
    judge = 1 if kind == "tester" else (executor if kind == "rubric" else 0)
    point = {"executor": executor, "editor": rounds * editors, "judge": judge, "reflector": reflector}
    retries = sum(int(e.get("attempts") or 1) for e in cfg.editors) if g("edit.enabled") else 0
    high = {"executor": executor + rounds * checks, "editor": rounds * retries,
            "judge": g("judge.attempts") if kind == "tester" else judge, "reflector": 2 * reflector}
    parts = {"edits": m, "probes": probes, "finals": finals, "rounds": rounds, "base": base_fresh,
             "checks": checks}
    return point, high, parts


def _cost(cfg: Config, point: dict, high_counts: dict, prices, warnings: list[str]) -> dict:
    g = cfg.get
    by_role: dict[str, dict] = {}
    exec_scale = _scale(g("executor.model"), "executor", prices, warnings)
    judge_scale = _scale(g("judge.model"), "judge", prices, warnings)
    refl_scale = _scale(g("reflector.model"), "reflector", prices, warnings)
    ed_scales = [_scale(e.get("model", ""), "editor", prices, warnings) for e in cfg.editors] \
        if g("edit.enabled") else []
    rounds = g("loop.rounds")

    def put(role: str, unit: float, n: float) -> None:
        by_role[role] = {"point": n * unit, "low": n * unit * LOW[role], "high": n * unit * HIGH[role]}

    put("executor", PRIORS["executor"].usd * exec_scale, point["executor"])
    put("judge", PRIORS["judge"].usd * judge_scale, point["judge"])
    ed_unit = PRIORS["editor"].usd
    ed_point = rounds * sum(ed_unit * s for s in ed_scales)
    ed_high = rounds * sum(ed_unit * s * min(2, int(e.get("attempts") or 1)) for s, e in zip(ed_scales, cfg.editors))
    by_role["editor"] = {"point": ed_point, "low": ed_point * LOW["editor"], "high": ed_high}
    r_unit = PRIORS["reflector"].usd * refl_scale
    put("reflector", r_unit, point["reflector"])
    checks = high_counts["executor"] - point["executor"]
    if checks:
        by_role["executor"]["high"] += checks * PRIORS["executor"].usd * exec_scale * HIGH["executor"]
    total = {k: sum(v[k] for v in by_role.values()) for k in ("point", "low", "high")}
    return {**{k: round(v, 4) for k, v in total.items()},
            "by_role": {r: {k: round(v, 4) for k, v in d.items()} for r, d in by_role.items()},
            "priors": {r: {"model": p.model, "usd": p.usd} for r, p in PRIORS.items()}}


def _time(cfg: Config, point: dict, parts: dict, order: Sequence[str]) -> dict:
    g = cfg.get
    w = max(1, g("run.width"))
    ex_t = g("executor.timeout_s") + g("score.timeout_s")
    waves: dict[str, int] = {}
    worst: dict[str, float] = {}
    judge = point["judge"] if g("judge.kind") == "tester" else 0
    if "judge" in order or "base" in order:
        waves["base"] = math.ceil((parts["base"] + judge) / w) if parts["base"] + judge else 0
        worst["base"] = max(waves["base"] * ex_t, g("judge.attempts") * g("executor.timeout_s") if judge else 0)
    eds = cfg.editors if g("edit.enabled") else []
    if eds:
        ew = g("run.editor_width") or len(eds)
        waves["edit"] = math.ceil(len(eds) / ew)
        worst["edit"] = waves["edit"] * max(int(e.get("timeout_s") or 0) * int(e.get("attempts") or 1) for e in eds)
    waves["probe"] = math.ceil(parts["probes"] / w) if parts["probes"] else 0
    worst["probe"] = waves["probe"] * ex_t
    if "final" in order and parts["finals"]:
        waves["final"] = math.ceil(parts["finals"] / w)
        worst["final"] = waves["final"] * ex_t
    if "check" in order and parts["checks"]:
        waves["check"] = parts["checks"] // 2
        worst["check"] = waves["check"] * ex_t
    if "reflect" in order:
        waves["reflect"] = 1
        worst["reflect"] = g("reflector.timeout_s")
    rounds = parts["rounds"]
    total = worst.get("base", 0) + rounds * sum(v for k, v in worst.items() if k != "base")
    return {"typical_s": None, "worst_s": float(total), "waves": waves, "worst_by_stage": worst,
            "typical": "unknown: the wall priors per session are not set"}


def make_plan(cfg: Config, *, edits: int | None = None, prices=None, checks: list[dict] | None = None) -> Plan:
    """The plan of a resolved config. `edits` is the edit count when known (edit.from folders); `prices` a
    `loopmath.price.PriceTable` (the packaged table when None); `checks` a preflight result to attach."""
    if prices is None:
        try:
            from ...price import load_prices

            prices = load_prices()
        except Exception:  # a broken table only loses the scaling, and says so
            prices = None
    g = cfg.get
    warnings: list[str] = []
    order = stage_order(cfg)
    point, high, parts = count_sessions(cfg, edits)
    cost = _cost(cfg, point, high, prices, warnings)
    time = _time(cfg, point, parts, order)
    df = residual_df(g("screen.design"), parts["edits"], repeats=g("screen.repeats"), pairs=g("screen.pairs"),
                     base_runs=g("base.runs"), base_rows=g("screen.base_rows"))
    if df is not None and df < 2:
        warnings.append(f"the design leaves {df} residual degrees of freedom: under 2 the fit is weak"
                        + (", and under 1 it is refused" if df < 1 else ""))
    caps = {"usd": g("budget.usd"), "runs": g("budget.runs"), "wall_s": parse_duration(g("budget.wall")),
            "set_usd": g("budget.set_usd")}
    if cost["high"] > caps["usd"]:
        warnings.append(f"the high estimate {cost['high']:.2f} USD passes budget.usd {caps['usd']:.2f}: the tune "
                        "stops at the cap if it gets there")
    if high["executor"] > caps["runs"]:
        warnings.append(f"{high['executor']} executor runs pass budget.runs {caps['runs']}")
    if time["worst_s"] > caps["wall_s"]:
        warnings.append(f"the worst case {_hours(time['worst_s'])} passes budget.wall {g('budget.wall')}: a tune "
                        "that slow stops at the cap")
    if parts["edits"] == 0:
        warnings.append("no edit can reach the design: the tune measures the base only")
    return Plan(stages=order, sessions=point, sessions_high=high, cost=cost, time=time, design=g("screen.design"),
                probes=parts["probes"], edits=parts["edits"], df=df, warnings=warnings,
                preflight=list(checks or []), caps=caps)


def _hours(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def render(plan: Plan) -> str:
    """The plan as the CLI prints it."""
    s = plan.sessions
    c = plan.cost
    parts = [_n(s["executor"], "executor run", "executor runs"),
             _n(s["editor"], "editor session", "editor sessions"),
             _n(s["reflector"], "reflector", "reflectors")]
    if s["judge"]:
        parts.append(_n(s["judge"], "test writer", "test writers"))
    lines = [
        f"stages: {', '.join(plan.stages)}",
        f"sessions: {', '.join(parts)}",
        f"cost: {c['point']:.2f} USD ({c['low']:.2f} to {c['high']:.2f})",
        f"time: typical unknown; worst case {_hours(plan.time['worst_s'])}",
        f"design: {plan.design}, {plan.probes} probe runs over up to {plan.edits} edits, residual df "
        f"{'n/a' if plan.df is None else plan.df}",
        f"caps: {plan.caps['usd']:.2f} USD, {plan.caps['runs']} runs, {_hours(plan.caps['wall_s'])}",
    ]
    lines += [f"warning: {w}" for w in plan.warnings]
    for chk in plan.preflight:
        lines.append(f"preflight {'ok' if chk['ok'] else 'FAIL'}: {chk['line']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- preflight

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _version(text: str) -> tuple[int, int, int] | None:
    m = _VERSION.search(text or "")
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def _run_version(binary: str, run: Callable) -> tuple[tuple | None, str]:
    try:
        out = run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return _version(out.stdout + " " + out.stderr), (out.stdout or out.stderr).strip()[:120]


def preflight(cfg: Config, *, repo: Path | None = None, work_root: Path | None = None, prices=None,
              which: Callable[[str], str | None] = shutil.which, run: Callable = subprocess.run) -> list[dict]:
    """[{"check", "ok", "line"}]: the claude version floor, the repo and base_ref, no instruction files above the
    work root, the token file's mode (never read), a price row per model, and the herdr floor for that runner.
    `run` and `which` are injectable; tests put stub binaries first on PATH."""
    g = cfg.get
    out: list[dict] = []

    def add(check: str, ok: bool, line: str) -> None:
        out.append({"check": check, "ok": bool(ok), "line": line})

    claude = which("claude")
    if not claude:
        add("claude", False, "claude is not on PATH")
    else:
        v, text = _run_version(claude, run)
        add("claude", v is not None and v >= CLAUDE_FLOOR,
            f"claude {'.'.join(map(str, v)) if v else 'version unknown'} (floor {'.'.join(map(str, CLAUDE_FLOOR))})"
            + ("" if v else f": {text}"))
    if repo is not None:
        try:
            got = run(["git", "-C", str(repo), "rev-parse", "--verify", f"{g('task.base_ref')}^{{commit}}"],
                      capture_output=True, text=True, timeout=60)
            ok = got.returncode == 0
            add("repo", ok, f"{repo} at {g('task.base_ref')}: " + (got.stdout.strip()[:12] if ok
                                                                    else (got.stderr or got.stdout).strip()[:200]))
        except (OSError, subprocess.TimeoutExpired) as exc:
            add("repo", False, f"git: {exc}")
    if work_root is not None:
        hits = []
        for d in [Path(work_root).resolve(), *Path(work_root).resolve().parents]:
            for name in ("CLAUDE.md", "AGENTS.md", ".claude"):
                if d == Path.home() and name == ".claude":
                    continue  # the user's config folder, which every run replaces with its own
                if (d / name).exists():
                    hits.append(str(d / name))
        add("work_root", not hits, "no instruction files above the work root" if not hits
            else f"instruction files above the work root would load into every run: {', '.join(hits[:3])}")
    if g("auth.mode") == "token-file":
        tf = g("auth.token_file")
        if not tf:
            add("token", False, "auth.token_file is not set (write one with `claude setup-token`)")
        else:
            p = Path(tf).expanduser()
            try:
                mode = stat.S_IMODE(os.stat(p).st_mode)
                add("token", mode == 0o600, f"{p}: mode {mode:o}" + ("" if mode == 0o600 else ", needs 600"))
            except OSError:
                add("token", False, f"{p}: no such file")
    else:
        add("token", bool(os.environ.get(g("auth.api_key_env"))), f"{g('auth.api_key_env')} "
            + ("is set" if os.environ.get(g("auth.api_key_env")) else "is not set"))
    if prices is None:
        try:
            from ...price import load_prices

            prices = load_prices()
        except Exception as exc:
            add("prices", False, f"the price table does not load: {exc}")
    if prices is not None:
        models = {g("executor.model")} | {e.get("model") for e in cfg.editors}
        if g("judge.kind") != "none":
            models.add(g("judge.model"))
        if g("reflector.enabled"):
            models.add(g("reflector.model"))
        missing = sorted(m for m in models if m and prices.rate(m) is None)
        add("prices", not missing, "every model has a price row" if not missing
            else f"no price row for {', '.join(missing)}")
    if g("run.runner") == "herdr":
        herdr = which("herdr")
        if not herdr:
            add("herdr", False, "herdr is not on PATH")
        else:
            v, text = _run_version(herdr, run)
            add("herdr", v is not None and v >= HERDR_FLOOR,
                f"herdr {'.'.join(map(str, v)) if v else 'version unknown'} (floor "
                f"{'.'.join(map(str, HERDR_FLOOR))})")
    return out
