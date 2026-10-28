"""What is delivered: the delivery rules, and the effects table the reflector and `show --effects` read.

final-vs-base-mean (the default): keep the edits the fit keeps. With none (no usable edit, a refused fit, nothing
separated) there is no final run and the better base run is delivered. Otherwise the kept set's build is run
`final.runs` times and delivered unless its final score is below the mean of the scored base runs; then the better
base run is. Cases the rule does not name: a final run with a deliverable that the judge could not score is
delivered (a judge failure says nothing about the run); a final with no run, or no deliverable, never beats a base
run; with no base run scored, a scored final run is delivered; ties between base runs, or no base scores, go to
the first base run.

fit: the build, with no final run. always-build: the build whenever an edit was kept (a final run, when
`final.runs` asks for one, is measured but cannot undo it). gate: `loopmath.gate.decide(base, final, policy)`:
adopt delivers the build, reject the base, and replicate asks for another base and final pair, up to
`check.max_pairs`, after which an unresolved gate keeps the base.

The end status follows from the choice: the build is `delivered`; the base is `no-signal` when the measurements
could not separate an edit (a refused fit, no kept edit on a usable fit), else `base`.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .contracts import Delivery, Edit, Fit, Refusal, RunResult

__all__ = ["RULES", "after_final", "base_mean", "before_final", "best_base", "effects_table", "gate_step",
           "lesson_why", "status_of"]

RULES = ("final-vs-base-mean", "fit", "gate", "always-build")


def best_base(base: Sequence[RunResult]) -> str:
    """The base run scored higher; ties and no scores go to the first."""
    scored = [(r.value, -i) for i, r in enumerate(base) if r.value is not None]
    if not scored:
        return base[0].label if base else ""
    return base[-max(scored)[1]].label


def base_mean(base: Sequence[RunResult]) -> float | None:
    got = [r.value for r in base if r.value is not None]
    return sum(got) / len(got) if got else None


def before_final(rule: str, base: Sequence[RunResult], fit: Fit | None, *, final_runs: int = 1,
                 no_edit: str = "") -> Delivery:
    """After the fit: the kept edits, and whether a final run is due. `no_edit` says why there was no edit to fit
    (no usable edit came back), when that is the case."""
    if rule not in RULES:
        raise ValueError(f"unknown delivery rule {rule!r}")
    best = best_base(base)
    usable = fit is not None and not fit.refused
    kept = list(fit.kept) if usable else []
    d = Delivery(rule=rule, choice="base", label=best, kept=kept, base_mean=base_mean(base), best_base=best)
    if no_edit or fit is None:
        d.reason = f"no edit to measure ({no_edit or 'no usable edit'}): deliver the better base run"
        d.status = "base"
    elif fit.refused:
        d.reason = f"the fit was refused ({fit.refused}): deliver the better base run"
        d.status = "no-signal"
    elif not kept:
        d.reason = "no edit separated from noise: deliver the better base run"
        d.status = "no-signal"
    elif rule == "fit":
        d.choice, d.label, d.status = "build", "build", "delivered"
        d.reason = f"the fit keeps {', '.join(kept)}: deliver their build, with no final run"
    elif rule == "always-build":
        d.choice, d.label, d.status = "build", "build", "delivered"
        d.run_final = final_runs > 0
        d.reason = f"an edit was kept ({', '.join(kept)}): deliver their build"
    else:
        d.run_final = final_runs > 0
        d.reason = "final run of the kept set pending"
    return d


def _final_score(finals: Sequence[RunResult]) -> float | None:
    got = [r.value for r in finals if r.value is not None and r.deliverable]
    return sum(got) / len(got) if got else None


def _best_final(finals: Sequence[RunResult]) -> RunResult | None:
    with_work = [r for r in finals if r.deliverable and not r.left_out]
    if not with_work:
        return None
    scored = [(r.value, -i) for i, r in enumerate(with_work) if r.value is not None]
    return with_work[-max(scored)[1]] if scored else with_work[0]


def after_final(d: Delivery, finals: Sequence[RunResult], *, why_no_run: str = "") -> Delivery:
    """After the final runs, for final-vs-base-mean and always-build. `why_no_run` says why a final is missing
    (the build did not validate, the executor failed on every try)."""
    if not d.run_final:
        return d
    d.final_score = _final_score(finals)
    best = _best_final(finals)
    if d.rule == "always-build":
        if best is not None:
            d.label = best.label
        return d
    if best is None:
        d.choice, d.label, d.status = "base", d.best_base, "base"
        d.reason = f"no final run to deliver ({why_no_run or 'no deliverable'}): deliver the better base run"
    elif d.final_score is None:
        d.choice, d.label, d.status = "build", best.label, "delivered"
        d.reason = "the final run has a deliverable but no score, which says nothing about the run: deliver it"
    elif d.base_mean is None:
        d.choice, d.label, d.status = "build", best.label, "delivered"
        d.reason = "no base run was scored, so nothing to compare against: deliver the final run"
    elif d.final_score < d.base_mean:
        d.choice, d.label, d.status = "base", d.best_base, "base"
        d.reason = (f"the final run scored {d.final_score:.1f}, below the base mean {d.base_mean:.1f}: deliver the "
                    "better base run")
    else:
        d.choice, d.label, d.status = "build", best.label, "delivered"
        d.reason = (f"the final run scored {d.final_score:.1f}, not below the base mean {d.base_mean:.1f}: deliver "
                    "it")
    return d


def gate_step(d: Delivery, base: Sequence[RunResult], finals: Sequence[RunResult], *, pairs_added: int,
              max_pairs: int, decide: Callable[[Sequence[RunResult], Sequence[RunResult]], Mapping]) -> str:
    """One look of the gate rule: "adopt", "reject" or "replicate". `decide(base, finals)` returns {"action",
    "reasons"} (the engine wraps `loopmath.gate.decide`). A replicate at `max_pairs` added pairs keeps the base."""
    d.final_score = _final_score(finals)
    got = dict(decide(base, finals))
    action = str(got.get("action") or "reject")
    why = "; ".join(map(str, got.get("reasons") or [])) or action
    best = _best_final(finals)
    if action == "replicate" and pairs_added < max_pairs:
        d.reason = f"the gate asks for another pair ({why})"
        return "replicate"
    if action == "adopt" and best is not None:
        d.choice, d.label, d.status = "build", best.label, "delivered"
        d.reason = f"the gate adopts the build ({why})"
        return "adopt"
    d.choice, d.label, d.status = "base", d.best_base, "base"
    d.reason = (f"the gate is unresolved after {pairs_added} added pairs: keep the base ({why})"
                if action == "replicate" else f"the gate keeps the base ({why})")
    return "reject"


def status_of(d: Delivery) -> str:
    return d.status or ("delivered" if d.choice == "build" else "base")


def lesson_why(d: Delivery) -> str:
    """Why the delivered harness was delivered, for a library lesson: the rule's branch with no score in it, since a
    lesson outlives the task its scores were given on."""
    kept = ", ".join(d.kept)
    if d.choice == "build":
        return f"delivered the build of the kept edits ({kept}), each measured positive"
    if d.status == "no-signal":
        return "delivered the base: no edit separated from noise"
    if not d.kept:
        return "delivered the base: no edit was measured"
    return f"delivered the base: the build of the kept edits ({kept}) did not hold up in its final run"


def _num(v: float | None, signed: bool = True, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def effects_table(edits: Sequence[Edit], fit: Fit | None, delivery: Delivery, base: Sequence[RunResult], *,
                  failures: Mapping | None = None, flags: Sequence[Mapping] = (), refusals: Sequence[Refusal] = (),
                  design_kind: str = "pb12", finals: Sequence[RunResult] = ()) -> str:
    """The experiment as markdown: one row per edit, then the runs and the delivery. Effects are in score points
    (0 to 100), on versus off."""
    fit = fit or Fit(edits=[])
    model = fit.head.get("edits", {}) if fit.head.get("available") else {}
    failures = failures or {}
    lines = [
        "# Edit effects",
        "",
        f"Design: {design_kind}. The base runs count as all-off rows. Effect: the least-squares change in the 0 to "
        "100 score with the edit on. Screen model: loopmath's shrinkage estimate on the probe runs, with its 80% "
        "range and its probability that the edit helps.",
        "",
        "| edit | editor | component | title | effect | SE | screen model (80%) | P(helps) | no deliverable on / off "
        "| kept |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in edits:
        m = model.get(e.id)
        mtxt = f"{_num(m['mean'])} ({_num(m['lo'])}, {_num(m['hi'])})" if m else "n/a"
        ptxt = f"{m['p_helps']:.2f}" if m else "n/a"
        f = failures.get(e.id)
        ftxt = f"{f['on'][0]}/{f['on'][1]} vs {f['off'][0]}/{f['off'][1]}" if f else "n/a"
        title = str(e.title).replace("|", "/")[:80]
        lines.append(f"| {e.id} | {e.editor} | {e.component} | {title} | {_num(fit.effects.get(e.id))} "
                     f"| {_num(fit.se.get(e.id), signed=False)} | {mtxt} | {ptxt} | {ftxt} "
                     f"| {'yes' if e.id in delivery.kept else 'no'} |")
    if not edits:
        lines.append("| (no usable edit) | | | | | | | | | |")
    lines.append("")
    if fit.edits:
        lines.append(f"Fit: {fit.n_rows} scored runs, residual SD {_num(fit.sigma, signed=False)} on {fit.df} "
                     "degrees of freedom." + (f" Refused: {fit.refused}." if fit.refused else ""))
    if fit.head and not fit.head.get("available"):
        lines.append(f"Screen model: not run ({fit.head.get('reason', 'no reason given')}).")
    lines += ["", "## Runs", ""]
    for r in base:
        lines.append(f"- {r.label} (base): {_num(r.value, signed=False)}")
    for r in finals:
        lines.append(f"- {r.label} (kept set): {_num(r.value, signed=False)}")
    if fit.excluded:
        lines.append(f"- left out of the fit: {', '.join(fit.excluded)}")
    lines += ["", f"Delivered: {delivery.label} ({delivery.choice}). {delivery.reason}."]
    if flags:
        pairs = "; ".join(f"{', '.join(f['edits'])} in {f['file']}" for f in flags)
        lines.append(f"Kept edits adding to one always-on file (built by appending): {pairs}.")
    if refusals:
        lines += ["", "## Edits refused before the probes", ""]
        lines += [f"- {r.id}: {r.reason}" for r in refusals]
    return "\n".join(lines) + "\n"
