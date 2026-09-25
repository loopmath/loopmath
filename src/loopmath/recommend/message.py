"""The user-facing message paragraph (spec 05, section 4).

The exact template, filled here:

"Your usual workflow ({usual_label}) has a {g_usual}% chance of an accepted result at
about {cost_usual} ({tokens_usual} tokens). {goal_sentence} Trying {explore_label}
alongside it costs {price} ({price_tokens} tokens) now. There is a {p_beats}% chance it
beats the recommended pick. Trying it once is expected to save about {gain} on each future
similar run, so it pays for itself after about {payback} similar runs.{max_gain_sentence}"

The gain is lane 05's look-ahead value in dollars: an expectation over every outcome of the
trial run, not a gain on the condition that it beats the goal, and its success, cost and score
parts belong to whichever workflow is best after learning, so only the dollars are shown (D96).

A value whose mean is above its interval's upper end adds TAIL inside its parentheses
(D107): after D98 and D106 that means a real heavy tail. The interval is not printed; the
note is keyed on it all the same.

For score rules the chance reads "chance of reaching {name} {op} {target}". When a
pick is paused by the budget cap or no candidate qualifies, plain variants of the
exploration sentences say so instead.

With no usual workflow the first sentence names the reference instead (spec 05
section 1, D118 N4): "You have no usual workflow for this task; the reference is
your best recorded workflow ({label}), with ..." (or "the default workflow"). A
workflow the user did not run is never "your usual" (F4).
"""

from __future__ import annotations

from typing import Any, Callable

from ..types import AcceptanceRule, Configuration, Prediction
from .gain import Exploration, Slot


TAIL = "the average is pulled up by rare very large outcomes"


def heavy(*values: Any) -> bool:
    """Whether a shown mean is above its interval's upper end (D107)."""
    return any(v is not None and v.mean > v.hi for v in values)


def tail(*values: Any) -> str:
    """The D107 note to append inside a value's existing parentheses, or ""."""
    return f"; {TAIL}" if heavy(*values) else ""


def noted(text: str, *values: Any) -> str:
    """`text`, then the D107 note in parentheses when a mean is pulled up."""
    return f"{text} ({TAIL})" if heavy(*values) else text


def run_cost(m: Any) -> str:
    """"$2.82 (6,760,000 tokens)", with the D107 note inside the parentheses when either mean is pulled up."""
    return f"{usd(m.usd.mean)} ({tokens(m.tokens.mean)} tokens{tail(m.usd, m.tokens)})"


def usd(x: float) -> str:
    if 0 < x < 0.01:
        return "under $0.01"
    return f"${x:,.2f}"


def tokens(x: float) -> str:
    return f"{int(round(x, -3) if x >= 10_000 else round(x)):,}"


def pct(p: float) -> str:
    return f"{round(p * 100):.0f}%"


def a_pct(p: float) -> str:
    """The percentage with its article as it is read aloud: an 8%, an 11%, an 18%, an 80% to an 89%."""
    text = pct(p)
    n = int(text[:-1])
    return ("an " if n in (8, 11, 18) or 80 <= n <= 89 else "a ") + text


def runs_text(n: float | None) -> str:
    k = 1 if n is None else max(1, round(n))
    return f"{k} similar run" if k == 1 else f"{k} similar runs"


def target_text(rule: AcceptanceRule | None) -> str | None:
    """"runtime_s <= 200" for a score rule, else None."""
    if rule is None or rule.score is None:
        return None
    op = ">=" if rule.score.better == "higher" else "<="
    return f"{rule.score.name} {op} {rule.score.target:g}"


def chance_phrase(rule: AcceptanceRule | None) -> str:
    t = target_text(rule)
    return f"chance of reaching {t}" if t else "chance of an accepted result"


def success_phrase(rule: AcceptanceRule | None) -> str:
    t = target_text(rule)
    return f"reaching {t}" if t else "an accepted result"


def saving(gain: dict[str, Any]) -> str:
    """The expected saving per future similar run, in dollars (D96)."""
    return usd(float(gain.get("usd") or 0.0))


REFERENCE_NAMES = {"usual": "your usual workflow", "best_recorded": "your best recorded workflow",
                   "default": "the default workflow"}


def goal_sentence(goal_level: int | None, goal_label: str, goal_pred: Prediction, *, usual_id: str,
                  goal_id: str, note: str | None, rule: AcceptanceRule | None, reference: str = "usual") -> str:
    if goal_level is None:
        head = f"{note[0].upper()}{note[1:]}. " if note else ""
        if goal_id == usual_id:
            which = "your usual workflow" if reference == "usual" else "the reference workflow"
            return head + (f"Your goal is the lowest expected cost to {success_phrase(rule)}, "
                           f"which is {which}.")
        return head + (f"Your goal is the lowest expected cost to {success_phrase(rule)}: {goal_label}, "
                       "about " + noted(f"{usd(goal_pred.ell.usd.mean)} per accepted result", goal_pred.ell.usd)
                       + ".")
    at = "at about " + noted(usd(goal_pred.cost.usd.mean), goal_pred.cost.usd)
    if note:
        return f"{note[0].upper()}{note[1:]}: {goal_label} {at}."
    return f"Your goal is the {goal_level}% row: {goal_label} {at}."


def compose(*, usual: Configuration, usual_pred: Prediction, goal: Configuration, goal_pred: Prediction,
            goal_level: int | None, goal_note: str | None, exploration: Exploration, rule: AcceptanceRule | None,
            label: Callable[[Configuration], str] | None = None, reference: str = "usual") -> str:
    """The one-paragraph message for the user; `label` names configurations (the shown labels, D89).
    `reference` is the baseline's kind: `usual`, `best_recorded` or `default`."""
    label_of = label or Configuration.label
    chance = noted(f"{a_pct(usual_pred.p_success.mean)} {chance_phrase(rule)}", usual_pred.p_success)
    if reference == "usual":
        first = f"Your usual workflow ({label_of(usual)}) has {chance} at about {run_cost(usual_pred.cost)}."
    else:
        first = (f"You have no usual workflow for this task; the reference is {REFERENCE_NAMES[reference]} "
                 f"({label_of(usual)}), with {chance} at about {run_cost(usual_pred.cost)}.")
    parts = [first,
             goal_sentence(goal_level, label_of(goal), goal_pred, usual_id=usual.id, goal_id=goal.id,
                           note=goal_note, rule=rule, reference=reference)]
    bv, mg = exploration.best_value, exploration.max_gain
    if bv.state == "none":
        parts.append("No workflow is worth trying alongside it right now: none gains more than 1 percent "
                     "of your goal's expected cost on future similar runs.")
        return " ".join(parts)
    p = bv.pick
    name = label_of(p.candidate.config)
    if bv.state == "paused":
        parts.append(f"Trying {name} alongside it would cost {run_cost(p.price)} now, but that passes your budget "
                     f"cap, so it is paused.")
    else:
        parts.append(f"Trying {name} alongside it costs {run_cost(p.price)} now. There is {a_pct(p.p_beats_goal)} "
                     f"chance it beats the recommended pick. Trying it once is expected to save about "
                     f"{saving(p.gain_per_run)} on each future similar run, so it pays for itself after about "
                     f"{runs_text(p.payback_runs)}.")
    text = " ".join(parts)
    return text + max_gain_sentence(mg, label_of)


def max_gain_sentence(mg: Slot, label_of: Callable[[Configuration], str] = Configuration.label) -> str:
    if mg.state in ("same_as", "none") or mg.pick is None:
        return ""
    p = mg.pick
    label = label_of(p.candidate.config)
    if mg.state == "paused":
        return (f" The option with the biggest gain is {label}, but its price of "
                f"{noted(usd(p.price.usd.mean), p.price.usd)} passes your budget cap, so it is paused.")
    payback = 1 if p.payback_runs is None else max(1, round(p.payback_runs))
    return (f" The option with the biggest gain is {label}: it costs {run_cost(p.price)} now, has "
            f"{a_pct(p.p_beats_goal)} chance to beat the recommended pick, is expected to save about "
            f"{saving(p.gain_per_run)} per future similar run, and pays for itself "
            f"after about {payback} {'run' if payback == 1 else 'runs'}.")
