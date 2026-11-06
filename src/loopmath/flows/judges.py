"""The judges fit, the combined quality of a run, and its unpooled estimate.

For run r and judge j the score on [0, 1] is `y_rj = q_r + b_j + e_rj`, with `e_rj ~ N(0, s_j^2)`. The run's
quality is `q_r = m_a + v_i + u_r`: its arm's mean, a fixed level for its input, and run noise
`u_r ~ N(0, s_run^2)`. So `s_run` is net of the input, as a pair on one input sees it. Arm means, input levels
and free biases have wide normal priors (sd LEVEL_SD).

The scale is anchored in one of two ways:
- once CALIBRATED_AT runs carry a person's score, a person anchors it (b = 0);
- before that, every judge's bias is free and the biases sum to 0, so they are relative to the judges' mean.

A person's noise has a half-normal prior of scale PERSON_PRIOR. Every other noise, and `s_run`, is log-uniform on
NOISE_RANGE.

Given the noises, the posterior of the levels and biases is normal in closed form. So the fit weighs a grid of the
noises by marginal likelihood times prior, as `screen/model.py` does. The grid has one dimension for `s_run` and
one per judge with a reading. Each later grid, up to PASSES in all, covers on every axis the bins that carry that
axis's weight, so the last one's steps are about a posterior sd or finer. Each run's scores have
covariance `s_run^2 11' + diag(s_j^2)`, so the sums the posterior needs run over reading patterns (the sets of
judges that scored a run), with counts per arm and input.

The smallest grid has 2 points per axis, so a fit takes at most `MAX_SCORED_JUDGES` judges with readings (the
person included): the largest J whose 2^(J+1) points fit GRID, 13 at 20000. With more, the fit is refused before
any grid is built. Its `refused` is `too-many-judges:<n>`, it holds no values, and every reader reads no score for
its runs. Declared judges with no reading do not count.

An input with one run takes that run's whole level, up to the wide prior. So the run tells next to nothing about
`s_run` or the arm means. Its judges' differences still inform the biases and noises, and its q is close to the
noise-weighted mean of its bias-corrected readings.

The fit is a dict with the keys `FIT_KEYS`:
- `judges`: a list of dicts with the keys `JUDGE_KEYS`, in the flow's order. `bias` and `noise` are posterior
  means, and `bias_90` and `noise_90` are `[lo, hi]` (5% and 95%).
  - `catch_n`: on runs a person and the judge both scored, the runs the person put below the rule's score target.
  - `catch_rate`: the share of those the judge put below it too, on raw scores. It is None while the fit is
    uncalibrated, when `catch_n` is 0, and for a person.
- `s_run`: the fitted run noise, or None.

Extra keys:
- `s_run_90`, `score` (the score name), `anchor` (`person` or `judges' mean`) and `n_runs`;
- per judge, `n` (runs scored) and `bias_sd`;
- `runs`: each fitted run's combined quality `q` and sd, its unpooled estimate `unpooled` and `unpooled_sd`, and
  `key`, a digest of its readings; `combined_quality` returns `q` and sd for that run, `unpooled_quality` the
  unpooled pair;
- `levels`: per arm and input, the posterior mean and sd of `m_a + v_i`;
- `refused`: None, or `too-many-judges:<n>` for a fit refused for its number of judges.

A reading is the newest finite score signal under the judge's score name whose `source.ref` is the judge's id:
newest by `observed_at`, then by id, so a corrected score written as a new signal is the one read. A linear
judge's score is read as a fraction of its range. `fit` takes the range from the flow; the signal may carry it
too, as `ext[JUDGE_EXT]["range"]`. A range is used only as two finite numbers of magnitude at most `RANGE_LIMIT`,
the low below the high; any other range reads as none. Any other score is read as the fraction it is. A judge whose
newest score lies outside its range has no reading: a linear score outside its range, any other outside [0, 1]; a
bound is in range. A score under the name from a judge or a person (`source.kind` judge or user) is no reading,
yet the run counts as judged, so it is never read raw (`judged`), when its `source.ref` is not a string
(unattributed, whatever its value), when its number is not finite as a float (an int past the float range, an
infinity or NaN), or when its value is not a number (none, null, a bool, text, a list or an object). So nothing a
document holds makes a reader raise.

The combined quality is pooled: q's posterior under the arm, input and run noise model, so it is pulled toward its
arm and input level. It is what `flow status` shows. Every reader that treats runs as observations reads the
unpooled estimate instead, so the runs are not pooled twice. At each grid point it is the run's bias-corrected
readings weighted by 1/s_j^2, with a flat prior on q (no level, no `s_run`): mean `sum_j a_j (y_j - b_j)` with
`a_j = (1/s_j^2) / D` and `D = sum_j 1/s_j^2`, and variance `1/D + a' B a`, with B the biases' posterior covariance
there. These are mixed over the grid's weights. A calibrated fit's person has bias 0, so the estimate is on the
person's scale; below `CALIBRATED_AT` the biases sum to 0 and it is on the judges' mean's.

`current_quality` serves every reader of stored runs: `belief.outcome`, the belief fit, `compare` and `store`.
Those readers have runs but no flow document. So it takes the judges from the runs' signals: `source.ref` names the
judge, and `source.kind` `user` marks a person. A linear score with no range on its signal, or a score outside its
range, is skipped, never read raw. Runs are admitted as `belief.fit.collect_rows` admits them, and only flow runs
are fitted. A caller that holds a fit from the flow (`fit`) reads the same value with `unpooled_quality`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from ..belief.outcome import parse_ts, signal_list
from .records import Flow, judge_scale, judge_score

FIT_KEYS = ("calibrated", "n_person", "judges", "s_run")
JUDGE_KEYS = ("id", "kind", "bias", "bias_90", "noise", "noise_90", "catch_rate", "catch_n")

JUDGE_EXT = "dev.loopmath.judge"  # the `ext` member of a linear judge's score signal that holds its `range`
RANGE_LIMIT = 2 ** 53  # the largest magnitude of a range bound that is read, as E214 holds it: exact as a float
UNRANGED = "judge score linear without a range"  # the belief fit's drop count for a skipped linear reading
UNATTRIBUTED = "judge score with no judge named"  # the belief fit's drop count for a score with no string ref
OUTSIDE_RANGE = "judge score outside its range"  # the belief fit's drop count for a reading outside its range
NONFINITE = "judge score not finite as a float"  # the belief fit's drop count for a score no finite float holds
NONNUMERIC = "judge score not a number"  # the belief fit's drop count for a score whose value is not a number
TOO_MANY = "too-many-judges"  # a refused fit's `refused`, as `too-many-judges:<n>`, and the reader's reason
REFUSED = "judge scores of a fit refused for too many judges"  # the belief fit's drop count for those runs
DROP_COUNTS = {"unranged": UNRANGED, "unattributed": UNATTRIBUTED, "outside-range": OUTSIDE_RANGE,
               "nonfinite": NONFINITE, "nonnumeric": NONNUMERIC}  # a reason's name (`Judged.reasons`) to its count
CALIBRATED_AT = 6  # runs with a person's score before a person anchors the scale
PERSON_PRIOR = 0.04  # the half-normal scale of a person's noise
PERSON_RANGE = (0.002, 0.3)
NOISE_RANGE = (0.005, 0.5)  # log-uniform: every other judge's noise, and s_run, on scores in [0, 1]
LEVEL_SD = 10.0  # the wide normal prior of arm means, input levels and free biases
GRID = 20000  # grid points per pass, at most
MAX_SCORED_JUDGES = GRID.bit_length() - 2  # the most judges with readings whose smallest grid, 2^(J+1), fits GRID
MAX_POINTS = 24  # per dimension
PASSES = 4  # grids: the first on the prior ranges, each next on the box that carries the last one's weight
BOX = 1e-6  # an axis's next box: its bins with at least this share of the largest marginal weight, a step wider
KEEP = 1e-9  # a grid point with less weight than this share of the largest is dropped, as screen does
CHUNK = 2048  # grid points per batch for the likelihood
TAIL = 0.05  # each tail of a 90% interval


class Unfitted(tuple):
    """`(None, None)`, no combined quality, with the reason a reader gives for it (`too-many-judges:<n>`)."""

    reason: str

    def __new__(cls, reason: str):
        value = super().__new__(cls, (None, None))
        value.reason = reason
        return value

    def __getnewargs__(self):
        return (self.reason,)


class Judged(NamedTuple):
    """A run's judge scores under one name, as the reader takes them (`judged`). The run is judged when any field
    is non-empty; fields are read by name. The last field is a tuple with a default, so a value built from the
    first five equals `judged`'s when no judge wrote a non-number."""

    usable: list[str]  # the tiers of the usable readings, one per judge
    unranged: list[str]  # judges whose newest reading is a linear score with no range on its signal
    unattributed: list[str]  # signal ids (`?` for none) of the scores with no string `source.ref` (`_unattributed`)
    outside_range: list[str]  # judges whose newest reading lies outside its range (`_fraction`)
    nonfinite: list[str]  # judges with a score whose number is not finite as a float (`_nonfinite`)
    nonnumeric: tuple[str, ...] = ()  # judges with a score whose value is not a number (`_nonnumeric`)

    @property
    def reasons(self) -> list[str]:
        """The reasons `belief.outcome` gives for these scores, in field order."""
        return ([f"unranged:{j}" for j in self.unranged] + [f"unattributed:{s}" for s in self.unattributed]
                + [f"outside-range:{j}" for j in self.outside_range] + [f"nonfinite:{j}" for j in self.nonfinite]
                + [f"nonnumeric:{j}" for j in self.nonnumeric])


@dataclass(frozen=True)
class _Judge:
    id: str
    kind: str | None
    name: str | None  # the score name it writes
    range: tuple[float, float] | None = None  # a linear judge's range, from the flow

    @property
    def person(self) -> bool:
        return self.kind == "person"


@dataclass(frozen=True)
class _Run:
    id: str
    arm: str
    input: str | None
    values: dict[str, float]  # judge id to a fraction


def fit(flow: Flow, docs: Sequence[dict]) -> dict:
    """The judges fit from the score signals of the flow's runs (`run.flow.id` is the flow's, any digest)."""
    judges = _flow_judges(flow)
    rule = flow.acceptance_rule if isinstance(flow.acceptance_rule, dict) else {}
    score = rule.get("score") if isinstance(rule.get("score"), dict) else {}
    target = (float(score["target"]), score.get("better") or "higher") if _finite(score.get("target")) else None
    return _fit(judges, _runs(docs, judges, flow.id), target, score.get("name") or flow.objective.score)


def combined_quality(flow: Flow, doc: dict, fit: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """One run's combined quality and its sd under `fit`."""
    return _combined(_flow_judges(flow), doc, fit)


def unpooled_quality(flow: Flow, doc: dict, fit: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """One run's unpooled estimate and its sd under `fit` (from `fit(flow, docs)`): the value `current_quality`
    gives, for a caller that already holds the fit."""
    return _unpooled(_flow_judges(flow), doc, fit)


def judge_warning(flow: Flow) -> str | None:
    """The warning `flow check` gives when `flow` declares more judges than a fit takes, else None."""
    n = len(flow.judges or ())
    if n <= MAX_SCORED_JUDGES:
        return None
    return (f"flow {flow.id!r} declares {n} judges; the judges fit takes at most {MAX_SCORED_JUDGES} with "
            f"readings, so runs scored by more read no score ({TOO_MANY}:<n>)")


def current_quality(docs: Iterable[dict | tuple[str | None, dict]], *,
                    without: tuple[str, ...] = ()) -> dict[str, tuple[float, float | None]]:
    """Run id to the unpooled estimate and its sd of each admitted flow run (one with `run.flow`) that holds a
    usable judge reading under its rule's score name, from one fit per flow and score name.

    `docs` and `without` are as `belief.fit.collect_rows` takes them, and runs are admitted as it admits them.
    Only the admitted flow runs are fitted."""
    from ..belief.fit import admitted

    return quality_map([(doc, parsed.rule) for _, doc, parsed in admitted(docs, without=without)
                        if _flow_ref(doc) is not None])


def quality_map(items: Iterable[tuple[dict, Any]]) -> dict[str, tuple[float, float | None]]:
    """Run id to the unpooled estimate and its sd for admitted `(document, rule)` pairs.

    The flow runs among them are grouped by flow id and the rule's score name, with one fit per group and the
    judges taken from the signals."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for doc, rule in items:
        ref, score = _flow_ref(doc), getattr(rule, "score", None)
        if ref is not None and isinstance(ref.get("id"), str) and score is not None:
            groups.setdefault((ref["id"], score.name), []).append(doc)
    out: dict[str, tuple[float, float | None]] = {}
    for (_, name), group in sorted(groups.items()):
        judges = signal_judges(group, name)
        runs = _runs(group, judges, None)
        got = _fit(judges, runs, None, name)
        if got["refused"]:
            out.update((r.id, Unfitted(got["refused"])) for r in runs)
        for run_id, v in got["runs"].items():
            out[run_id] = (v["unpooled"], v["unpooled_sd"])
    return out


def signal_judges(docs: Iterable[dict], name: str) -> list[_Judge]:
    """The judges that wrote score signals named `name` in `docs`, by id. A judge is a person when its
    `source.kind` is `user`, else a model judge."""
    kinds: dict[str, str] = {}
    for doc in docs:
        for s in _judge_scores(doc, name):
            kinds.setdefault(s["source"]["ref"], "person" if s["source"]["kind"] == "user" else "model")
    return [_Judge(ref, kind, name) for ref, kind in sorted(kinds.items())]


def judged(doc: dict, name: str) -> Judged:
    """The run's judge scores under `name`, as the reader takes them (`Judged`), with the ranges on the signals.
    Every field is empty when no judge scored the run under `name`."""
    tiers, unranged, outside = [], [], []
    for judge in signal_judges([doc], name):
        last = _newest([s for s in _judge_scores(doc, name) if s["source"]["ref"] == judge.id])
        why = _fraction(last, None)[1]
        if why == "unranged":
            unranged.append(judge.id)
        elif why == "outside-range":
            outside.append(judge.id)
        else:
            tiers.append(last.get("tier"))
    return Judged(tiers, unranged, _unattributed(doc, name), outside, _nonfinite(doc, name),
                  tuple(_nonnumeric(doc, name)))


# Readings


def _flow_ref(doc: Any) -> dict | None:
    run = doc.get("run") if isinstance(doc, dict) else None
    ref = run.get("flow") if isinstance(run, dict) else None
    return ref if isinstance(ref, dict) else None


def _flow_judges(flow: Flow) -> list[_Judge]:
    return [_Judge(j.id, j.kind, judge_score(j, flow.objective),
                   _range(j.range) if judge_scale(j) == "linear" else None) for j in flow.judges or ()]


def _range(value: Any) -> tuple[float, float] | None:
    """A linear range as (low, high) floats; None unless it is two finite numbers of magnitude at most
    `RANGE_LIMIT` whose floats have the low below the high."""
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_finite(v) and abs(v) <= RANGE_LIMIT for v in value)):
        return None
    low, high = float(value[0]), float(value[1])
    return (low, high) if high > low else None


def _signal_range(sig: dict) -> tuple[float, float] | None:
    ext = sig.get("ext") if isinstance(sig.get("ext"), dict) else {}
    member = ext.get(JUDGE_EXT) if isinstance(ext.get(JUDGE_EXT), dict) else {}
    return _range(member.get("range"))


def _judge_scores(doc: dict, name: str | None) -> list[dict]:
    """The run's finite score signals under `name` from a judge or a person (`source.kind` judge or user)."""
    run = doc.get("run") if isinstance(doc, dict) and isinstance(doc.get("run"), dict) else {}
    return [s for s in signal_list(run) if isinstance(s, dict) and s.get("kind") == "score"
            and s.get("name") == name and _finite(s.get("value")) and isinstance(s.get("source"), dict)
            and s["source"].get("kind") in ("judge", "user") and isinstance(s["source"].get("ref"), str)]


def _unattributed(doc: dict, name: str) -> list[str]:
    """The ids (`?` for none) of the run's score signals under `name` from a judge or a person whose `source.ref`
    is not a string, whatever their value: no judge can be named for them, so none is a reading."""
    run = doc.get("run") if isinstance(doc, dict) and isinstance(doc.get("run"), dict) else {}
    return [s["id"] if isinstance(s.get("id"), str) else "?" for s in signal_list(run)
            if isinstance(s, dict) and s.get("kind") == "score" and s.get("name") == name
            and isinstance(s.get("source"), dict) and s["source"].get("kind") in ("judge", "user")
            and not isinstance(s["source"].get("ref"), str)]


def _nonfinite(doc: dict, name: str) -> list[str]:
    """The judges (a string `source.ref`, `source.kind` judge or user) with a score under `name` whose number is not
    finite as a float: an int past the float range, an infinity or NaN. Such a score is no reading."""
    run = doc.get("run") if isinstance(doc, dict) and isinstance(doc.get("run"), dict) else {}
    return sorted({s["source"]["ref"] for s in signal_list(run)
                   if isinstance(s, dict) and s.get("kind") == "score" and s.get("name") == name
                   and isinstance(s.get("value"), (int, float)) and not isinstance(s.get("value"), bool)
                   and not _finite(s.get("value")) and isinstance(s.get("source"), dict)
                   and s["source"].get("kind") in ("judge", "user") and isinstance(s["source"].get("ref"), str)})


def _nonnumeric(doc: dict, name: str) -> list[str]:
    """The judges (a string `source.ref`, `source.kind` judge or user) with a score under `name` whose value is not
    a number: none, null, a bool, text, a list or an object. Such a score is no reading."""
    run = doc.get("run") if isinstance(doc, dict) and isinstance(doc.get("run"), dict) else {}
    return sorted({s["source"]["ref"] for s in signal_list(run)
                   if isinstance(s, dict) and s.get("kind") == "score" and s.get("name") == name
                   and (isinstance(s.get("value"), bool) or not isinstance(s.get("value"), (int, float)))
                   and isinstance(s.get("source"), dict) and s["source"].get("kind") in ("judge", "user")
                   and isinstance(s["source"].get("ref"), str)})


def _newest(signals: list[dict]) -> dict | None:
    """The one reading among a judge's scores under one name: the newest by `observed_at`, then by id. A missing
    time sorts first; full ties keep document order."""
    if not signals:
        return None
    far_past = datetime.min.replace(tzinfo=timezone.utc)
    return max(enumerate(signals), key=lambda p: (parse_ts(p[1].get("observed_at")) or far_past,
                                                   str(p[1].get("id") or ""), p[0]))[1]


def _readings(doc: dict, judges: Sequence[_Judge]) -> dict[str, float]:
    """Judge id to its reading in `doc`, as a fraction: the newest finite score under the judge's name from that
    judge (`_newest`), read by `_fraction` with the judge's range from the flow."""
    values: dict[str, float] = {}
    for judge in judges:
        last = _newest([s for s in _judge_scores(doc, judge.name) if s["source"]["ref"] == judge.id])
        if last is None:
            continue
        value = _fraction(last, judge.range)[0]
        if value is not None:
            values[judge.id] = value
    return values


def _fraction(last: dict, span: tuple[float, float] | None) -> tuple[float | None, str | None]:
    """A judge's newest finite score as a fraction and None, or None and why it is no reading. It is mapped by
    `span` (the flow's range), else, when linear, by the range on its signal (`unranged` with none); any other
    score is the fraction it is. A score outside its range, or outside [0, 1] when not mapped, is `outside-range`;
    a bound is in range. The score is compared as stored, so an int one past a bound never rounds onto it."""
    value = last["value"]
    if span is None and last.get("scale") == "linear":
        span = _signal_range(last)
        if span is None:
            return None, "unranged"
    low, high = span or (0.0, 1.0)
    if not low <= value <= high:
        return None, "outside-range"
    value = float(value)
    return (value if span is None else (value - low) / (high - low)), None


def _runs(docs: Iterable[dict], judges: Sequence[_Judge], flow_id: str | None) -> list[_Run]:
    """The runs of `flow_id` (any flow when None) in `docs` with a reading, by run id; the first copy of an id
    counts."""
    runs: dict[str, _Run] = {}
    for doc in docs:
        ref = _flow_ref(doc)
        run = doc["run"] if ref is not None else {}
        if ref is None or (flow_id is not None and ref.get("id") != flow_id) or not isinstance(run.get("id"), str) \
                or run["id"] in runs:
            continue
        values = _readings(doc, judges)
        if not values:
            continue
        task = run.get("task") if isinstance(run.get("task"), dict) else {}
        arm = ref.get("arm") if isinstance(ref.get("arm"), str) else "base"
        runs[run["id"]] = _Run(run["id"], arm, task.get("id") if isinstance(task.get("id"), str) else None, values)
    return [runs[k] for k in sorted(runs)]


def _key(values: Mapping[str, float]) -> str:
    return hashlib.sha256(json.dumps(sorted(values.items())).encode()).hexdigest()[:12]


# The fit


def _fit(judges: Sequence[_Judge], runs: Sequence[_Run], target: tuple[float, str] | None, name: str | None, *,
         inputs: bool = True) -> dict:
    """The fit of `runs` under `judges`. `inputs` False leaves the input levels out (for comparison only)."""
    n_person = sum(1 for r in runs if any(j.person and j.id in r.values for j in judges))
    calibrated = n_person >= CALIBRATED_AT
    scored = sorted((j for j in judges if any(j.id in r.values for r in runs)), key=lambda j: j.id)
    refused = f"{TOO_MANY}:{len(scored)}" if len(scored) > MAX_SCORED_JUDGES else None
    post = _posterior(scored, runs, calibrated, inputs) if scored and refused is None else None
    rows = []
    for judge in judges:
        row: dict[str, Any] = {"id": judge.id, "kind": judge.kind, "bias": None, "bias_90": None, "noise": None,
                               "noise_90": None, "catch_rate": None, "catch_n": None,
                               "n": sum(1 for r in runs if judge.id in r.values), "bias_sd": None}
        if post is not None and judge.id in post["judges"]:
            row.update(post["judges"][judge.id])
        if target is not None and not judge.person:
            caught, n = _catch(judges, judge, runs, target)
            row["catch_n"] = n
            row["catch_rate"] = caught / n if calibrated and n else None
        rows.append(row)
    empty = post is None
    return {"calibrated": calibrated, "n_person": n_person, "judges": rows,
            "s_run": None if empty else post["s_run"], "s_run_90": None if empty else post["s_run_90"],
            "score": name, "anchor": "person" if calibrated else "judges' mean", "n_runs": len(runs),
            "runs": {} if empty else post["runs"], "levels": [] if empty else post["levels"], "refused": refused}


def _fails(value: float, target: tuple[float, str]) -> bool:
    t, better = target
    return value > t if better == "lower" else value < t


def _catch(judges: Sequence[_Judge], judge: _Judge, runs: Sequence[_Run], target: tuple[float, str]):
    persons = [j.id for j in judges if j.person]
    caught = n = 0
    for r in runs:
        person = next((r.values[p] for p in persons if p in r.values), None)
        if person is None or judge.id not in r.values or not _fails(person, target):
            continue
        n += 1
        caught += _fails(r.values[judge.id], target)
    return caught, n


def _centred(J: int) -> np.ndarray:
    """An orthonormal basis of the biases that sum to 0 (Helmert columns), so the prior treats judges alike."""
    M = np.zeros((J, max(J - 1, 0)))
    for k in range(1, J):
        M[:k, k - 1] = 1.0 / math.sqrt(k * (k + 1))
        M[k, k - 1] = -k / math.sqrt(k * (k + 1))
    return M


class _Data:
    """The sums the posterior needs, by reading pattern; parameters are (arm means, input levels, free biases),
    with the biases `M @ free`."""

    def __init__(self, scored: Sequence[_Judge], runs: Sequence[_Run], calibrated: bool, inputs: bool):
        J = len(scored)
        index = {j.id: k for k, j in enumerate(scored)}
        self.arms = sorted({r.arm for r in runs})
        self.inputs = sorted({r.input for r in runs if r.input is not None}) if inputs else []
        pattern_of = [tuple(sorted(index[j] for j in r.values)) for r in runs]
        patterns = sorted(set(pattern_of))
        self.mask = np.zeros((len(patterns), J))
        for p, pat in enumerate(patterns):
            self.mask[p, list(pat)] = 1.0
        if calibrated:
            self.M = np.eye(J)[:, [k for k, j in enumerate(scored) if not j.person]]
        else:
            self.M = _centred(J)
        A, I, P = len(self.arms), len(self.inputs), len(patterns)
        self.A, self.I, self.K = A, I, self.M.shape[1]
        self.size = A + I + self.K
        self.Y = np.zeros((len(runs), J))
        for k, r in enumerate(runs):
            for j, v in r.values.items():
                self.Y[k, index[j]] = v
        number = {pat: p for p, pat in enumerate(patterns)}
        arm_of = {a: k for k, a in enumerate(self.arms)}
        input_of = {i: k for k, i in enumerate(self.inputs)}
        self.pidx = np.array([number[pat] for pat in pattern_of], dtype=int)
        self.aidx = np.array([arm_of[r.arm] for r in runs], dtype=int)
        self.iidx = np.array([input_of.get(r.input, -1) for r in runs], dtype=int)
        self.has_i = (self.iidx >= 0).astype(float)
        sel = self.iidx >= 0
        self.n_p = np.bincount(self.pidx, minlength=P).astype(float)
        self.N_pa = np.zeros((P, A))
        np.add.at(self.N_pa, (self.pidx, self.aidx), 1.0)
        self.N_pi = np.zeros((P, I))
        np.add.at(self.N_pi, (self.pidx[sel], self.iidx[sel]), 1.0)
        self.N_pai = np.zeros((P, A, I))
        np.add.at(self.N_pai, (self.pidx[sel], self.aidx[sel], self.iidx[sel]), 1.0)
        self.T_pa = np.zeros((P, A, J))
        np.add.at(self.T_pa, (self.pidx, self.aidx), self.Y)
        self.T_pi = np.zeros((P, I, J))
        np.add.at(self.T_pi, (self.pidx[sel], self.iidx[sel]), self.Y[sel])
        self.T_p = np.zeros((P, J))
        np.add.at(self.T_p, self.pidx, self.Y)
        self.Q_p = np.zeros((P, J, J))
        np.add.at(self.Q_p, self.pidx, self.Y[:, :, None] * self.Y[:, None, :])
        self.n_scores = float(self.mask[self.pidx].sum())


def _evaluate(data: _Data, points: np.ndarray, posterior: bool = False):
    """At each grid point (s_run, s_1 .. s_J): the log marginal likelihood; with `posterior`, also the posterior
    means and covariances of the parameters."""
    A, I, P, M = data.A, data.I, data.size, data.M
    J = data.mask.shape[1]
    r2 = points[:, 0] ** 2
    S2 = points[:, 1:] ** 2
    dp = (1.0 / S2)[:, None, :] * data.mask[None]  # (G, patterns, J): D^-1 on each pattern's judges
    Sp = dp.sum(-1)
    kap = 1.0 / (1.0 / r2[:, None] + Sp)
    u = dp * (1.0 - kap * Sp)[..., None]  # V^-1 1, by Sherman-Morrison
    w = u.sum(-1)  # 1' V^-1 1
    Vinv = dp[..., :, None] * np.eye(J) - kap[..., None, None] * dp[..., :, None] * dp[..., None, :]
    Amat = np.zeros((len(points), P, P))
    ia, ii = np.arange(A), A + np.arange(I)
    Amat[:, ia, ia] = w @ data.N_pa
    Amat[:, ii, ii] = w @ data.N_pi
    ai = np.einsum("gp,pai->gai", w, data.N_pai)
    Amat[:, :A, A:A + I] = ai
    Amat[:, A:A + I, :A] = ai.transpose(0, 2, 1)
    ag = np.einsum("gpj,pa,jc->gac", u, data.N_pa, M)
    Amat[:, :A, A + I:] = ag
    Amat[:, A + I:, :A] = ag.transpose(0, 2, 1)
    vg = np.einsum("gpj,pi,jc->gic", u, data.N_pi, M)
    Amat[:, A:A + I, A + I:] = vg
    Amat[:, A + I:, A:A + I] = vg.transpose(0, 2, 1)
    Amat[:, A + I:, A + I:] = np.einsum("jc,gjl,ld->gcd", M, np.einsum("p,gpjl->gjl", data.n_p, Vinv), M)
    Amat[:, np.arange(P), np.arange(P)] += 1.0 / LEVEL_SD ** 2
    h = np.concatenate([np.einsum("gpj,paj->ga", u, data.T_pa), np.einsum("gpj,pij->gi", u, data.T_pi),
                        np.einsum("gpjl,pl->gj", Vinv, data.T_p) @ M], axis=1)
    quad = np.einsum("gpjl,pjl->g", Vinv, data.Q_p)
    logdet_v = (data.mask[None] * np.log(S2)[:, None, :]).sum(-1) @ data.n_p + np.log1p(r2[:, None] * Sp) @ data.n_p
    chol = np.linalg.cholesky(Amat)
    logdet_a = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(-1)
    theta = np.linalg.solve(Amat, h[..., None])[..., 0]
    loglik = -0.5 * (data.n_scores * math.log(2 * math.pi) + logdet_v + P * math.log(LEVEL_SD ** 2) + logdet_a
                     + quad - (h * theta).sum(-1))
    if not posterior:
        return loglik
    return loglik, theta, np.linalg.inv(Amat)


def _axes(scored: Sequence[_Judge], n: int, boxes: Sequence[tuple[float, float]] | None = None) -> list[np.ndarray]:
    ranges = boxes or [NOISE_RANGE] + [PERSON_RANGE if j.person else NOISE_RANGE for j in scored]
    return [np.geomspace(lo, hi, n) for lo, hi in ranges]


def _logprior(scored: Sequence[_Judge], points: np.ndarray) -> np.ndarray:
    """The log prior of each point of a grid with log-spaced axes. Log-uniform is constant there; a person's
    half-normal density is multiplied by s, the log grid's Jacobian."""
    out = np.zeros(len(points))
    for k, judge in enumerate(scored, start=1):
        if judge.person:
            out += -0.5 * (points[:, k] / PERSON_PRIOR) ** 2 + np.log(points[:, k])
    return out


def _weights(data: _Data, scored: Sequence[_Judge], axes: Sequence[np.ndarray]):
    points = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)
    logw = np.concatenate([_evaluate(data, points[i:i + CHUNK]) for i in range(0, len(points), CHUNK)])
    return points, logw + _logprior(scored, points)


def _zoomed(data: _Data, scored: Sequence[_Judge]):
    """The points per axis, and the last grid's axes, points and log weights."""
    n = max(2, min(MAX_POINTS, int(GRID ** (1.0 / (len(scored) + 1)) + 1e-9)))
    shape = [n] * (len(scored) + 1)
    axes = _axes(scored, n)
    points, logw = _weights(data, scored, axes)
    for _ in range(PASSES - 1):
        w = np.exp(logw - logw.max())
        boxes = []
        for ax, i in zip(axes, np.unravel_index(np.arange(len(points)), shape)):
            on = np.flatnonzero((m := np.bincount(i, weights=w, minlength=n)) >= BOX * m.max())
            boxes.append((ax[max(int(on.min()) - 1, 0)], ax[min(int(on.max()) + 1, n - 1)]))
        zoomed = _axes(scored, n, boxes)
        if all(np.array_equal(a, b) for a, b in zip(zoomed, axes)):
            break
        axes = zoomed
        points, logw = _weights(data, scored, axes)
    return n, axes, points, logw


def _posterior(scored: Sequence[_Judge], runs: Sequence[_Run], calibrated: bool, inputs: bool) -> dict:
    data = _Data(scored, runs, calibrated, inputs)
    n, axes, points, logw = _zoomed(data, scored)
    shape = [n] * (len(scored) + 1)
    kept = np.flatnonzero(logw >= logw.max() + math.log(KEEP))
    w = np.exp(logw[kept] - logw.max())
    w = w / w.sum()
    pts = points[kept]
    marg = [np.bincount(i, weights=w, minlength=n) for i in np.unravel_index(kept, shape)]
    pairs = [(a, None) for a in range(data.A)] if not data.I else \
        [(a, i) for a in range(data.A) for i in range(data.I)]
    bias_mu, bias_var = [], []
    q1, q2 = np.zeros(len(runs)), np.zeros(len(runs))
    u1, u2 = np.zeros(len(runs)), np.zeros(len(runs))
    l1, l2 = np.zeros(len(pairs)), np.zeros(len(pairs))
    step = max(16, CHUNK * 50 // max(1, len(runs)))
    for i in range(0, len(pts), step):
        part, wi = pts[i:i + step], w[i:i + step]
        _, theta, cov = _evaluate(data, part, posterior=True)
        mu, var = _bias_moments(data, theta, cov)
        bias_mu.append(mu)
        bias_var.append(var)
        qm, qv = _q_moments(data, part, theta, cov)
        q1 += wi @ qm
        q2 += wi @ (qv + qm ** 2)
        um, uv = _u_moments(data, part, theta, cov)
        u1 += wi @ um
        u2 += wi @ (uv + um ** 2)
        lm, lv = _level_moments(data, pairs, theta, cov)
        l1 += wi @ lm
        l2 += wi @ (lv + lm ** 2)
    bias_mu, bias_var = np.concatenate(bias_mu), np.concatenate(bias_var)
    out: dict[str, Any] = {"s_run": float(w @ pts[:, 0]), "s_run_90": _grid_interval(axes[0], marg[0]),
                           "judges": {}, "runs": {}, "levels": []}
    for k, judge in enumerate(scored):
        anchored = calibrated and judge.person
        mean = float(w @ bias_mu[:, k])
        sd = math.sqrt(max(0.0, float(w @ (bias_var[:, k] + bias_mu[:, k] ** 2)) - mean * mean))
        out["judges"][judge.id] = {
            "bias": 0.0 if anchored else mean, "bias_sd": 0.0 if anchored else sd,
            "bias_90": [0.0, 0.0] if anchored else _mixture_interval(w, bias_mu[:, k], np.sqrt(bias_var[:, k])),
            "noise": float(w @ pts[:, k + 1]), "noise_90": _grid_interval(axes[k + 1], marg[k + 1])}
    sd = np.sqrt(np.maximum(0.0, q2 - q1 ** 2))
    usd = np.sqrt(np.maximum(0.0, u2 - u1 ** 2))
    for k, r in enumerate(runs):
        out["runs"][r.id] = {"q": float(q1[k]), "sd": float(sd[k]), "unpooled": float(u1[k]),
                             "unpooled_sd": float(usd[k]), "key": _key(r.values)}
    lsd = np.sqrt(np.maximum(0.0, l2 - l1 ** 2))
    for k, (a, i) in enumerate(pairs):
        out["levels"].append({"arm": data.arms[a], "input": None if i is None else data.inputs[i],
                              "mean": float(l1[k]), "sd": float(lsd[k])})
    return out


def _bias_moments(data: _Data, theta: np.ndarray, cov: np.ndarray):
    g = slice(data.A + data.I, data.size)
    mu = theta[:, g] @ data.M.T
    var = np.einsum("jc,gcd,jd->gj", data.M, cov[:, g, g], data.M)
    return mu, np.maximum(var, 0.0)


def _q_moments(data: _Data, points: np.ndarray, theta: np.ndarray, cov: np.ndarray):
    """Per grid point and run, the posterior mean and variance of q_r. Given the parameters, q_r is normal with
    precision 1/s_run^2 + the sum of 1/s_j^2 over its judges, and its mean is linear in them."""
    A, g = data.A, slice(data.A + data.I, data.size)
    r2 = points[:, 0] ** 2
    d = 1.0 / points[:, 1:] ** 2
    delta = data.mask[data.pidx][None] * d[:, None, :]  # (G, R, J)
    t2 = 1.0 / (1.0 / r2[:, None] + delta.sum(-1))
    ii = A + np.maximum(data.iidx, 0) if data.I else data.aidx
    has = data.has_i[None]
    bias = theta[:, g] @ data.M.T
    level = theta[:, data.aidx] + has * theta[:, ii]
    mean = t2 * ((data.Y[None] * d[:, None, :]).sum(-1) + level / r2[:, None] - (delta * bias[:, None, :]).sum(-1))
    level_var = cov[:, data.aidx, data.aidx] + has * (2 * cov[:, data.aidx, ii] + cov[:, ii, ii])
    md = delta @ data.M  # (G, R, K)
    cross = ((cov[:, data.aidx, g] + has[..., None] * cov[:, ii, g]) * md).sum(-1)
    spread = np.einsum("grc,gcd,grd->gr", md, cov[:, g, g], md)
    var = t2 + t2 ** 2 * (level_var / r2[:, None] ** 2 - 2 * cross / r2[:, None] + spread)
    return mean, np.maximum(var, 0.0)


def _u_moments(data: _Data, points: np.ndarray, theta: np.ndarray, cov: np.ndarray):
    """Per grid point and run, the mean and variance of the unpooled estimate: the run's readings less the biases,
    weighted by 1/s_j^2, with a flat prior on q_r. The variance is 1/sum(1/s_j^2) plus the biases' posterior
    variance through those weights."""
    g = slice(data.A + data.I, data.size)
    delta = data.mask[data.pidx][None] * (1.0 / points[:, 1:] ** 2)[:, None, :]  # (G, R, J)
    total = delta.sum(-1)
    a = delta / total[..., None]
    bias = theta[:, g] @ data.M.T
    mean = (a * (data.Y[None] - bias[:, None, :])).sum(-1)
    am = a @ data.M  # (G, R, K)
    var = 1.0 / total + np.einsum("grc,gcd,grd->gr", am, cov[:, g, g], am)
    return mean, np.maximum(var, 0.0)


def _level_moments(data: _Data, pairs: Sequence[tuple[int, int | None]], theta: np.ndarray, cov: np.ndarray):
    """Per grid point and (arm, input) pair, the posterior mean and variance of m_a + v_i."""
    a = np.array([p[0] for p in pairs], dtype=int)
    if not data.I:
        return theta[:, a], cov[:, a, a]
    i = data.A + np.array([p[1] for p in pairs], dtype=int)
    return theta[:, a] + theta[:, i], cov[:, a, a] + 2 * cov[:, a, i] + cov[:, i, i]


def _grid_interval(axis: np.ndarray, pmf: np.ndarray) -> list[float]:
    """The 5% and 95% points of a grid marginal: its CDF at the bins' edges in log s, joined by a monotone cubic
    (PCHIP), by bisection in the bin that holds each point. A straight line between the edges is up to 3e-3 off
    a 200000-point grid on 11 points per axis; the cubic is within 2e-3."""
    from scipy.interpolate import PchipInterpolator

    la = np.log(axis)
    half = (la[1] - la[0]) / 2
    edges = np.concatenate([[la[0] - half], (la[:-1] + la[1:]) / 2, [la[-1] + half]])
    cdf = np.concatenate([[0.0], np.cumsum(pmf)])
    cdf = cdf / cdf[-1]
    curve = PchipInterpolator(edges, cdf)
    out = []
    for q in (TAIL, 1 - TAIL):
        k = min(max(int(np.searchsorted(cdf, q, side="left")), 1), len(edges) - 1)
        lo, hi = float(edges[k - 1]), float(edges[k])
        for _ in range(60):
            mid = (lo + hi) / 2
            if float(curve(mid)) < q:
                lo = mid
            else:
                hi = mid
        out.append(math.exp((lo + hi) / 2))
    return out


def _mixture_interval(w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> list[float]:
    """The 5% and 95% points of the normal mixture with weights `w`, by bisection on its CDF."""
    from scipy.special import ndtr

    sd = np.maximum(sd, 1e-12)
    out = []
    for q in (TAIL, 1 - TAIL):
        lo, hi = float((mu - 9 * sd).min()), float((mu + 9 * sd).max())
        for _ in range(80):
            mid = (lo + hi) / 2
            if float(w @ ndtr((mid - mu) / sd)) < q:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2)
    return out


# One run


def _prior_noise(judge: _Judge) -> float:
    """The median of a judge's noise prior: half-normal for a person, log-uniform otherwise."""
    if judge.person:
        return PERSON_PRIOR * 0.6744897501960817
    return math.sqrt(NOISE_RANGE[0] * NOISE_RANGE[1])


def _combined(judges: Sequence[_Judge], doc: dict, fit: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """One run's combined quality and sd under `fit`.

    - A fitted run with the same readings: the stored value.
    - Another run with a reading: the normal posterior of q from its bias-corrected readings, each with variance
      noise^2 + bias_sd^2. The prior is the fit's level for the run's arm and input, with variance
      s_run^2 + the level's sd^2; with no such level, or no `s_run`, it is flat.
    - A judge the fit has no values for: bias 0 and the median of its noise prior.
    - No reading: (None, None). A refused fit: `Unfitted` with its reason."""
    values = _readings(doc, judges)
    if not values:
        return None, None
    if fit.get("refused"):
        return Unfitted(str(fit["refused"]))
    run = doc["run"]
    stored = fit.get("runs") if isinstance(fit.get("runs"), Mapping) else {}
    hit = stored.get(run.get("id")) if isinstance(run.get("id"), str) else None
    if isinstance(hit, Mapping) and hit.get("key") == _key(values):
        return hit["q"], hit["sd"]
    rows = {r.get("id"): r for r in fit.get("judges") or [] if isinstance(r, Mapping)}
    num = den = 0.0
    for judge in judges:
        if judge.id not in values:
            continue
        row = rows.get(judge.id) or {}
        bias = float(row["bias"]) if _finite(row.get("bias")) else 0.0
        noise = float(row["noise"]) if _finite(row.get("noise")) and row["noise"] > 0 else _prior_noise(judge)
        bias_sd = float(row["bias_sd"]) if _finite(row.get("bias_sd")) else 0.0
        weight = 1.0 / (noise * noise + bias_sd * bias_sd)
        num += weight * (values[judge.id] - bias)
        den += weight
    level = _level(fit, doc)
    if level is not None:
        weight = 1.0 / (float(fit["s_run"]) ** 2 + level[1] ** 2)
        num += weight * level[0]
        den += weight
    return num / den, 1.0 / math.sqrt(den)


def _unpooled(judges: Sequence[_Judge], doc: dict, fit: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """One run's unpooled estimate and sd under `fit`.

    - A fitted run with the same readings: the stored value.
    - Another run with a reading: its bias-corrected readings weighted by 1/noise^2 (the judges' posterior means),
      with variance 1/D plus each weight's square times its bias_sd^2. The fit holds no bias covariances, so they
      are left out off the fit.
    - A judge the fit has no values for: bias 0 and the median of its noise prior.
    - No reading: (None, None). A refused fit: `Unfitted` with its reason."""
    values = _readings(doc, judges)
    if not values:
        return None, None
    if fit.get("refused"):
        return Unfitted(str(fit["refused"]))
    run = doc["run"]
    stored = fit.get("runs") if isinstance(fit.get("runs"), Mapping) else {}
    hit = stored.get(run.get("id")) if isinstance(run.get("id"), str) else None
    if isinstance(hit, Mapping) and hit.get("key") == _key(values) and _finite(hit.get("unpooled")):
        return hit["unpooled"], hit["unpooled_sd"]
    rows = {r.get("id"): r for r in fit.get("judges") or [] if isinstance(r, Mapping)}
    parts = []
    for judge in judges:
        if judge.id not in values:
            continue
        row = rows.get(judge.id) or {}
        bias = float(row["bias"]) if _finite(row.get("bias")) else 0.0
        noise = float(row["noise"]) if _finite(row.get("noise")) and row["noise"] > 0 else _prior_noise(judge)
        bias_sd = float(row["bias_sd"]) if _finite(row.get("bias_sd")) else 0.0
        parts.append((1.0 / (noise * noise), values[judge.id] - bias, bias_sd))
    total = sum(d for d, _, _ in parts)
    mean = sum(d * y for d, y, _ in parts) / total
    var = 1.0 / total + sum((d / total) ** 2 * s * s for d, _, s in parts)
    return mean, math.sqrt(var)


def _level(fit: Mapping[str, Any], doc: dict) -> tuple[float, float] | None:
    """The fit's level (mean, sd) for the run's arm and input, when the fit has it and an `s_run`."""
    if not _finite(fit.get("s_run")) or not isinstance(fit.get("levels"), list):
        return None
    run = doc["run"]
    ref = _flow_ref(doc) or {}
    task = run.get("task") if isinstance(run.get("task"), dict) else {}
    key = (ref.get("arm") if isinstance(ref.get("arm"), str) else "base", task.get("id"))
    for row in fit["levels"]:
        if isinstance(row, Mapping) and (row.get("arm"), row.get("input")) in (key, (key[0], None)) \
                and _finite(row.get("mean")) and _finite(row.get("sd")):
            return float(row["mean"]), float(row["sd"])
    return None


def _finite(x: Any) -> bool:
    """A number other than a bool that is finite as a float; an int beyond the float range is not."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False
