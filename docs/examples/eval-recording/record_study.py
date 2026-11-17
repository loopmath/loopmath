"""Record a tiny study the way a pipeline would. The example behind docs/eval-recording.md.

    python docs/examples/eval-recording/record_study.py --out my-study

It writes `my-study/` (rubric.json, judges.json, value.json, runs.jsonl, scores.jsonl, pairs.jsonl) and,
beside it, `my-study-reviewer.csv` and `my-study-reviewer-pairs.csv`: the tables a human reviewer filled in.

Everything here is made up. No model is called: the "pipeline" and the "judges" below are random numbers
drawn from fixed formulas, so the same command always writes the same files. What is real is the shape of
every record. `record_run`, `record_reading` and `record_pair` are the part to copy into your own code.

Because the runs are made up, every run line is marked `"simulated": true`, and `eval fit` takes the study
only with `--allow-simulated`. Runs of a real system carry no such mark.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from pathlib import Path

RUBRIC = {
    "schema": "loopmath.eval-rubric/1",
    "id": "answer",
    "version": "1",
    "components": [
        {"id": "correct", "title": "The facts are right", "scale": {"kind": "binary"}},
        {"id": "clarity", "title": "Clear to a new reader",
         "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}},
        {"id": "depth", "title": "Depth of the answer", "scale": {"kind": "linear", "range": [0, 10]},
         "review_minutes": 2},
        {"id": "overall", "title": "Overall", "role": "overall", "scale": {"kind": "linear", "range": [0, 10]}},
    ],
}
RUBRIC_REF = f"{RUBRIC['id']}@{RUBRIC['version']}"   # every reading names the rubric it was scored under

JUDGES = {
    "schema": "loopmath.eval-judges/1",
    "anchor": ["reviewer-1"],
    "judges": [
        {"id": "reviewer-1", "kind": "person", "blinded": True},
        {"id": "judge-main", "kind": "model", "model": {"id": "model-large", "family": "alpha"},
         "effort": "high", "prompt": "rubric-prompt-v2", "mode": "per_component", "blinded": True,
         "cost_usd_per_call": 0.03},
        {"id": "judge-cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"},
         "effort": "low", "prompt": "rubric-prompt-v2", "mode": "panel", "blinded": True,
         "cost_usd_per_call": 0.004},
    ],
}

VALUE = {
    "schema": "loopmath.eval-value/1",
    "weights": {"clarity": 1, "depth": 2},
    "gates": [{"component": "correct"}],
    "objective": {"kind": "best_value", "reference": "baseline"},
}

# The two configurations compared. An arm is one fixed setting of every step.
ARMS = {
    "baseline": {"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
                 "variant": "prompt-a"},
    "candidate": {"step": "final", "model": {"id": "model-large", "family": "alpha"}, "effort": "high",
                  "variant": "prompt-a"},
}


# ---------------------------------------------------------------------------------------------------
# The part to copy: one function per kind of record. Each appends one JSON line.

def _append(path: Path, record: dict) -> None:
    """Append one JSON line. A member with no value is left out, never written as null."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: v for k, v in record.items() if v is not None}) + "\n")


def record_run(study: Path, *, run: str, input: str, arm: str, steps: list[dict], cost_usd: float | None = None,
               refs: dict | None = None, simulated: bool | None = None) -> None:
    """One output of the pipeline: which input it answers, under which arm, and what each step ran on.
    Pass `simulated=True` only for made-up runs. Leave it out for the runs of a real system."""
    _append(study / "runs.jsonl", {"schema": "loopmath.eval-runs/1", "run": run, "input": input, "arm": arm,
                                   "steps": steps, "cost_usd": cost_usd, "simulated": simulated, "refs": refs})


def record_reading(study: Path, *, run: str, component: str, judge: str, call: str, value, observed_at: str,
                   rubric: str = RUBRIC_REF, step: str | None = None, item: str | None = None,
                   of: int | None = None, cost_usd: float | None = None, reading_id: str | None = None,
                   replaces: str | None = None, assignment: str | None = None) -> str:
    """One score out of one request to a judge. Returns the id of the line.

    `call` is the id of the request. Every line that came out of one request has the same `call`: the
    components of a panel answer, and the check items of an items answer. A new request is a new `call`.

    The id of the line is the call, the component and the item, so it is the same on every rerun. To correct
    a line that was stored wrong, write a new line with its own `reading_id`, the SAME `call` (no new request
    was made) and the id of the wrong line in `replaces`."""
    reading_id = reading_id or ".".join(part for part in (call, component, item) if part)
    _append(study / "scores.jsonl", {
        "schema": "loopmath.eval-scores/1", "id": reading_id, "run": run, "step": step, "component": component,
        "judge": judge, "call": call, "value": value, "item": item, "of": of, "rubric": rubric,
        "observed_at": observed_at, "cost_usd": cost_usd, "replaces": replaces, "assignment": assignment})
    return reading_id


def record_pair(study: Path, *, a: str, b: str, judge: str, call: str, verdict: str, shown_first: str,
                observed_at: str, rubric: str = RUBRIC_REF, component: str | None = None,
                step: str | None = None, cost_usd: float | None = None, pair_id: str | None = None,
                replaces: str | None = None) -> str:
    """One comparison of two outputs of the same input. `verdict` is "a", "b" or "tie". Returns the id.

    A request that compares on several components writes one line per component, all with the same `call`.
    The id of the line is the call, and the component when one is named. A correction is a new line with its
    own `pair_id`, the same `call` and the id of the wrong line in `replaces`."""
    pair_id = pair_id or ".".join(part for part in (call, component) if part)
    _append(study / "pairs.jsonl", {
        "schema": "loopmath.eval-pairs/1", "id": pair_id, "a": a, "b": b, "step": step, "component": component,
        "judge": judge, "call": call, "shown_first": shown_first, "verdict": verdict, "rubric": rubric,
        "observed_at": observed_at, "cost_usd": cost_usd, "replaces": replaces})
    return pair_id


# ---------------------------------------------------------------------------------------------------
# A made-up pipeline and made-up judges. Replace all of this with your own system.

MEAN = {"baseline": {"correct": 0.52, "clarity": 0.50, "depth": 0.50, "overall": 0.50},
        "candidate": {"correct": 0.66, "clarity": 0.60, "depth": 0.63, "overall": 0.62}}
RUN_COST = {"baseline": 0.011, "candidate": 0.094}
BIAS = {"reviewer-1": 0.0, "judge-main": 0.04, "judge-cheap": 0.10}      # how lenient the judge is
PER_OUTPUT = {"reviewer-1": 0.03, "judge-main": 0.05, "judge-cheap": 0.08}  # its repeatable error on one output
PER_CALL = {"reviewer-1": 0.04, "judge-main": 0.04, "judge-cheap": 0.07}    # what changes from call to call


class Made:
    """The made-up world: a hidden quality per output, and what each judge would say about it.

    `asked` lists every request made of a judge, in order: ("score", judge, run, components) for `ask` and
    ("pair", judge, a, b, shown_first) for `compare`. The recorded files must show the same requests."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.quality: dict[tuple[str, str], float] = {}
        self.view: dict[tuple[str, str, str], float] = {}
        self.clock = dt.datetime(2026, 3, 2, 9, 0, 0, tzinfo=dt.timezone.utc)
        self.calls = 0
        self.asked: list[tuple] = []

    def call_id(self) -> str:
        """A new id for one request to a judge. In your code: the request id your judge API returns, or a UUID."""
        self.calls += 1
        return f"c{self.calls:04d}"

    def now(self) -> str:
        """A fixed clock, so the files come out the same. In your code: the time the judge answered,
        `datetime.datetime.now().astimezone().isoformat(timespec="seconds")`."""
        self.clock += dt.timedelta(seconds=20)
        return self.clock.isoformat()

    def run_pipeline(self, run: str, arm: str, input_level: dict[str, float]) -> float:
        whole = self.rng.gauss(0, 0.04)                                    # a good output is good on every count
        for component, mean in MEAN[arm].items():
            self.quality[(run, component)] = mean + input_level[component] + whole + self.rng.gauss(0, 0.04)
        return round(RUN_COST[arm] * self.rng.uniform(0.8, 1.25), 4)

    def _look(self, judge: str, run: str, component: str, shared: float) -> float:
        if (judge, run, component) not in self.view:
            self.view[(judge, run, component)] = (self.quality[(run, component)] + BIAS[judge]
                                                  + self.rng.gauss(0, PER_OUTPUT[judge]))
        return self.view[(judge, run, component)] + shared + self.rng.gauss(0, PER_CALL[judge])

    def ask(self, judge: str, run: str, components: list[str]) -> dict:
        """One request to a judge about one output: a raw score per component asked, each on its own scale."""
        self.asked.append(("score", judge, run, tuple(components)))
        shared = self.rng.gauss(0, 0.03) if len(components) > 1 else 0.0   # scores of one request move together
        out = {}
        for component in components:
            x = self._look(judge, run, component, shared)
            if component == "correct":
                out[component] = 1 if x > 0.5 else 0
            elif component == "clarity":
                out[component] = "poor" if x < 0.42 else "ok" if x < 0.64 else "good"
            else:
                out[component] = round(10 * min(max(x, 0.03), 0.97), 1)
        return out

    def compare(self, judge: str, a: str, b: str, shown_first: str) -> str:
        """One request to a judge to compare two outputs overall."""
        self.asked.append(("pair", judge, a, b, shown_first))
        gap = 10 * (self._look(judge, a, "overall", 0.0) - self._look(judge, b, "overall", 0.0))
        gap += 0.3 if shown_first == "a" else -0.3                         # a small lean to the first one shown
        return "tie" if abs(gap) < 0.4 else "a" if gap > 0 else "b"


def main(out: Path, seed: int = 7, inputs: int = 16) -> Made:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("runs.jsonl", "scores.jsonl", "pairs.jsonl"):
        (out / name).write_text("", encoding="utf-8")                      # start the tables again
    for name, document in (("rubric.json", RUBRIC), ("judges.json", JUDGES), ("value.json", VALUE)):
        (out / name).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    made = Made(seed)
    components = [row["id"] for row in RUBRIC["components"]]

    # 1. Runs. Every input goes through every arm. Run ids are neutral and handed out in shuffled order, so
    #    neither the id nor its number tells a reviewer which arm wrote the output.
    todo = [(f"q{n:02d}", arm) for n in range(1, inputs + 1) for arm in ARMS]
    made.rng.shuffle(todo)
    level = {f"q{n:02d}": {c: made.rng.gauss(0, 0.06) for c in components} for n in range(1, inputs + 1)}
    run_of: dict[tuple[str, str], str] = {}
    for n, (input_id, arm) in enumerate(todo, 1):
        run = f"r{n:03d}"
        run_of[(input_id, arm)] = run
        cost = made.run_pipeline(run, arm, level[input_id])
        record_run(out, run=run, input=input_id, arm=arm, steps=[ARMS[arm]], cost_usd=cost,
                   refs={"input": f"inputs/{input_id}.md", "output": f"outputs/{run}.md"},
                   simulated=True)                                         # made up: real runs leave it out
    runs = sorted(run_of.values())

    # 2. judge-main is asked for one component per request (mode per_component): one line per request.
    #    Every other output gets a second request on the two number scales. A repeat is a new request, so it
    #    has a new call id. It is never a second line under the old call.
    for n, run in enumerate(runs):
        for component in components:
            record_reading(out, run=run, component=component, judge="judge-main", call=made.call_id(),
                           value=made.ask("judge-main", run, [component])[component], observed_at=made.now())
        if n % 2 == 0:
            for component in ("depth", "overall"):
                record_reading(out, run=run, component=component, judge="judge-main", call=made.call_id(),
                               value=made.ask("judge-main", run, [component])[component], observed_at=made.now())

    # 3. judge-cheap is asked for every component in one request (mode panel): the lines of the request
    #    share its call id. Every fourth output gets a second panel request.
    for n, run in enumerate(runs):
        for _ in range(2 if n % 4 == 1 else 1):
            call = made.call_id()
            answer, at = made.ask("judge-cheap", run, components), made.now()
            for component in components:
                slip = n == 0 and component == "depth"                     # stored with the point one place off
                record_reading(out, run=run, component=component, judge="judge-cheap", call=call,
                               value=round(answer[component] / 10, 2) if slip else answer[component],
                               observed_at=at)
            if n == 0:
                # The fix for the slip: a new line with its own id, the SAME call (the judge was not asked
                # again) and the id of the wrong line in `replaces`. The panel request stays one request.
                record_reading(out, run=run, component="depth", judge="judge-cheap", call=call,
                               value=answer["depth"], observed_at=at, reading_id=f"{call}.depth.fix1",
                               replaces=f"{call}.depth")

    # 4. Comparisons by judge-main on half of the inputs, each pair in both orders (two requests).
    for n in range(1, inputs + 1, 2):
        a, b = run_of[(f"q{n:02d}", "baseline")], run_of[(f"q{n:02d}", "candidate")]
        if made.rng.random() < 0.5:
            a, b = b, a                                                    # which arm is "a" is random too
        for first in ("a", "b"):
            record_pair(out, a=a, b=b, judge="judge-main", call=made.call_id(),
                        verdict=made.compare("judge-main", a, b, first), shown_first=first, observed_at=made.now())

    # 5. The reviewer's sheets. A person scored both outputs of 6 inputs and saw only the output files. The
    #    reviewer worked one component at a time: all 12 outputs on the first component, then the next. So
    #    every score is a request of its own (mode per_component, the default), and the sheet gives each one
    #    its own `call`. A reviewer who scores all components of an output in one sitting is a panel judge:
    #    then the rows of one sitting share a `call`, and the judges file says "mode": "panel".
    reviewed = [run_of[(f"q{n:02d}", arm)] for n in range(2, 13, 2) for arm in ARMS]
    rows, pair_rows = [], []
    for component in components:
        for run in reviewed:
            look = f"rv.{run}.{component}"
            rows.append({"id": look, "call": look, "run": run, "component": component, "judge": "reviewer-1",
                         "value": made.ask("reviewer-1", run, [component])[component],
                         "observed_at": made.now(), "replaces": ""})
    for component in ("depth", "overall"):                                 # a second look at 4 outputs: repeats
        for run in reviewed[:2] + reviewed[6:8]:
            look = f"rv.{run}.{component}.2"
            rows.append({"id": look, "call": look, "run": run, "component": component, "judge": "reviewer-1",
                         "value": made.ask("reviewer-1", run, [component])[component],
                         "observed_at": made.now(), "replaces": ""})
    slip = next(row for row in rows if row["component"] == "depth")         # one depth score was mistyped
    right = slip["value"]
    slip["value"] = round(min(right + 3.0, 9.7), 1)
    # The fix: a new row with its own id, the same call (the reviewer did not look again) and `replaces`.
    rows.append({**slip, "id": slip["id"] + ".fixed", "value": right, "observed_at": made.now(),
                 "replaces": slip["id"]})
    for k, n in enumerate(range(2, 13, 2)):
        a, b = run_of[(f"q{n:02d}", "candidate")], run_of[(f"q{n:02d}", "baseline")]
        if k % 3 == 0:
            a, b = b, a
        first = "a" if k % 2 == 0 else "b"
        verdict = made.compare("reviewer-1", a, b, first)
        choice = "tie" if verdict == "tie" else "left" if verdict == first else "right"
        pair_rows.append({"a": a, "b": b, "judge": "reviewer-1", "shown_first": first, "choice": choice,
                          "observed_at": made.now()})
    for path, table in ((out.with_name(out.name + "-reviewer.csv"), rows),
                        (out.with_name(out.name + "-reviewer-pairs.csv"), pair_rows)):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    return made


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the tiny synthetic study of docs/eval-recording.md.")
    parser.add_argument("--out", required=True, metavar="DIR", help="the study folder to write")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(Path(args.out), args.seed)
    print(f"wrote {args.out}")
