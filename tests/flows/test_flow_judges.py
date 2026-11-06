"""The judges fit on the example flow: its keys, the 90% coverage of planted biases and noises over 20 seeds, the
closed-form algebra against a dense computation, catch rates, the uncalibrated anchor, the combined quality of a
run, inputs with one run, the most judges a fit takes, and linear judges read as fractions, with ranges and scores
no float holds read as none, on signals and on the flow; and the unpooled estimate every statistical reader takes,
against a dense computation, on both anchors, and for a run far from its arm."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
import time
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from loopmath.belief import fit as belief_fit
from loopmath.belief.state import transform
from loopmath.flows import judges, load
from loopmath.flows import simulate as sim
from loopmath.flows.folder import save_plan
from loopmath.flows.records import PLAN_SCHEMA, Arm, Edit, Experiment, Plan, RunSpec
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.types import FlowRef

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
ARMS = ("base", "drop-critic")
INPUTS = tuple(f"q{k:02d}" for k in range(1, 21))
SEEDS = range(1, 21)  # fixed before the first coverage run
S_RUN = TRUTH["quality"]["run_sd"]  # 0.12, net of the input effect
NOW = "2026-11-03T12:00:00-08:00"
CALIBRATED_SEED = 7
CALIBRATED_PROBE = "run_cal-drop-critic-q05-1"  # a run of calibrated_runs whose two values differ by over 0.01


# Folders and runs


def _drop_critic(cfg):
    w = cfg["workflow"]
    w["pieces"] = [p for p in w["pieces"] if p["id"] != "critic"]
    w["artifacts"] = [a for a in w["artifacts"] if a["id"] != "verdict"]
    w["edges"] = [e for e in w["edges"] if "critic" not in e]
    w["control"]["gates"] = ["extract"]
    w["control"]["repair"] = {"extract": "extract"}
    del cfg["settings"]["critic"]


def flow_folder(directory: Path, flow_change=None):
    """The example flow in `directory`, changed by `flow_change`, with a plan of two arms: base and drop-critic."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("questions.jsonl", "rubric.md", "truth.json"):
        shutil.copyfile(EXAMPLE / name, directory / name)
    doc = json.loads((EXAMPLE / "flow.ocp.json").read_text(encoding="utf-8"))
    if flow_change is not None:
        flow_change(doc["flow"])
        doc["flow"]["digest"] = flow_digest(doc["flow"])
    (directory / "flow.ocp.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    first = load(directory)
    base = first.flow.configuration
    drop = copy.deepcopy(base)
    _drop_critic(drop)
    drop["source"] = "designed"
    drop["id"] = config_id(drop["workflow"], drop["settings"])
    plan = Plan(schema=PLAN_SCHEMA, id="pln_lit-brief_judges",
                flow=FlowRef(id=first.flow.id, version=first.flow.version, digest=first.digest), seed=1103,
                margin=0.1, arms=(Arm(id="base", configuration=copy.deepcopy(base), edits=()),
                                  Arm(id="drop-critic", configuration=drop, edits=(Edit(kind="drop", piece="critic"),))),
                experiments=(Experiment(id="judges-sim", kind="screen", arms=("drop-critic",)),))
    save_plan(directory, plan)
    return load(directory)


def _runs(folder, rows, seed: int) -> list[dict]:
    """Simulated runs from (run id, arm, input, model judges, person) rows."""
    specs = []
    for run_id, arm, task, model_judges, person in rows:
        cfg = folder.configuration(arm)[1]
        split = "holdout" if task in folder.flow.inputs.holdout else "train"
        ref = FlowRef(id=folder.flow.id, version=folder.flow.version, digest=folder.digest, arm=arm,
                      plan=folder.plan.id, split=split)
        specs.append(RunSpec(run_id=run_id, input=task, arm=arm, configuration_id=cfg["id"],
                             settings=copy.deepcopy(cfg["settings"]), experiment="judges-sim", split=split,
                             slate=None, judges=tuple(model_judges), person=person, flow=ref))
    return sim.simulate(folder, arm="base", n=len(specs), seed=seed, truth=TRUTH, specs=specs)


def corpus(folder, seed: int) -> list[dict]:
    """200 runs: each arm on each of the 20 inputs 5 times, both model judges on every run, and a person on 40 (on
    each arm and input once, so 10 in each group of 5 inputs)."""
    rows = [(f"run_s{seed}-{arm}-{task}-{rep}", arm, task, ("j-claude", "j-astra"), rep == k % 5)
            for arm in ARMS for rep in range(5) for k, task in enumerate(INPUTS)]
    return _runs(folder, rows, seed)


def calibrated_runs(directory: Path) -> list[dict]:
    """20 run documents of the example flow, with a new folder for them at `directory`: both arms on q01 to q05
    twice, both model judges on every run and a person on 8 (the first repeat of q01 to q04). The planted biases
    are +0.05 (j-claude) and -0.03 (j-astra), so the fit is calibrated. For `CALIBRATED_PROBE` the reader's
    value over all 20 (`judges.current_quality(docs)`) differs from the run's own (`judges.current_quality([doc])`)
    by more than 0.01. For reader tests in other modules."""
    folder = flow_folder(directory)
    rows = [(f"run_cal-{arm}-{task}-{rep}", arm, task, ("j-claude", "j-astra"), rep == 0 and task != "q05")
            for arm in ARMS for rep in range(2) for task in INPUTS[:5]]
    return _runs(folder, rows, CALIBRATED_SEED)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return flow_folder(tmp_path_factory.mktemp("judges") / "lit-brief")


@pytest.fixture(scope="module")
def seed1(folder):
    docs = corpus(folder, 1)
    return docs, judges.fit(folder.flow, docs)


def _strip(doc, keep):
    """A copy of `doc` holding only the judge signals of judges in `keep`."""
    doc = copy.deepcopy(doc)
    doc["run"]["signals"] = [s for s in doc["run"]["signals"]
                             if not (s.get("kind") == "score" and (s.get("source") or {}).get("ref") not in keep)]
    return doc


def _row(fit, judge_id):
    return next(r for r in fit["judges"] if r["id"] == judge_id)


def _points(fit):
    """Every number of a fit that the grid sets, but the noises' and s_run's interval endpoints, in a fixed order."""
    out = [fit["s_run"]]
    for r in fit["judges"]:
        out += [r["bias"], *r["bias_90"], r["noise"], r["bias_sd"]]
    for run_id in sorted(fit["runs"]):
        out += [fit["runs"][run_id]["q"], fit["runs"][run_id]["sd"]]
    for level in fit["levels"]:
        out += [level["mean"], level["sd"]]
    return np.array(out, dtype=float)


def _endpoints(fit):
    """The 90% interval endpoints the grid marginals give: s_run's and each judge's noise's."""
    return np.array([*fit["s_run_90"], *(v for r in fit["judges"] for v in r["noise_90"])], dtype=float)


def _values(fit):
    """Every number of a fit that the grid sets."""
    return np.concatenate([_points(fit), _endpoints(fit)])


def _counts(fit):
    """The parts of a fit the grid does not set: counts, rates, names and keys."""
    return ({k: fit[k] for k in ("calibrated", "n_person", "score", "anchor", "n_runs")},
            [{k: r[k] for k in ("id", "kind", "n", "catch_n", "catch_rate")} for r in fit["judges"]],
            {k: v["key"] for k, v in fit["runs"].items()}, [(r["arm"], r["input"]) for r in fit["levels"]])


# 1. Keys


def test_fit_keys(seed1, folder):
    docs, fit = seed1
    unscored = [_strip(d, ()) for d in docs[:10]]
    for got in (fit, judges.fit(folder.flow, unscored), judges.fit(folder.flow, [])):
        assert set(judges.FIT_KEYS) <= set(got)
        assert [r["id"] for r in got["judges"]] == ["j-claude", "j-astra", "person"]
        for r in got["judges"]:
            assert set(judges.JUDGE_KEYS) <= set(r)
            for key in ("bias_90", "noise_90"):
                assert r[key] is None or (len(r[key]) == 2 and r[key][0] <= r[key][1])
    for got in (judges.fit(folder.flow, unscored), judges.fit(folder.flow, [])):
        assert (got["calibrated"], got["n_person"], got["s_run"], got["runs"]) == (False, 0, None, {})
        assert all(r[k] is None for r in got["judges"] for k in ("bias", "noise", "bias_90", "noise_90"))
    assert fit["calibrated"] and fit["n_person"] == 40 and fit["n_runs"] == 200 and fit["anchor"] == "person"
    assert fit["score"] == "quality" and len(fit["runs"]) == 200
    json.dumps(fit, allow_nan=False)


# 2. Coverage and recovery (the acceptance loop)


def test_coverage_and_recovery(folder):
    """Seeds 1 to 20 under the shipped truth. Each of the five planted values sits in its 90% interval in at least
    16 seeds; s_run is within 0.12 +- 0.03 in every seed and its mean within 0.12 +- 0.01; without the input
    levels each bias and model judge's noise moves by at most 0.01, the person's noise by at most 0.015."""
    truth = {("j-claude", "bias"): 0.05, ("j-astra", "bias"): -0.03, ("j-claude", "noise"): 0.08,
             ("j-astra", "noise"): 0.12, ("person", "noise"): 0.03}
    held = dict.fromkeys(truth, 0)
    s_runs, s_held, shifts, flat = [], 0, [], []
    js = judges._flow_judges(folder.flow)
    for seed in SEEDS:
        docs = corpus(folder, seed)
        fit = judges.fit(folder.flow, docs)
        bare = judges._fit(js, judges._runs(docs, js, folder.flow.id), (0.7, "higher"), "quality", inputs=False)
        for (jid, what), value in truth.items():
            lo, hi = _row(fit, jid)[f"{what}_90"]
            held[jid, what] += lo <= value <= hi
        s_runs.append(fit["s_run"])
        s_held += fit["s_run_90"][0] <= S_RUN <= fit["s_run_90"][1]
        flat.append(bare["s_run"])
        shifts.append({key: abs(_row(fit, key[0])[key[1]] - _row(bare, key[0])[key[1]]) for key in truth})
    print("\nvalue              truth  held of 20")
    for (jid, what), value in truth.items():
        print(f"{jid + ' ' + what:18} {value:+.3f}  {held[jid, what]}")
    print(f"s_run              {S_RUN:.3f}  {s_held} (reported); mean {np.mean(s_runs):.4f}, "
          f"range {min(s_runs):.4f} to {max(s_runs):.4f}")
    largest = {key: max(d[key] for d in shifts) for key in truth}
    print(f"s_run, no inputs   mean {np.mean(flat):.4f}; largest shifts "
          + ", ".join(f"{j} {k} {v:.4f}" for (j, k), v in largest.items()))
    assert all(n >= 16 for n in held.values()), held
    assert all(abs(s - S_RUN) <= 0.03 for s in s_runs), s_runs
    assert abs(float(np.mean(s_runs)) - S_RUN) <= 0.01
    # without v_i the person's noise absorbs input spread that the input levels tell apart from it
    assert all(v <= (0.015 if key == ("person", "noise") else 0.01) for key, v in largest.items()), largest


# 3. The algebra against a dense computation


def _hand_runs():
    """12 runs: two arms, three inputs and one run with none, three reading patterns."""
    rng = np.random.default_rng(5)
    patterns = [("j1", "j2"), ("j1", "j2", "p"), ("j1",)]
    runs = []
    for k in range(12):
        values = {j: float(rng.uniform(0.2, 0.9)) for j in patterns[k % 3]}
        runs.append(judges._Run(f"run_{k:02d}", "a" if k % 2 else "b", None if k == 11 else f"i{k % 3}", values))
    return [judges._Judge("j1", "model", "quality"), judges._Judge("j2", "model", "quality"),
            judges._Judge("p", "person", "quality")], runs


def _exact_loglik(X, y, V, L2):
    """The log density of y under N(0, X X' L2 + V), by exact rational elimination: the float dense form loses
    about 1e-8 here to the wide level prior's conditioning."""
    n = len(y)
    S = [[Fraction(float(L2)) * sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(X[i], X[k]))
          + Fraction(float(V[i, k])) for k in range(n)] + [Fraction(float(y[i]))] for i in range(n)]
    det = Fraction(1)
    for c in range(n):
        det *= S[c][c]
        for r in range(c + 1, n):
            f = S[r][c] / S[c][c]
            if f:
                S[r] = [a - f * b for a, b in zip(S[r], S[c])]
    x = [Fraction(0)] * n
    for r in range(n - 1, -1, -1):
        x[r] = (S[r][n] - sum(S[r][k] * x[k] for k in range(r + 1, n))) / S[r][r]
    quad = sum(Fraction(float(a)) * b for a, b in zip(y, x))
    logdet = math.log(det.numerator) - math.log(det.denominator)
    return -0.5 * (n * math.log(2 * math.pi) + logdet + float(quad))


def _dense(data, runs, scored, point):
    """The posterior and log marginal likelihood from the full covariance of every score."""
    index = {j.id: k for k, j in enumerate(scored)}
    s_run, noise = point[0], point[1:]
    X, y, blocks, owner = [], [], [], []
    for r_k, (r, ai, ii) in enumerate(zip(runs, data.aidx, data.iidx)):
        js = sorted(index[j] for j in r.values)
        for j in js:
            x = np.zeros(data.size)
            x[ai] = 1.0
            if ii >= 0:
                x[data.A + ii] = 1.0
            x[data.A + data.I:] = data.M[j]
            X.append(x)
            y.append(r.values[scored[j].id])
            owner.append((r_k, j))
        blocks.append(js)
    X, y = np.array(X), np.array(y)
    n = len(y)
    V = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if owner[a][0] == owner[b][0]:
                V[a, b] = s_run ** 2 + (noise[owner[a][1]] ** 2 if a == b else 0.0)
    L2 = judges.LEVEL_SD ** 2
    loglik = _exact_loglik(X, y, V, L2)
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X + np.eye(data.size) / L2
    cov = np.linalg.inv(A)
    theta = cov @ X.T @ Vi @ y
    # q_r = level_r + u_r, from the joint posterior of (parameters, u) with the judges' noise only
    R = len(runs)
    U = np.zeros((n, R))
    for a, (r_k, _) in enumerate(owner):
        U[a, r_k] = 1.0
    Z = np.hstack([X, U])
    D = np.diag([noise[j] ** 2 for _, j in owner])
    prior = np.diag([1.0 / L2] * data.size + [1.0 / s_run ** 2] * R)
    C = np.linalg.inv(Z.T @ np.linalg.inv(D) @ Z + prior)
    mu = C @ Z.T @ np.linalg.inv(D) @ y
    qm, qv = [], []
    for r_k in range(R):
        c = np.zeros(data.size + R)
        c[data.aidx[r_k]] = 1.0
        if data.iidx[r_k] >= 0:
            c[data.A + data.iidx[r_k]] = 1.0
        c[data.size + r_k] = 1.0
        qm.append(c @ mu)
        qv.append(c @ C @ c)
    return loglik, theta, cov, np.array(qm), np.array(qv)


@pytest.mark.parametrize("calibrated", [False, True])
def test_closed_form_equals_dense(calibrated):
    scored, runs = _hand_runs()
    data = judges._Data(scored, runs, calibrated, True)
    points = np.array([[0.12, 0.08, 0.12, 0.03], [0.05, 0.2, 0.04, 0.01], [0.3, 0.01, 0.3, 0.1]])
    loglik, theta, cov = judges._evaluate(data, points, posterior=True)
    qm, qv = judges._q_moments(data, points, theta, cov)
    for g, point in enumerate(points):
        d_loglik, d_theta, d_cov, d_qm, d_qv = _dense(data, runs, scored, point)
        assert abs(loglik[g] - d_loglik) <= 1e-9
        assert np.allclose(theta[g], d_theta, rtol=1e-9, atol=1e-9)
        assert np.allclose(cov[g], d_cov, rtol=1e-9, atol=1e-9)
        assert np.allclose(qm[g], d_qm, rtol=1e-9, atol=1e-9)
        assert np.allclose(qv[g], d_qv, rtol=1e-9, atol=1e-9)


def test_uncalibrated_biases_are_centred():
    M = judges._centred(4)
    assert np.allclose(M.T @ M, np.eye(3)) and np.allclose(M.sum(0), 0.0)


# 4. Catch rate


def _doc(run_id, values, *, arm="base", task="q01", flow="lit-brief", extra=()):
    signals = [{"id": f"sig_{run_id}_{j}", "kind": "score", "name": "quality", "value": v, "better": "higher",
                "scale": "fraction", "observed_at": "2026-01-05T10:00:00+00:00", "tier": "reported",
                "source": {"kind": "user" if j == "person" else "judge", "ref": j}} for j, v in values.items()]
    return {"run": {"id": run_id, "flow": {"id": flow, "arm": arm}, "task": {"id": task},
                    "signals": signals + list(extra)}}


PERSON = [0.5, 0.6, 0.65, 0.69, 0.8, 0.85, 0.9, 0.75]
CLAUDE = [0.6, 0.65, 0.68, 0.75, 0.72, 0.6, 0.95, 0.5]


def _catch_docs():
    return [_doc(f"run_c{k}", {"person": p, "j-claude": c, "j-astra": 0.9}, task=f"q{k + 1:02d}")
            for k, (p, c) in enumerate(zip(PERSON, CLAUDE))]


def test_catch_rate_on_known_counts():
    flow = load(EXAMPLE).flow
    fit = judges.fit(flow, _catch_docs())
    assert fit["calibrated"]
    assert (_row(fit, "j-claude")["catch_rate"], _row(fit, "j-claude")["catch_n"]) == (0.75, 4)
    assert (_row(fit, "j-astra")["catch_rate"], _row(fit, "j-astra")["catch_n"]) == (0.0, 4)
    assert (_row(fit, "person")["catch_rate"], _row(fit, "person")["catch_n"]) == (None, None)
    rule = flow.acceptance_rule
    lower = dataclasses.replace(flow, acceptance_rule={**rule, "score": {**rule["score"], "better": "lower"}})
    fit = judges.fit(lower, _catch_docs())
    assert (_row(fit, "j-claude")["catch_rate"], _row(fit, "j-claude")["catch_n"]) == (0.5, 4)
    no_target = dataclasses.replace(flow, acceptance_rule={"requires": ["claims-check"]})
    fit = judges.fit(no_target, _catch_docs())
    assert all((r["catch_rate"], r["catch_n"]) == (None, None) for r in fit["judges"])


def test_catch_n_is_a_direct_count(seed1):
    docs, fit = seed1
    for jid in ("j-claude", "j-astra"):
        n = caught = 0
        for doc in docs:
            values = {s["source"]["ref"]: s["value"] for s in doc["run"]["signals"] if s.get("kind") == "score"}
            if "person" in values and jid in values and values["person"] < 0.7:
                n += 1
                caught += values[jid] < 0.7
        assert n > 0 and _row(fit, jid)["catch_n"] == n and _row(fit, jid)["catch_rate"] == caught / n


# 5. Uncalibrated


def test_anchor_moves_at_six_person_runs(seed1, folder):
    docs, _ = seed1
    marked = [d["run"]["id"] for d in docs if any(s.get("source", {}).get("kind") == "user"
                                                  for s in d["run"]["signals"])]
    for n, calibrated in ((5, False), (6, True)):
        kept = set(marked[:n])
        some = [d if d["run"]["id"] in kept else _strip(d, ("j-claude", "j-astra")) for d in docs[:100]]
        fit = judges.fit(folder.flow, some)
        assert (fit["n_person"], fit["calibrated"]) == (n, calibrated)
        biases = [r["bias"] for r in fit["judges"]]
        if calibrated:
            assert fit["anchor"] == "person" and _row(fit, "person")["bias"] == 0.0
            assert _row(fit, "person")["bias_90"] == [0.0, 0.0]
            assert all(_row(fit, j)["catch_rate"] is not None for j in ("j-claude", "j-astra"))
        else:
            assert fit["anchor"] == "judges' mean" and abs(sum(biases)) < 1e-9
            assert all(r["catch_rate"] is None for r in fit["judges"])


# 6. Combined quality


def test_combined_quality(seed1, folder):
    docs, fit = seed1
    flow = folder.flow
    for doc in docs[:20]:
        q, sd = judges.combined_quality(flow, doc, fit)
        stored = fit["runs"][doc["run"]["id"]]
        assert (q, sd) == (stored["q"], stored["sd"])
        refs = [s["source"]["ref"] for s in doc["run"]["signals"] if s.get("kind") == "score"]
        # below the best judge's noise; a person's noise posterior is wide, so for a person-scored run the sd,
        # which carries that spread, is held to the person noise's 95% point
        bound = _row(fit, "person")["noise_90"][1] if "person" in refs else min(_row(fit, r)["noise"] for r in refs)
        assert 0 < sd < bound
    assert judges.combined_quality(flow, _strip(docs[0], ()), fit) == (None, None)
    # a run the fit does not hold: the plug-in posterior under the fit's values and its arm and input's level
    new = _strip(docs[0], ("j-claude", "j-astra"))
    new["run"]["id"] = "run_new"
    values = {s["source"]["ref"]: s["value"] for s in new["run"]["signals"] if s.get("kind") == "score"}
    num = den = 0.0
    for jid, v in values.items():
        r = _row(fit, jid)
        w = 1.0 / (r["noise"] ** 2 + r["bias_sd"] ** 2)
        num, den = num + w * (v - r["bias"]), den + w
    level = next(lv for lv in fit["levels"]
                 if (lv["arm"], lv["input"]) == (new["run"]["flow"]["arm"], new["run"]["task"]["id"]))
    w = 1.0 / (fit["s_run"] ** 2 + level["sd"] ** 2)
    num, den = num + w * level["mean"], den + w
    q, sd = judges.combined_quality(flow, new, fit)
    assert abs(q - num / den) < 1e-12 and abs(sd - den ** -0.5) < 1e-12
    # an empty fit: the prior noises' medians, bias 0 and no level
    q, sd = judges.combined_quality(flow, new, {})
    prior = math.sqrt(judges.NOISE_RANGE[0] * judges.NOISE_RANGE[1])
    assert abs(q - sum(values.values()) / 2) < 1e-12 and abs(sd - prior / math.sqrt(2)) < 1e-12


def test_fit_is_deterministic_and_order_free(seed1, folder):
    docs, fit = seed1
    some = docs[:60]
    first = judges.fit(folder.flow, some)
    assert json.dumps(first, sort_keys=True) == json.dumps(judges.fit(folder.flow, some), sort_keys=True)
    assert json.dumps(first, sort_keys=True) == json.dumps(judges.fit(folder.flow, some[::-1]), sort_keys=True)


@pytest.mark.parametrize("seed", [1, 8, 17])
def test_a_further_grid_pass_moves_nothing(seed, seed1, folder, monkeypatch):
    """The grid constants are settled. One or two more passes, and a 200000-point grid with eight passes, move no
    value the grid sets by more than 1e-3, a tenth of the tightest check on the fit's values (s_run, each judge's
    bias, bias sd, bias interval and noise, each run's q and sd, each level's mean and sd), and no interval
    endpoint of a grid marginal (s_run's and each noise's) by more than 2e-3; the counts, rates and keys stay
    equal. A pass whose zoom leaves the axes as they are ends the passes, so a further pass can be a no-op; the
    dense grid is the check that is not. Seeds 8 and 17 are the two where the person's noise moves most without
    the input levels."""
    docs, fit = seed1 if seed == 1 else (None, None)
    if docs is None:
        docs = corpus(folder, seed)
        fit = judges.fit(folder.flow, docs)
    assert (judges.PASSES, judges.BOX, judges.KEEP, judges.GRID) == (4, 1e-6, 1e-9, 20000)
    assert len(fit["levels"]) == 2 * len(INPUTS)
    weights, grids = judges._weights, []
    monkeypatch.setattr(judges, "_weights", lambda *a: grids.append(1) or weights(*a))
    for grid, passes in ((20000, 4), (20000, 5), (20000, 6), (200000, 8)):
        monkeypatch.setattr(judges, "GRID", grid)
        monkeypatch.setattr(judges, "PASSES", passes)
        grids.clear()
        start = time.perf_counter()
        other = judges.fit(folder.flow, docs)
        took = time.perf_counter() - start
        points, ends = np.abs(_points(other) - _points(fit)), np.abs(_endpoints(other) - _endpoints(fit))
        print(f"\nseed {seed} GRID {grid} PASSES {passes} ({took:.2f} s): {len(grids)} grids; "
              f"{len(points)} values, largest change {points.max():.2e}; {len(ends)} endpoints, {ends.max():.2e}")
        assert _counts(other) == _counts(fit)
        assert points.max() <= 1e-3 and ends.max() <= 2e-3


@pytest.mark.parametrize("seed", [1, 8, 17])
def test_grid_interval_reads_a_monotone_cubic(seed, seed1, folder, monkeypatch):
    """At the shipped grid each grid marginal's interval is finite and ordered inside its axis, the cubic through
    the marginal's CDF is monotone, and the bisection agrees with reading the same cubic at 20001 points."""
    from scipy.interpolate import PchipInterpolator

    docs = seed1[0] if seed == 1 else corpus(folder, seed)
    read, seen = judges._grid_interval, []
    monkeypatch.setattr(judges, "_grid_interval",
                        lambda axis, pmf: seen.append((axis, pmf, read(axis, pmf))) or seen[-1][2])
    judges.fit(folder.flow, docs)
    assert len(seen) == 4  # s_run and three judges' noises
    for axis, pmf, (lo, hi) in seen:
        la = np.log(axis)
        half = (la[1] - la[0]) / 2
        edges = np.concatenate([[la[0] - half], (la[:-1] + la[1:]) / 2, [la[-1] + half]])
        xs = np.linspace(edges[0], edges[-1], 20001)
        ys = PchipInterpolator(edges, np.concatenate([[0.0], np.cumsum(pmf)]) / pmf.sum())(xs)
        assert np.all(np.diff(ys) >= -1e-15)
        assert math.isfinite(lo) and math.isfinite(hi) and math.exp(edges[0]) <= lo < hi <= math.exp(edges[-1])
        for got, q in ((lo, judges.TAIL), (hi, 1 - judges.TAIL)):
            assert abs(math.log(got) - float(np.interp(q, ys, xs))) <= 1e-6


# Inputs with one run


def test_inputs_with_one_run(seed1, folder):
    docs, _ = seed1
    # one run on each of 8 inputs: no input has two runs, so s_run keeps its prior
    single = [d for d in docs if d["run"]["id"].startswith("run_s1-base-") and d["run"]["id"].endswith("-0")][:8]
    assert len({d["run"]["task"]["id"] for d in single}) == 8
    js = judges._flow_judges(folder.flow)
    runs = judges._runs(single, js, folder.flow.id)
    scored = sorted((j for j in js if any(j.id in r.values for r in runs)), key=lambda j: j.id)
    data = judges._Data(scored, runs, False, True)
    n = max(2, min(judges.MAX_POINTS, int(judges.GRID ** (1.0 / (len(scored) + 1)) + 1e-9)))
    axes = judges._axes(scored, n)
    points, logw = judges._weights(data, scored, axes)
    w = np.exp(logw - logw.max())
    marg = np.bincount(np.unravel_index(np.arange(len(points)), [n] * len(axes))[0], weights=w / w.sum(),
                       minlength=n)
    print(f"\ns_run marginal over {len(runs)} single-run inputs: largest gap to the prior {np.abs(marg - 1 / n).max():.2e}")
    assert np.abs(marg - 1.0 / n).max() <= 1e-3
    fit = judges.fit(folder.flow, single)
    assert all(math.isfinite(v["q"]) and math.isfinite(v["sd"]) for v in fit["runs"].values())
    # half-recorded pairs: some inputs with two runs, some with one
    half = docs[:40] + [d for d in docs[40:] if d["run"]["id"].endswith("-1")][:7]
    got = judges.fit(folder.flow, half)
    assert got["s_run"] is not None and math.isfinite(got["s_run"])
    counts = {}
    for d in half:
        key = (d["run"]["flow"]["arm"], d["run"]["task"]["id"])
        counts[key] = counts.get(key, 0) + 1
    assert any(c == 1 for c in counts.values()) and any(c > 1 for c in counts.values())
    # a single run's q is the noise-weighted mean of its bias-corrected readings at each grid point, averaged with
    # the grid's weights. The sd-10 level prior shrinks it by (1 / 10^2) / sum_j 1 / s_j^2, about 5.6e-5 here,
    # times q's distance from the arm level, so the gap is under 2e-5, not 0.
    data = judges._Data(scored, runs, fit["calibrated"], True)
    _, _, points, logw = judges._zoomed(data, scored)
    kept = logw >= logw.max() + math.log(judges.KEEP)
    w = np.exp(logw[kept] - logw.max())
    w /= w.sum()
    _, theta, cov = judges._evaluate(data, points[kept], posterior=True)
    qm, _ = judges._q_moments(data, points[kept], theta, cov)
    mu, _ = judges._bias_moments(data, theta, cov)
    d = data.mask[data.pidx][None] / points[kept][:, None, 1:] ** 2
    mean = ((data.Y[None] - mu[:, None, :]) * d).sum(-1) / d.sum(-1)
    gap = np.abs(w @ qm - w @ mean)
    assert np.allclose(w @ qm, [fit["runs"][r.id]["q"] for r in runs], rtol=0, atol=1e-12)
    print(f"single-run q against the weighted mean: largest gap {gap.max():.2e}")
    assert gap.max() <= 2e-5


# A corrected score


def test_a_corrected_score_is_the_one_reading(seed1, folder):
    """Per run, judge and score name the one reading is the newest score by `observed_at`, then by id, wherever
    it sits in the document: a correction written as a new signal replaces the score it corrects."""
    js = judges._flow_judges(folder.flow)
    base = copy.deepcopy(seed1[0][0])
    old = next(s for s in base["run"]["signals"] if s.get("kind") == "score" and s["source"]["ref"] == "j-claude")
    old.update(id="sig_01JB", observed_at="2026-11-03T12:00:00-08:00", tier="reported")
    assert "verified" not in judges.judged(base, "quality")[0]

    def read(*after, before=()):
        doc = copy.deepcopy(base)
        doc["run"]["signals"] = ([dict(old, **e) for e in before] + doc["run"]["signals"]
                                 + [dict(old, **e) for e in after])
        return judges._readings(doc, js)["j-claude"], judges.judged(doc, "quality")[0], doc

    assert read()[0] == old["value"]
    later = {"id": "sig_01JC", "value": 0.123, "tier": "verified", "observed_at": "2026-11-03T13:00:00-08:00"}
    value, tiers, doc = read(before=[later])
    assert value == 0.123 and "verified" in tiers
    run_id = base["run"]["id"]
    keys = [judges.fit(folder.flow, [d])["runs"][run_id]["key"] for d in (base, doc)]
    assert keys[0] != keys[1]
    # at an equal time the greater id wins, before or after in the document
    assert read(before=[dict(later, observed_at=old["observed_at"])])[0] == 0.123
    assert read(dict(later, observed_at=old["observed_at"], id="sig_01JA"))[0] == old["value"]
    # a score with no time is older than any timed one
    assert read(dict(later, observed_at=None))[0] == old["value"]


# The grid budget


EXTRA = tuple(f"j{k:02d}" for k in range(3, 15))  # model judges beside the example's three: 15 declared


def _more_judges(flow):
    flow["judges"] += [dict(flow["judges"][0], id=j) for j in EXTRA]


@pytest.fixture(scope="module")
def many(tmp_path_factory):
    """The example flow with 15 declared judges; it validates."""
    return flow_folder(tmp_path_factory.mktemp("judges") / "lit-brief", _more_judges)


def _scored_by(ids, *, flow="lit-brief", tag="g", runs=4):
    """`runs` runs of `flow`, each on its own input, scored by the judges `ids`."""
    return [_doc(f"run_{tag}-{k}", {j: round(0.3 + 0.04 * k + 0.01 * n, 4) for n, j in enumerate(ids)},
                 task=f"q{k + 1:02d}", flow=flow) for k in range(runs)]


def _grids(monkeypatch):
    """The grids `_weights` builds, as point counts, recorded and passed through."""
    built, weights = [], judges._weights

    def record(data, scored, axes):
        built.append(math.prod(len(a) for a in axes))
        return weights(data, scored, axes)

    monkeypatch.setattr(judges, "_weights", record)
    return built


def _rule(name):
    return types.SimpleNamespace(score=types.SimpleNamespace(name=name))


def test_the_judge_limit_is_computed_from_the_grid():
    limit = judges.MAX_SCORED_JUDGES
    assert limit == max(j for j in range(64) if 2 ** (j + 1) <= judges.GRID)
    assert (judges.GRID, limit) == (20000, 13) and 2 ** (limit + 1) <= judges.GRID < 2 ** (limit + 2)


def test_thirteen_judges_fit_and_fourteen_are_refused_before_any_grid(many, monkeypatch):
    """13 judges with readings fit on grids within GRID; a 14th, the person here, refuses the fit before any grid
    is built. Declared judges with no reading do not count."""
    built = _grids(monkeypatch)
    flow = many.flow
    model = ["j-claude", "j-astra", *EXTRA[:11]]
    docs = _scored_by(model)
    fit = judges.fit(flow, docs)
    assert len(flow.judges) == 15 and fit["refused"] is None
    assert built and max(built) == 2 ** 14 <= judges.GRID
    assert sorted(fit["runs"]) == [d["run"]["id"] for d in docs] and fit["s_run"] is not None
    assert {r["id"] for r in fit["judges"] if r["noise"] is None} == {"person", EXTRA[11]}
    built.clear()
    docs = _scored_by(["person", *model])
    fit = judges.fit(flow, docs)
    assert fit["refused"] == f"{judges.TOO_MANY}:14" == "too-many-judges:14" and built == []
    assert set(judges.FIT_KEYS) <= set(fit) and fit["n_runs"] == 4
    assert (fit["runs"], fit["levels"], fit["s_run"], fit["s_run_90"]) == ({}, [], None, None)
    assert all(r["bias"] is None and r["noise"] is None for r in fit["judges"])
    assert {r["id"]: r["n"] for r in fit["judges"]} == {**{j: 4 for j in ["person", *model]}, EXTRA[11]: 0}
    for doc in docs:
        got = judges.combined_quality(flow, doc, fit)
        assert isinstance(got, judges.Unfitted) and got == (None, None) and got.reason == "too-many-judges:14"
        q, sd = copy.deepcopy(got)
        assert (q, sd) == (None, None) and copy.deepcopy(got).reason == got.reason
        assert json.dumps(got) == "[null, null]"
    assert built == []


def test_a_refused_group_leaves_the_other_groups(monkeypatch):
    """One fit per flow and score name: a group above the limit reads no score, and the other groups' values are
    the ones they get alone."""
    over = _scored_by(["person", "j-claude", "j-astra", *EXTRA[:11]], tag="over")
    other = _scored_by(["person", "j-claude", "j-astra"], flow="other", tag="other")
    accuracy = _scored_by(["j-claude", "j-astra"], tag="acc")
    for doc in accuracy:
        for s in doc["run"]["signals"]:
            s["name"] = "accuracy"
    built = _grids(monkeypatch)
    alone = {**judges.quality_map([(d, _rule("quality")) for d in other]),
             **judges.quality_map([(d, _rule("accuracy")) for d in accuracy])}
    grids, built[:] = sorted(built), []
    got = judges.quality_map([(d, _rule("quality")) for d in over + other]
                             + [(d, _rule("accuracy")) for d in accuracy])
    assert len(alone) == 8 and all(type(v) is tuple and None not in v for v in alone.values())
    assert {k: got[k] for k in alone} == alone and sorted(built) == grids and max(grids) <= judges.GRID
    refused = {d["run"]["id"]: got[d["run"]["id"]] for d in over}
    assert set(got) == set(alone) | set(refused)
    assert all(isinstance(v, judges.Unfitted) and v == (None, None) and v.reason == "too-many-judges:14"
               for v in refused.values())


def test_flow_check_warning_names_the_limit(many):
    flow = many.flow
    assert judges.judge_warning(load(EXAMPLE).flow) is None
    assert judges.judge_warning(dataclasses.replace(flow, judges=flow.judges[:13])) is None
    for n in (14, 15):
        warning = judges.judge_warning(dataclasses.replace(flow, judges=flow.judges[:n]))
        assert f"declares {n} judges" in warning and "at most 13 with readings" in warning
        assert "too-many-judges:<n>" in warning


# 7. Linear judges


def _linear_astra(flow):
    judge = next(j for j in flow["judges"] if j["id"] == "j-astra")
    judge["scale"], judge["range"] = "linear", [1, 5]


def _as_fraction(doc):
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"]["ref"] == "j-astra":
            s["value"], s["scale"] = (s["value"] - 1) / (5 - 1), "fraction"
            s.pop("ext")
    return doc


def _unranged(doc):
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        s.pop("ext", None)
    return doc


@pytest.fixture(scope="module")
def linear(tmp_path_factory):
    folder = flow_folder(tmp_path_factory.mktemp("judges") / "lit-brief", _linear_astra)
    docs = corpus(folder, 2)[:60]
    return folder, docs


def test_linear_judge_signals_carry_the_range(linear):
    _, docs = linear
    astra = [s for d in docs for s in d["run"]["signals"] if s.get("kind") == "score" and s["source"]["ref"] == "j-astra"]
    assert astra and all(s["scale"] == "linear" and s["ext"] == {judges.JUDGE_EXT: {"range": [1, 5]}} for s in astra)
    others = [s for d in docs for s in d["run"]["signals"] if s.get("kind") == "score" and s["source"]["ref"] != "j-astra"]
    assert others and all("ext" not in s for s in others)


def test_linear_judge_fits_as_fractions(linear):
    folder, docs = linear
    fraction = [_as_fraction(d) for d in docs]
    a, b = judges.fit(folder.flow, docs), judges.fit(load(EXAMPLE).flow, fraction)
    assert np.allclose(_values(a), _values(b), rtol=0, atol=1e-12)
    qa, qb = judges.current_quality(docs), judges.current_quality(fraction)
    assert set(qa) == set(qb) == {d["run"]["id"] for d in docs}
    assert all(abs(qa[k][0] - qb[k][0]) <= 1e-12 and abs(qa[k][1] - qb[k][1]) <= 1e-12 for k in qa)
    ra, rb = _score_rows(docs), _score_rows(fraction)
    assert ra[1] == rb[1] == {"quality": {"scale": "fraction", "better": "higher", "unit": None}}
    assert ra[0].keys() == rb[0].keys()
    assert all(abs(ra[0][k] - rb[0][k]) <= 1e-12 for k in ra[0])


def test_linear_judge_without_a_range_is_skipped(linear):
    folder, docs = linear
    bare = [_unranged(d) for d in docs]
    without = [_strip(d, ("j-claude", "person")) for d in docs]
    qa, qb = judges.current_quality(bare), judges.current_quality(without)
    assert qa.keys() == qb.keys() and all(abs(qa[k][0] - qb[k][0]) <= 1e-12 for k in qa)
    rows, info, dropped = _score_rows(bare, dropped=True)
    assert dropped[judges.UNRANGED] == len(docs)
    assert rows.keys() == _score_rows(without)[0].keys()
    assert all(abs(rows[k] - _score_rows(without)[0][k]) <= 1e-12 for k in rows)
    # a run with the unranged judge only: no score at all, and the named warnings
    from loopmath.belief.rules import evidence, resolve_rule
    from loopmath.belief.outcome import parse_ts

    alone = _strip(bare[0], ("j-astra",))
    ev = evidence(alone, resolve_rule(alone), now=parse_ts(NOW))
    assert "quality" not in ev.scores and "unranged:j-astra" in ev.reasons and "unfitted:quality" in ev.reasons
    raw = [s["value"] for s in alone["run"]["signals"] if s.get("kind") == "score"]
    assert raw and not set(raw) & set(ev.scores.values())


BAD_RANGES = ([0, 2 ** 1024], [2 ** 60, 2 ** 60 + 1], [0, 2 ** 53 + 1])  # E214 refuses each; older documents may not


def _ranged(doc, span):
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if "ext" in s:
            s["ext"][judges.JUDGE_EXT]["range"] = span
    return doc


def test_unusable_ranges_read_unranged(linear):
    from loopmath.belief.rules import evidence, resolve_rule

    _, docs = linear
    assert judges.RANGE_LIMIT == 2 ** 53 and judges._range([0, 2 ** 53]) == (0.0, 9007199254740992.0)
    assert judges._range([-(2 ** 53), 2 ** 53]) == (-9007199254740992.0, 9007199254740992.0)
    bare = [_unranged(d) for d in docs]
    want, want_rows = judges.current_quality(bare), _score_rows(bare)[0]
    for span in BAD_RANGES:
        assert judges._range(span) is None and judges._range(span[::-1]) is None
        bad = [_ranged(d, span) for d in docs]
        got = judges.current_quality(bad)
        assert got.keys() == want.keys() and all(abs(got[k][0] - want[k][0]) <= 1e-12 for k in got)
        rows, _, dropped = _score_rows(bad, dropped=True)
        assert dropped[judges.UNRANGED] == len(docs)
        assert rows.keys() == want_rows.keys() and all(abs(rows[k] - want_rows[k]) <= 1e-12 for k in rows)
        alone = _strip(bad[0], ("j-astra",))
        assert judges.judged(alone, "quality") == ([], ["j-astra"], [], [], [], ())
        assert "unranged:j-astra" in evidence(alone, resolve_rule(alone), now=_now()).reasons


def test_a_score_beyond_the_float_range_is_no_reading(linear):
    _, docs = linear
    assert not judges._finite(2 ** 1024) and judges._finite(2 ** 53 + 1) and not judges._finite(True)
    huge = copy.deepcopy(docs[0])
    for s in huge["run"]["signals"]:
        if s.get("kind") == "score" and s["source"]["ref"] == "j-astra":
            s["value"] = 2 ** 1024
    assert "j-astra" not in {j.id for j in judges.signal_judges([huge], "quality")}
    assert len(judges.judged(huge, "quality")[0]) == len(judges.judged(docs[0], "quality")[0]) - 1
    assert judges.current_quality([huge]).keys() == judges.current_quality([_strip(docs[0], ("j-claude", "person"))]).keys()


def test_unusable_flow_ranges_read_unranged(linear):
    """A flow whose linear judge has a range no float reads: the fit does not raise, and with no range on the
    signals that judge has no reading, so the fit is the one without its scores."""
    folder, docs = linear
    bare = [_unranged(d) for d in docs]
    without = [_strip(d, ("j-claude", "person")) for d in bare]
    for span in BAD_RANGES:
        flow = dataclasses.replace(folder.flow, judges=tuple(
            dataclasses.replace(j, range=tuple(span)) if j.id == "j-astra" else j for j in folder.flow.judges))
        got = judges.fit(flow, bare)
        assert got["refused"] is None and _row(got, "j-astra")["n"] == 0
        assert got["runs"] == judges.fit(flow, without)["runs"] and len(got["runs"]) == len(docs)


@pytest.mark.parametrize("target", [10 ** 400, float("inf")])
def test_a_flow_target_no_float_holds_is_no_target(seed1, folder, target):
    """A flow rule's score target no finite float holds reads as no target. The public fit does not raise and
    decides no catch counts, so it equals the fit with no target."""
    docs, fit = seed1
    rule = folder.flow.acceptance_rule
    score = {k: v for k, v in rule["score"].items() if k != "target"}
    huge = dataclasses.replace(folder.flow, acceptance_rule={**rule, "score": {**score, "target": target}})
    none = dataclasses.replace(folder.flow, acceptance_rule={**rule, "score": score})
    got = judges.fit(huge, docs)
    assert all((r["catch_rate"], r["catch_n"]) == (None, None) for r in got["judges"])
    assert got == judges.fit(none, docs)
    assert _row(fit, "j-claude")["catch_n"] > 0  # the flow's own target does decide catch counts


def test_judged_fields_and_reasons():
    """`judged` returns `Judged`: callers read its fields by name, and `reasons` gives the reader's reasons in
    field order. A run is judged when any field is non-empty. `nonnumeric`, last, is a tuple with the default (),
    so a value built from the first five equals `judged`'s when no judge wrote a non-number."""
    assert judges.Judged._fields == ("usable", "unranged", "unattributed", "outside_range", "nonfinite", "nonnumeric")
    got = judges.Judged(["reported"], ["j-a"], ["sig_1", "?"], ["j-b"], ["j-c"], ("j-d", "j-e"))
    assert got.reasons == ["unranged:j-a", "unattributed:sig_1", "unattributed:?", "outside-range:j-b",
                           "nonfinite:j-c", "nonnumeric:j-d", "nonnumeric:j-e"]
    assert any(got) and not any(judges.Judged([], [], [], [], [])) and any(judges.Judged([], [], [], [], [], ("j",)))
    five = judges.Judged(["reported"], [], [], [], [])
    assert five.nonnumeric == () and five == judges.Judged(["reported"], [], [], [], [], ()) and five.reasons == []
    read = {"id": "sig_1", "kind": "score", "name": "quality", "value": 0.5, "scale": "fraction", "tier": "reported",
            "source": {"kind": "judge", "ref": "j-e"}}
    null = dict(read, id="sig_2", value=None, source={"kind": "judge", "ref": "j-d"})
    assert judges.judged({"run": {"signals": []}}, "quality") == judges.Judged([], [], [], [], [])
    assert judges.judged({"run": {"signals": [read]}}, "quality") == five
    got = judges.judged({"run": {"signals": [read, null]}}, "quality")
    assert type(got.nonnumeric) is tuple and got == judges.Judged(["reported"], [], [], [], [], ("j-d",))


def test_each_judge_reason_has_its_drop_count():
    """The belief fit counts a skipped judge score under its reason's count; `UNRANGED` keeps its text."""
    assert judges.DROP_COUNTS == {"unranged": "judge score linear without a range",
                                  "unattributed": "judge score with no judge named",
                                  "outside-range": "judge score outside its range",
                                  "nonfinite": "judge score not finite as a float",
                                  "nonnumeric": "judge score not a number"}
    assert judges.REFUSED == "judge scores of a fit refused for too many judges"


def _astra(doc, value):
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"]["ref"] == "j-astra":
            s["value"] = value
    return doc


@pytest.mark.parametrize("value", [5.5, 0.5, 1e308])
def test_a_reading_outside_its_flow_range_is_no_reading(linear, value):
    """j-astra's range is [1, 5], on the flow and on its signals: a score outside it is no reading, so the flow's
    fit and `current_quality` are those without that score, and `judged` names j-astra. A bound reads 0 or 1."""
    folder, docs = linear
    k = next(i for i, d in enumerate(docs) if "j-astra" in {j.id for j in judges.signal_judges([d], "quality")})
    out, without = list(docs), list(docs)
    out[k], without[k] = _astra(docs[k], value), _strip(docs[k], ("j-claude", "person"))
    assert judges.judged(out[k], "quality").outside_range == ["j-astra"]
    assert judges.fit(folder.flow, out) == judges.fit(folder.flow, without)
    assert judges.current_quality(out) == judges.current_quality(without)
    for bound, want in ((1, 0.0), (5, 1.0)):
        assert judges._readings(_astra(docs[k], bound), judges._flow_judges(folder.flow))["j-astra"] == want


@pytest.mark.parametrize("span", [[0, 2 ** 53], [-(2 ** 53), 0]])
def test_an_int_one_past_the_largest_bound_is_outside_its_range(linear, span):
    """At the largest bound read (`RANGE_LIMIT`, 2**53), the int one past a bound is that bound as a float, yet it
    lies outside the range: on the signal's range and on the flow's, it is no reading. The bounds read 0 and 1."""
    folder, docs = linear
    k = next(i for i, d in enumerate(docs) if "j-astra" in {j.id for j in judges.signal_judges([d], "quality")})
    flow = [dataclasses.replace(j, range=judges._range(span)) if j.id == "j-astra" else j
            for j in judges._flow_judges(folder.flow)]
    low, high = span
    for value, want in ((low - 1, None), (low, 0.0), (high, 1.0), (high + 1, None)):
        on_signal = _ranged(_astra(docs[k], value), span)
        assert judges.judged(on_signal, "quality").outside_range == ([] if want is not None else ["j-astra"])
        assert judges._readings(on_signal, judges.signal_judges([on_signal], "quality")).get("j-astra") == want
        assert judges._readings(_unranged(_astra(docs[k], value)), flow).get("j-astra") == want


def _score_rows(docs, dropped=False):
    """Run id to the score head's value, the score info, and with `dropped` the drop counts."""
    out = belief_fit.collect_rows(iter(docs), now=_now())
    heads, info = out[0], out[1]
    hr = heads.get("score:quality")
    rows = {} if hr is None else dict(zip(hr.runs, hr.y))
    return (rows, info, out[2]) if dropped else (rows, info)


def _now():
    from loopmath.belief.outcome import parse_ts

    return parse_ts(NOW)


# calibrated_runs, the fixture for reader tests


def test_calibrated_runs_differ_from_the_single_run_value(tmp_path):
    docs = calibrated_runs(tmp_path / "lit-brief")
    assert len(docs) == 20
    fit = judges.fit(load(tmp_path / "lit-brief").flow, docs)
    assert fit["calibrated"] and fit["n_person"] == 8
    every = judges.current_quality(docs)
    probe = next(d for d in docs if d["run"]["id"] == CALIBRATED_PROBE)
    own = judges.current_quality([probe])[CALIBRATED_PROBE]
    assert abs(every[CALIBRATED_PROBE][0] - own[0]) > 0.01
    assert abs(transform(every[CALIBRATED_PROBE][0], "fraction") - transform(own[0], "fraction")) > 0.0


# 8. The unpooled estimate


def _plain_unpooled(runs, scored, data, point, theta, cov):
    """Per run, the unpooled mean and variance at one grid point in plain loops: the readings less the biases,
    weighted by 1/s_j^2, and 1/D plus the biases' posterior variance through those weights."""
    index = {j.id: k for k, j in enumerate(scored)}
    g = slice(data.A + data.I, data.size)
    beta, B = data.M @ theta[g], data.M @ cov[g, g] @ data.M.T
    means, variances = [], []
    for r in runs:
        js = sorted(index[j] for j in r.values)
        d = np.array([1.0 / point[1 + j] ** 2 for j in js])
        a = d / d.sum()
        y = np.array([r.values[scored[j].id] for j in js])
        means.append(float(a @ (y - beta[js])))
        variances.append(float(1.0 / d.sum() + a @ B[np.ix_(js, js)] @ a))
    return np.array(means), np.array(variances)


@pytest.mark.parametrize("calibrated", [False, True])
def test_unpooled_matches_a_dense_computation(calibrated):
    scored, runs = _hand_runs()
    data = judges._Data(scored, runs, calibrated, True)
    points = np.array([[0.12, 0.08, 0.12, 0.03], [0.05, 0.2, 0.04, 0.01], [0.3, 0.01, 0.3, 0.1]])
    _, theta, cov = judges._evaluate(data, points, posterior=True)
    um, uv = judges._u_moments(data, points, theta, cov)
    for g, point in enumerate(points):
        _, d_theta, d_cov, _, _ = _dense(data, runs, scored, point)
        pm, pv = _plain_unpooled(runs, scored, data, point, d_theta, d_cov)
        assert np.allclose(um[g], pm, rtol=1e-10, atol=1e-10)
        assert np.allclose(uv[g], pv, rtol=1e-10, atol=1e-10)


def test_the_stored_unpooled_value_is_the_grid_mixture():
    """A fitted run's `unpooled` and `unpooled_sd` are the mixture of the per-point moments over the kept grid."""
    scored, runs = _hand_runs()
    got = judges._fit(scored, runs, None, "quality")
    data = judges._Data(scored, runs, got["calibrated"], True)
    _, _, points, logw = judges._zoomed(data, scored)
    kept = np.flatnonzero(logw >= logw.max() + math.log(judges.KEEP))
    w = np.exp(logw[kept] - logw.max())
    w /= w.sum()
    _, theta, cov = judges._evaluate(data, points[kept], posterior=True)
    m, v = judges._u_moments(data, points[kept], theta, cov)
    mean = w @ m
    sd = np.sqrt(np.maximum(0.0, w @ (v + m ** 2) - mean ** 2))
    for k, r in enumerate(runs):
        assert abs(got["runs"][r.id]["unpooled"] - mean[k]) <= 1e-10
        assert abs(got["runs"][r.id]["unpooled_sd"] - sd[k]) <= 1e-10


@pytest.mark.parametrize("n_person", [5, 6])
def test_the_unpooled_scale_follows_the_anchor(folder, n_person):
    """Below six runs with a person score the biases sum to 0, and a run the person alone scored reads its score
    less the person's bias. At six the person anchors the scale (bias 0), so that run reads the score itself."""
    rng = np.random.default_rng(11)
    docs = []
    for k in range(10):
        q = 0.4 + 0.04 * k
        values = {"j-claude": q + 0.05 + rng.normal(0, 0.03), "j-astra": q - 0.03 + rng.normal(0, 0.05)}
        if k < n_person - 1:
            values["person"] = q + rng.normal(0, 0.02)
        docs.append(_doc(f"run_a{k}", values, task=f"q{k + 1:02d}"))
    docs.append(_doc("run_p", {"person": 0.61}, task="q11"))
    fit = judges.fit(folder.flow, docs)
    assert fit["n_person"] == n_person and fit["calibrated"] == (n_person >= judges.CALIBRATED_AT)
    stored = fit["runs"]["run_p"]
    assert judges.unpooled_quality(folder.flow, docs[-1], fit) == (stored["unpooled"], stored["unpooled_sd"])
    person = _row(fit, "person")
    if fit["calibrated"]:
        assert person["bias"] == 0.0 and abs(stored["unpooled"] - 0.61) <= 1e-12
    else:
        assert abs(sum(r["bias"] for r in fit["judges"])) <= 1e-9 and abs(person["bias"]) > 1e-3
        assert abs(stored["unpooled"] - (0.61 - person["bias"])) <= 1e-9
    assert stored["unpooled_sd"] > 0


def test_a_run_far_from_its_arm_stays_unpooled(seed1, folder):
    """A run whose readings sit 0.35 below its arm and input's level, in the 200-run corpus (5 runs a cell): its
    unpooled estimate stays at its own bias-corrected readings, while the pooled combined quality is pulled toward
    the level; `current_quality`, `unpooled_quality`, `evidence` and the belief fit's score row all give the
    unpooled value."""
    from loopmath.belief.rules import evidence, resolve_rule

    docs, first = seed1
    flow = folder.flow
    level = {(lv["arm"], lv["input"]): lv["mean"] for lv in first["levels"]}[("base", "q03")]
    far = copy.deepcopy(next(d for d in docs if d["run"]["id"] == "run_s1-base-q03-1"))
    far["run"]["id"] = "run_far"
    planted = {"j-claude": TRUTH["judges"]["j-claude"]["bias"], "j-astra": TRUTH["judges"]["j-astra"]["bias"]}
    far["run"]["signals"] = [dict(s, value=round(level - 0.35 + planted[s["source"]["ref"]], 6))
                             for s in far["run"]["signals"] if s.get("kind") == "score"] \
        + [s for s in far["run"]["signals"] if s.get("kind") != "score"]
    every_doc = docs + [far]
    fit = judges.fit(flow, every_doc)
    stored = fit["runs"]["run_far"]
    values = {s["source"]["ref"]: s["value"] for s in far["run"]["signals"] if s.get("kind") == "score"}
    assert set(values) == set(planted)
    d = {j: _row(fit, j)["noise"] ** -2 for j in values}
    plug = sum(d[j] * (values[j] - _row(fit, j)["bias"]) for j in values) / sum(d.values())
    assert abs(stored["unpooled"] - plug) < 0.02 and abs(stored["unpooled"] - (level - 0.35)) < 0.05
    refit = {(lv["arm"], lv["input"]): lv["mean"] for lv in fit["levels"]}[("base", "q03")]
    q, _ = judges.combined_quality(flow, far, fit)
    assert stored["unpooled"] + 0.04 <= q < refit + 0.01
    every = judges.current_quality(every_doc)
    assert every["run_far"] == judges.unpooled_quality(flow, far, fit) == (stored["unpooled"], stored["unpooled_sd"])
    ev = evidence(far, resolve_rule(far), now=_now(), quality=every["run_far"])
    assert ev.scores["quality"] == stored["unpooled"]
    assert _score_rows(every_doc)[0]["run_far"] == transform(stored["unpooled"], "fraction")
