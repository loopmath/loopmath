"""The one number rule of harness factors: a store with no harness record gives 0.2.8's numbers exactly.

Two guards:
- pinned: the fit's seed key, head seeds and design nodes on simulated stores without records, recorded
  from 0.2.8. They hash the rows every head is fitted on, so any changed row moves them; they do not
  depend on the machine's floating point;
- the CLI: `fit`, `recommend` and `posterior --json` from a 0.2.8 checkout (`LOOPMATH_BASELINE_SRC`, its
  `src` folder) and from this tree, on the same stores, compared leaf by leaf. Each side runs in its own temp
  folder, so in every string that folder becomes `<root>` and the new fit's id `<fit>`: a path compares by its
  place in the store, wherever the checkout and the temp folder are. It runs on a fresh store, and on a temp
  copy of a store named by `LOOPMATH_COMPARE_STORE` (a private one stays outside the suite), and skips
  without `LOOPMATH_BASELINE_SRC`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "belief"))
import simdata  # noqa: E402

from loopmath.belief import fit as F  # noqa: E402

NOW = datetime.fromisoformat("2026-10-18T12:00:00-07:00")
# Recorded from loopmath 0.2.8 (the release merge) on `_docs` below; both stores give the same values.
PIN_028 = {"seed_key": "13dbca953f2169503a46fb35fdc638e2dcbcbe4e6d3c1725c2e878d1e2159b8a",
           "seeds": {"cost": 2345460883, "gate": 4247663959, "rate": 2926494645, "success": 3997672742},
           "nodes": "ea0980bd5c46e570562062f502b570da7d4a99893ea1e706963ce326abf94a0e"}


def _docs(other_ext: bool) -> list[dict]:
    docs = simdata.simulate(240, seed=19, source="user", n_tasks=60)[0]
    if other_ext:
        docs = copy.deepcopy(docs)
        for i, d in enumerate(docs):  # a third party's ext keys, and a name close to the record's
            d["run"]["ext"]["org.example.tool"] = {"stage": f"s{i % 3}"}
            d["run"]["ext"]["harnessed"] = True
    return docs


@pytest.mark.parametrize("other_ext", [False, True])
def test_a_store_without_records_fits_on_0_2_8_rows(tmp_path, other_ext):
    path = Path(F.fit(tmp_path, docs=_docs(other_ext), no_prior=True, now=NOW))
    meta = json.loads((path / "meta.json").read_text())
    design = json.loads((path / "design.json").read_text())
    assert meta["seed_key"] == PIN_028["seed_key"]
    assert {h: meta["heads"][h]["seed"] for h in sorted(meta["heads"])} == PIN_028["seeds"]
    assert hashlib.sha256(json.dumps(design["nodes"], sort_keys=True).encode()).hexdigest() == PIN_028["nodes"]
    assert "harness_variants" not in design
    assert not any(n["id"].startswith("hfactor:") for n in design["nodes"])


# ---------------------------------------------------------------- the CLI against a 0.2.8 checkout

BASELINE = os.environ.get("LOOPMATH_BASELINE_SRC")
STORE = os.environ.get("LOOPMATH_COMPARE_STORE")
HERE = Path(__file__).resolve().parents[2] / "src"
# Masked by exact path per command: wall-clock values and ids made from the wall clock (the fit id is its second, a
# task or recommendation id a ULID); none is a number the model computes. Paths are normalized, not masked
# (`_normalized`).
_FIT = (r"^fit\.id$", r"^fit\.at$")
VOLATILE = {"fit": (r"^seconds$", *_FIT),
            "recommend": (r"^rec$", r"^created_at$", r"^fit\.age_s$", r"^search\.seconds$", r"^task\.id$",
                          r"^task\.group_chain\.2\.1$", *_FIT),
            "posterior": (r"^generated_at$", r"^task\.rec$", r"^data\.fit_time_s$", r"^results\.rescue\.rec$", *_FIT)}
EXPECT = {"fit": ("fit",), "recommend": ("task", "rule", "fit", "choices", "search"),
          "posterior": ("schema", "fit", "task", "levels", "heads")}
TASK = ["--type", "bug_fix", "--repo", "acme/api", "--title", "Fix the parser crash"]


def _leaves(value, masks=(), path=(), out=None):
    """Every leaf by its path; a dict also leaves `#dict` (its sorted keys) and a list `#list` (its length), so a
    missing key, an empty list and an empty object differ. A path matching a mask is `<volatile>`."""
    out = {} if out is None else out
    key = ".".join(path)
    if any(re.search(m, key) for m in masks):
        out[key] = "<volatile>"
    elif isinstance(value, dict):
        out[key + "#dict"] = sorted(value)
        for k, child in value.items():
            _leaves(child, masks, (*path, str(k)), out)
    elif isinstance(value, list):
        out[key + "#list"] = len(value)
        for i, child in enumerate(value):
            _leaves(child, masks, (*path, str(i)), out)
    else:
        out[key] = value
    return out


def _normalized(value, root: Path, fit_id: str):
    """`value` with the temp folder `root` (as given or resolved) as `<root>` and the new fit's id as `<fit>` in
    every string, keys included."""
    assert fit_id
    subs = [(s, "<root>") for s in sorted({str(root.resolve()), str(root)}, key=len, reverse=True)]
    subs.append((fit_id, "<fit>"))

    def norm(v):
        if isinstance(v, dict):
            return {norm(k): norm(c) for k, c in v.items()}
        if isinstance(v, list):
            return [norm(c) for c in v]
        if isinstance(v, str):
            for old, new in subs:
                v = v.replace(old, new)
        return v

    return norm(value)


def test_the_flattening_tells_empty_containers_and_missing_keys_apart():
    assert len({json.dumps(_leaves(v), sort_keys=True) for v in ({"x": []}, {}, {"x": {}}, {"x": None})}) == 4
    masked = _leaves({"generated_at": 1, "x": {"generated_at": 2}}, VOLATILE["posterior"])
    assert masked["generated_at"] == "<volatile>" and masked["x.generated_at"] == 2


def test_paths_compare_by_their_place_in_the_store(tmp_path):
    def payload(root, fit_id):
        return {"path": f"{root}/store/fits/{fit_id}", "n": 3, "notes": [f"{root}/store/runs"], f"{root}/x": True}

    before, after = tmp_path / "before", tmp_path / "after"
    want = {"path": "<root>/store/fits/<fit>", "n": 3, "notes": ["<root>/store/runs"], "<root>/x": True}
    assert _normalized(payload(before, "fit_20261018120000"), before, "fit_20261018120000") == want
    assert _normalized(payload(after.resolve(), "fit_20261018120002"), after, "fit_20261018120002") == want
    assert _normalized(payload(before / "elsewhere", "fit_20261018120000"), before, "fit_20261018120000") != want


def _run(src: Path, home: Path, cache: Path, *argv: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(src), "LOOPMATH_HOME": str(home), "LOOPMATH_CACHE_DIR": str(cache),
           "HOME": str(home.parent / "home"), "NO_COLOR": "1"}
    (home.parent / "home").mkdir(exist_ok=True)
    out = subprocess.run([sys.executable, "-m", "loopmath", *argv, "--json"], env=env, capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, (src, argv, out.returncode, out.stderr[-2000:])
    payload = json.loads(out.stdout)
    assert isinstance(payload, dict) and all(k in payload for k in EXPECT[argv[0]]), (src, argv, sorted(payload))
    return payload


def _collect(src: Path, store: Path | None, root: Path) -> dict:
    home = root / "store"
    if store is not None:
        shutil.copytree(store, home, symlinks=True, ignore=shutil.ignore_patterns("views", "lock"))
    else:
        home.mkdir(parents=True)
    cache = root / "cache"
    out = {"fit": _run(src, home, cache, "fit"), "recommend": _run(src, home, cache, "recommend", *TASK),
           "posterior": _run(src, home, cache, "posterior", "--type", "bug_fix", "--repo", "acme/api")}
    out = _normalized(out, root, out["fit"]["fit"]["id"])
    assert out["fit"]["path"] == "<root>/store/fits/<fit>", out["fit"]["path"]
    return out


@pytest.mark.skipif(not BASELINE, reason="set LOOPMATH_BASELINE_SRC to the src folder of a 0.2.8 checkout")
@pytest.mark.parametrize("which", ["fresh", "store"])
def test_cli_numbers_equal_0_2_8(tmp_path, which):
    if which == "store" and not STORE:
        pytest.skip("set LOOPMATH_COMPARE_STORE to a store to copy")
    store = Path(STORE) if which == "store" else None
    before = _collect(Path(BASELINE), store, tmp_path / "before")
    after = _collect(HERE, store, tmp_path / "after")
    for cmd in before:
        a, b = _leaves(before[cmd], VOLATILE[cmd]), _leaves(after[cmd], VOLATILE[cmd])
        differ = sorted(k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>"))
        assert not differ, (cmd, differ[:20])
