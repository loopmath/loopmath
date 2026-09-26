"""The results page names the starting prior once, under the lede (lane 23P); 23N owns the rest of the page."""

from __future__ import annotations

import json

import loopmath
from loopmath.priors.show import starting_prior
from loopmath.views import posterior as P
from tests.views._posterior_helpers import FIT_ID, FakeState

REPO = "acme/bench"
NOW = "2026-09-24T19:00:00-07:00"
MARK = '<img src=x onerror="window.__lmMark=(window.__lmMark||0)+1">'
READ = """(() => {
  const p = document.querySelectorAll('#prior');
  return {n: p.length, text: p.length ? p[0].textContent : null, after: p.length ? p[0].previousElementSibling.id : null,
    count: document.getElementById('app').textContent.split('Starting prior:').length - 1, mark: window.__lmMark || 0};
})()"""


def _data(tmp_path, meta=None):
    home = tmp_path / "home"
    (home / "fits" / FIT_ID).mkdir(parents=True)
    if meta is not None:
        (home / "fits" / FIT_ID / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return P.build_view(FakeState(), home=home, task_type="feature", repo=REPO, now=NOW)


def test_the_page_payload_carries_the_starting_prior(tmp_path):
    data = _data(tmp_path)
    assert data["prior"] == starting_prior()
    assert P.extract_data(P.render(data))["prior"]["line"] == starting_prior()["line"]


def test_the_page_names_the_fits_prior_when_the_fit_left_shipped_runs_out(tmp_path):
    shipped = starting_prior()["runs_by_source"]
    first = next(iter(shipped))
    used = {k: n for k, n in shipped.items() if k != first}
    meta = {"code_version": loopmath.__version__, "options": {"without": [first]},
            "runs_by_source": {**used, "user": 2}}
    line = _data(tmp_path, meta)["prior"]["line"]
    counts = ", ".join(f"{k} {n:,}" for k, n in used.items())
    assert line == (f"Starting prior of this fit: loopmath {loopmath.__version__}, {sum(used.values()):,} runs "
                    f"({counts}), with benchmarks. loopmath prior show lists the installed prior.")


def test_the_page_shows_the_line_once_under_the_lede_as_text(tmp_path, probe):
    data = _data(tmp_path)
    data["prior"]["line"] += MARK
    page = tmp_path / "results-prior.html"
    page.write_text(P.render(data), encoding="utf-8")
    got = probe(page, READ)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    assert got["result"] == {"n": 1, "text": data["prior"]["line"], "after": "lede", "count": 1, "mark": 0}


def test_a_page_without_the_prior_draws_no_line(tmp_path, probe):
    data = _data(tmp_path)
    del data["prior"]
    page = tmp_path / "results-noprior.html"
    page.write_text(P.render(data), encoding="utf-8")
    got = probe(page, READ)
    assert got["exceptions"] == [] and got["console_errors"] == [], got
    assert got["result"]["n"] == 0 and got["result"]["count"] == 0
