"""0.2.4 P3-12: the recommend text's exploration lines give the payback the summary and the pair choice give: one
number, one rounding. The fakes are test_recommend_command's."""

from __future__ import annotations

import re

from test_recommend_command import env, run_json  # noqa: F401 (env is a fixture)


def test_the_payback_in_the_list_is_the_summarys(env, capsys):
    argv = ["recommend", "--type", "feature", "--repo", "acme/app"]
    _, obj = run_json(capsys, [*argv, "--json"])
    text = run_json(capsys, argv)[1]["_text"]
    listed = [int(n) for n in re.findall(r"best value: .*pays back after (\d+) similar runs?", text)]
    said = [int(n) for n in re.findall(r"pays for itself after about (\d+) similar runs?", obj["message"])]
    assert listed and said and listed[0] == said[0]
    pair = next((c for c in obj["choices"] if c["key"] == "pair"), None)
    if pair is not None:
        assert pair["payback_runs"] == listed[0]
    assert not re.search(r"payback \d+\.\d runs", text)
