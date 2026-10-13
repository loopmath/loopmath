"""`loopmath share` carries the outcome under `outcome.q.<tier>` from config (0.2.6, with fix (1)).

Before 0.2.6 an export carried the built-in q while `run finish` and, from 0.2.6, the fit used the
configured one. Unset, the export is 0.2.5's. The planted store of share_store.py, nothing real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from loopmath import cli


def _outcomes(capsys, home) -> list[tuple]:
    assert cli.main(["share", "--preview", "--home", str(home)]) == 0
    obj = json.loads(capsys.readouterr().out)
    found = [d["run"]["ext"]["dev.loopmath.share"]["outcome"] for d in obj["runs"]]
    return sorted(((o["z"], o["q"], o["tier"]) for o in found), key=repr)


def test_share_carries_the_configured_q(store, capsys):
    before = _outcomes(capsys, store)
    assert any(tier == "verified" for _, _, tier in before)
    assert cli.main(["config", "set", "outcome.q.verified", "0.9", "--home", str(store)]) == 0
    capsys.readouterr()
    after = _outcomes(capsys, store)
    assert after == sorted(((z, 0.9 if tier == "verified" else q, tier) for z, q, tier in before), key=repr)


BROKEN = ["not TOML", "no permission", "a directory", "not UTF-8"]


def break_config(home: Path, how: str) -> Path:
    """Make `home/config.toml` unreadable one way."""
    path = home / "config.toml"
    if path.is_file():
        path.unlink()
    if how == "a directory":
        path.mkdir()
    elif how == "not UTF-8":
        path.write_bytes(b"\xff\xfeoutcome = 1\n")
    else:
        path.write_text("outcome = [\n" if how == "not TOML" else "", encoding="utf-8")
        if how == "no permission":
            path.chmod(0)
            if os.access(path, os.R_OK):
                path.chmod(0o644)
                pytest.skip("this user reads a file with mode 000")
    return path


def _preview(capsys, home) -> tuple[int, dict | None, str]:
    code = cli.main(["share", "--preview", "--home", str(home)])
    out, err = capsys.readouterr()
    obj = json.loads(out) if code == 0 else None
    if obj:
        obj.pop("created_at")
    return code, obj, err


@pytest.mark.parametrize("how", [b for b in BROKEN if b != "not UTF-8"])
def test_an_unreadable_config_keeps_the_built_in_q_and_says_so(store, capsys, how):
    _, unset, _ = _preview(capsys, store)
    path = break_config(store, how)
    try:
        code, obj, err = _preview(capsys, store)
    finally:
        if path.is_file():
            path.chmod(0o644)
    assert code == 0 and "Traceback" not in err  # as before 0.2.6, the share goes on
    assert obj["runs"] == unset["runs"]  # the built-in q; `org_hash` follows the unreadable `org`, as in 0.2.5
    (line,) = [ln for ln in err.splitlines() if "outcomes use the built-in q" in ln]
    assert line.startswith(f"warn: {path}: ") and line.endswith("; outcomes use the built-in q")


def test_a_config_that_is_not_utf8_stops_a_share_as_before(store, capsys):
    # 0.2.5's `store_org` cannot decode it either: the share exits 1 with "cannot read the store", no traceback.
    break_config(store, "not UTF-8")
    code, _, err = _preview(capsys, store)
    assert code == 1 and "Traceback" not in err and "cannot read the store" in err
