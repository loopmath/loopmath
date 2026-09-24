"""Contract tests for ``loopmath verify-receipts`` (E2 task T10)."""

import hashlib
import json

from loopmath.cli import main


def _fixture(tmp_path, rows):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("# Frozen experiment\n", encoding="utf-8")
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    (tmp_path / "FROZEN.txt").write_text(f"{digest}  {prereg.name}\n", encoding="utf-8")
    ledger = tmp_path / "e2-receipts.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return ledger, prereg


def _row(timestamp, spend, *, task="T10", arm="X", cap=1):
    return {
        "timestamp": timestamp,
        "task": task,
        "arm": arm,
        "spend_usd": spend,
        "cap_usd": cap,
    }


def test_clean_ledger_prints_ok_summary(tmp_path, capsys):
    ledger, _ = _fixture(
        tmp_path,
        [
            _row("2026-08-31T10:00:00Z", 0.25),
            _row("2026-08-31T10:01:00+00:00", 0.50),
            _row("2026-08-31T10:02:00Z", 1.00, task="T11", arm="Y", cap=2),
        ],
    )

    assert main(["verify-receipts", str(ledger)]) == 0
    assert capsys.readouterr().out == (
        "OK: 3 receipts, 2 task-arms, $1.75 spend; "
        "prereg hash verified (PREREG.md)\n"
    )


def test_tampered_prereg_reports_hash_mismatch(tmp_path, capsys):
    ledger, prereg = _fixture(tmp_path, [_row("2026-08-31T10:00:00Z", 0.25)])
    prereg.write_text("tampered\n", encoding="utf-8")

    assert main(["verify-receipts", str(ledger)]) == 1
    assert capsys.readouterr().out == "FAIL: prereg hash mismatch for PREREG.md\n"


def test_cumulative_task_arm_spend_over_cap_is_first_violation(tmp_path, capsys):
    ledger, _ = _fixture(
        tmp_path,
        [
            _row("2026-08-31T10:00:00Z", 0.60),
            _row("2026-08-31T10:01:00Z", 0.41),
        ],
    )

    assert main(["verify-receipts", str(ledger)]) == 1
    assert capsys.readouterr().out == (
        "FAIL: line 2: task-arm T10/X spend $1.01 exceeds cap $1.00\n"
    )


def test_out_of_order_timestamp_is_first_violation(tmp_path, capsys):
    ledger, _ = _fixture(
        tmp_path,
        [
            _row("2026-08-31T10:01:00Z", 0.25),
            _row("2026-08-31T10:00:00Z", 0.25),
        ],
    )

    assert main(["verify-receipts", str(ledger)]) == 1
    assert capsys.readouterr().out == (
        "FAIL: line 2: timestamp 2026-08-31T10:00:00Z is before "
        "line 1 timestamp 2026-08-31T10:01:00Z\n"
    )


def test_schema_violation_is_reported_with_line_number(tmp_path, capsys):
    ledger, _ = _fixture(tmp_path, [{"timestamp": "2026-08-31T10:00:00Z"}])

    assert main(["verify-receipts", str(ledger)]) == 1
    assert capsys.readouterr().out == "FAIL: line 1: missing required field 'arm'\n"
