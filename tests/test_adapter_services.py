"""Contract tests for the adapter-to-host extraction boundary."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from loopmath import adapter_host
from loopmath.adapter_host import default_adapter_services
from loopmath.adapters import (
    Adapter,
    AdapterServices,
    PricingResult,
    Selection,
    Usage,
    VENDOR_SESSION_REASONS,
    VendorSession,
    VendorSessionResult,
    register,
)
from loopmath.cli import main


UUID = "11111111-1111-4111-8111-111111111111"


class _Adapter(Adapter):
    name = "host-contract"

    def discover(self):
        return ()

    def sessions(self):
        return ()

    def emit(self, selection: Selection):
        producer = {"name": self.name}
        if self.producer_version() is not None:
            producer["version"] = self.producer_version()
        return {"ocp": "0.2", "producer": producer, "nodes": []}


register(_Adapter)


class _LegacyNoArgumentAdapter(_Adapter):
    name = "host-contract-legacy"

    def __init__(self) -> None:
        self.constructed_without_arguments = True


register(_LegacyNoArgumentAdapter)


class _Parsed:
    def __init__(self, record: dict) -> None:
        self._record = record

    def to_dict(self) -> dict:
        return dict(self._record)


def _record(kind: str, correlate: str = UUID) -> dict:
    return {
        "run_id": f"{'cc' if kind == 'claude-code' else 'cx'}_{correlate}",
        "model": "opus-5" if kind == "claude-code" else "gpt-5.6-sol",
        "effort": "max",
        "tokens": {"in": 800, "cache_read": 200, "cache_write": 10, "out": 50},
        "ts": "2026-09-01T16:02:00Z",
        "wall_s": 60.0,
    }


def test_hook_signatures_and_no_argument_unavailable_behavior() -> None:
    assert list(inspect.signature(Adapter.__init__).parameters) == ["self", "services"]
    assert list(inspect.signature(Adapter.price_usage).parameters) == [
        "self",
        "model",
        "usage",
    ]
    assert list(inspect.signature(Adapter.configure_services).parameters) == [
        "self",
        "services",
    ]
    assert list(inspect.signature(Adapter.resolve_vendor_session).parameters) == [
        "self",
        "kind",
        "correlate",
    ]
    assert list(inspect.signature(Adapter.fixture_selection).parameters) == [
        "self",
        "fixture_dir",
    ]
    assert list(inspect.signature(Adapter.producer_version).parameters) == ["self"]

    adapter = _Adapter()
    usage = Usage(11, 12, 13, 14)
    result = adapter.price_usage("opus-5", usage)
    assert result == PricingResult(
        usage=usage,
        usd=None,
        priced=False,
        reason="host pricing service is unavailable",
        provisional=False,
        estimate_usd=None,
    )
    assert result.to_ocp() == {
        "input_tokens": 11,
        "cached_input_tokens": 12,
        "cache_creation_tokens": 13,
        "output_tokens": 14,
        "basis": "measured",
    }
    assert adapter.resolve_vendor_session("claude-code", UUID) == VendorSessionResult(
        None, "service_unavailable"
    )
    assert adapter.producer_version() is None


def test_cli_injects_default_host_services(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        adapter_host,
        "default_adapter_services",
        lambda: AdapterServices(producer_version="cli-injected"),
    )
    assert main(["adapt", "host-contract-legacy"]) == 0
    assert json.loads(capsys.readouterr().out)["producer"]["version"] == "cli-injected"


def test_injected_callables_are_narrow_and_checked() -> None:
    usage = Usage(1, 2, 3, 4)
    session = VendorSession(
        kind="claude-code",
        correlate=UUID,
        run_id=f"cc_{UUID}",
        started_at=None,
        wall_s=None,
        model="opus-5",
        model_tier="verified",
        effort=None,
        usage=usage,
        match_evidence="unique synthetic match",
    )
    calls: list[tuple] = []

    def price(model, received):
        calls.append(("price", model, received))
        return PricingResult(received, 0.25, True, None)

    def vendor(kind, correlate):
        calls.append(("vendor", kind, correlate))
        return VendorSessionResult(session, "resolved")

    adapter = _Adapter(AdapterServices(price, vendor, "9.8.7"))
    assert adapter.price_usage("opus-5", usage).to_ocp() == {
        **usage.to_ocp(),
        "usd": 0.25,
    }
    assert adapter.resolve_vendor_session(
        "claude-code", UUID
    ) == VendorSessionResult(session, "resolved")
    assert adapter.producer_version() == "9.8.7"
    assert calls == [
        ("price", "opus-5", usage),
        ("vendor", "claude-code", UUID),
    ]

    wrong = AdapterServices(
        vendor_session=lambda _kind, correlate: VendorSessionResult(
            VendorSession(
                kind="codex",
                correlate=correlate,
                run_id=f"cx_{correlate}",
                started_at=None,
                wall_s=None,
                model=None,
                model_tier=None,
                effort=None,
                usage=None,
                match_evidence="wrong kind",
            ),
            "resolved",
        )
    )
    with pytest.raises(ValueError, match="different kind"):
        _Adapter(wrong).resolve_vendor_session("claude-code", UUID)


def test_usage_and_pricing_dataclasses_reject_non_ocp_values() -> None:
    assert VENDOR_SESSION_REASONS == (
        "resolved",
        "service_unavailable",
        "unsupported_kind",
        "invalid_correlate",
        "not_found",
        "unparseable",
        "ambiguous",
    )
    with pytest.raises(ValueError, match="input_tokens"):
        Usage(-1, 0, 0, 0)
    with pytest.raises(ValueError, match="input_tokens"):
        Usage(True, 0, 0, 0)
    with pytest.raises(ValueError, match="must not carry monetary values"):
        PricingResult(Usage(0, 0, 0, 0), 0.0, False, "unknown")
    with pytest.raises(ValueError, match="requires a reason"):
        PricingResult(Usage(0, 0, 0, 0), None, False, None)
    with pytest.raises(ValueError, match="settled usd"):
        PricingResult(
            Usage(0, 0, 0, 0),
            0.0,
            True,
            None,
            provisional=True,
            estimate_usd=None,
        )
    with pytest.raises(ValueError, match="requires usage and estimate_usd"):
        PricingResult(
            Usage(0, 0, 0, 0),
            None,
            False,
            "price-table entry is provisional",
            provisional=True,
            estimate_usd=None,
        )
    with pytest.raises(ValueError, match="standard provisional reason"):
        PricingResult(
            Usage(0, 0, 0, 0),
            None,
            False,
            "estimated",
            provisional=True,
            estimate_usd=0.0,
        )
    with pytest.raises(ValueError, match="requires a session"):
        VendorSessionResult(None, "resolved")


def test_dagr_pricing_hook_returns_exact_ocp_cost_and_honest_unknowns() -> None:
    assert default_adapter_services().producer_version == adapter_host.__version__
    adapter = _Adapter(default_adapter_services(producer_version="test-version"))
    usage = Usage(800, 200, 0, 50)
    priced = adapter.price_usage("gpt-5.6-sol", usage)
    assert priced.priced is True
    assert priced.reason is None
    assert priced.to_ocp() == {
        "input_tokens": 800,
        "cached_input_tokens": 200,
        "cache_creation_tokens": 0,
        "output_tokens": 50,
        "basis": "measured",
        "usd": pytest.approx(0.00428),
    }

    unknown = adapter.price_usage("unknown-model", usage)
    assert unknown.priced is False
    assert unknown.usd is None
    assert "no price entry" in (unknown.reason or "")
    assert unknown.to_ocp() == usage.to_ocp()

    missing = adapter.price_usage("gpt-5.6-sol", None)
    assert missing.priced is False
    assert missing.to_ocp() is None
    assert "no complete token measurements" in (missing.reason or "")
    assert adapter.producer_version() == "test-version"


def test_todo_price_is_explicitly_provisional_and_omitted_from_ocp() -> None:
    adapter = _Adapter(default_adapter_services(producer_version="test-version"))
    usage = Usage(1_000_000, 0, 0, 0)
    result = adapter.price_usage("gpt-5.5", usage)

    assert result.priced is False
    assert result.usd is None
    assert result.provisional is True
    assert result.estimate_usd == pytest.approx(2.5)
    assert result.reason == "price-table entry is provisional"
    assert result.to_ocp() == usage.to_ocp()
    assert "usd" not in result.to_ocp()
    assert "estimate_usd" not in result.to_ocp()


def test_claude_vendor_hook_normalizes_unique_record_and_rejects_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "claude"
    first = root / f"{UUID}.jsonl"
    first.parent.mkdir()
    first.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        adapter_host.claude_code,
        "parse_session",
        lambda _path: _Parsed(_record("claude-code")),
    )

    adapter = _Adapter(
        default_adapter_services(
            claude_code_root=root,
            codex_root=tmp_path / "no-codex",
            producer_version="test",
        )
    )
    result = adapter.resolve_vendor_session("claude-code", UUID.upper())
    assert result.reason == "resolved"
    assert result.session == VendorSession(
        kind="claude-code",
        correlate=UUID,
        run_id=f"cc_{UUID}",
        started_at="2026-09-01T16:02:00Z",
        wall_s=60.0,
        model="opus-5",
        model_tier="verified",
        effort="max",
        usage=Usage(800, 200, 10, 50),
        match_evidence="one correlate matched exactly one parseable vendor session",
    )
    assert adapter.resolve_vendor_session("other", UUID).reason == "unsupported_kind"
    assert (
        adapter.resolve_vendor_session("claude-code", "not-a-uuid").reason
        == "invalid_correlate"
    )
    assert (
        adapter.resolve_vendor_session(
            "claude-code", "22222222-2222-4222-8222-222222222222"
        ).reason
        == "not_found"
    )

    unparseable_root = tmp_path / "unparseable-claude"
    unparseable_root.mkdir()
    (unparseable_root / f"{UUID}.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(adapter_host.claude_code, "parse_session", lambda _path: None)
    unparseable = _Adapter(
        default_adapter_services(
            claude_code_root=unparseable_root,
            codex_root=tmp_path / "no-codex",
            producer_version="test",
        )
    )
    assert unparseable.resolve_vendor_session("claude-code", UUID).reason == "unparseable"

    second = root / "copy" / f"{UUID}.jsonl"
    second.parent.mkdir()
    second.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        adapter_host.claude_code,
        "parse_session",
        lambda _path: _Parsed(_record("claude-code")),
    )
    fresh = _Adapter(
        default_adapter_services(
            claude_code_root=root,
            codex_root=tmp_path / "no-codex",
            producer_version="test",
        )
    )
    assert fresh.resolve_vendor_session("claude-code", UUID).reason == "ambiguous"


def test_codex_vendor_hook_groups_continuations_by_correlate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "codex"
    paths = [root / "a.jsonl", root / "b.jsonl"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"session_id": UUID},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    received: list[Path] = []

    def parse(session_paths):
        received.extend(session_paths)
        return _Parsed(_record("codex"))

    monkeypatch.setattr(adapter_host.codex, "parse_session", parse)
    adapter = _Adapter(
        default_adapter_services(
            claude_code_root=tmp_path / "no-claude",
            codex_root=root,
            producer_version="test",
        )
    )
    result = adapter.resolve_vendor_session("codex", UUID)
    assert result.reason == "resolved"
    session = result.session
    assert session is not None
    assert session.run_id == f"cx_{UUID}"
    assert session.model == "gpt-5.6-sol"
    assert session.model_tier == "reported"
    assert received == paths


def test_adapter_package_modules_do_not_import_dagr_internals() -> None:
    adapters_dir = Path(__file__).resolve().parents[1] / "src" / "loopmath" / "adapters"
    forbidden = {"__version__", "graph", "ingest", "price", "pricing", "report"}
    for path in adapters_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                parts.extend(alias.name for alias in node.names)
                assert not forbidden.intersection(parts)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    assert not forbidden.intersection(parts)
