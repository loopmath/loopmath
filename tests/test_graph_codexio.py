"""A2: codex-side writes, reads and origin (`graph/codexio.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loopmath.graph import codexio, codexio_parse
from loopmath.graph.codexio import (
    codex_output_paths,
    exec_commands_from_js,
    output_writes_from_launch,
    patch_paths,
    patches_from_js,
    scan_codex_session,
)

FIXTURE = Path(__file__).parent / "fixtures" / "graph" / "codex_rollout.jsonl"
T = "2026-08-31T10:00:{:02d}.000Z".format


def _fake_writes(command, ts, cwd=None, *, exclusions=None):
    """Stand-in for A1's recognizer: pins what codexio hands to bashwrites."""
    out = []
    for line in command.split("\n"):
        line = line.strip()
        if line.startswith("cat > "):
            out.append({"ts": ts, "path": f"{cwd}/{line.split()[2]}", "tier": "heuristic", "how": "redirect", "cwd_seen": cwd})
        if line == "cp only-source" and exclusions is not None:
            exclusions["invalid_cp_mv"] += 1
    return out


def _fake_reads(command, ts, cwd=None, *, exclusions=None):
    out = []
    for line in command.split("\n"):
        line = line.strip()
        if line.startswith("sed -n") or (line.startswith("cat ") and ">" not in line):
            out.append({"ts": ts, "path": f"{cwd}/{line.split()[-1]}", "tier": "heuristic", "how": line.split()[0], "cwd_seen": cwd})
    return out


@pytest.fixture
def fake_a1(monkeypatch):
    monkeypatch.setattr(codexio.bashwrites, "writes_from_command", _fake_writes)
    monkeypatch.setattr(codexio.bashwrites, "reads_from_command", _fake_reads)


def _rollout(tmp_path, *records: str) -> Path:
    p = tmp_path / "r.jsonl"
    p.write_text("".join(r + "\n" for r in records))
    return p


def test_origin_is_reported_tier_from_session_meta():
    out = scan_codex_session(FIXTURE)
    assert out["origin"] == {
        "originator": "codex_exec",
        "source": "exec",
        "cwd": "/work/ws",
        "cli_version": "0.144.1",
        "missing": [],
        "tier": "reported",
    }


def test_apply_patch_tier_follows_the_matched_result():
    out = scan_codex_session(FIXTURE)
    patch_writes = [w for w in out["writes"] if w["how"].startswith("apply_patch")]
    assert patch_writes == [
        # call_2: direct call answered with the Success banner.
        {"ts": T(10), "path": "/work/ws/src/new_module.py", "tier": "verified", "how": "apply_patch add"},
        {"ts": T(10), "path": "/work/ws/src/old_module.py", "tier": "verified", "how": "apply_patch update"},
        # call_6 (failed) emits nothing.
        # call_7: wrapped in an exec script, the script output lists the path under the banner.
        {"ts": T(34), "path": "/work/ws/docs/wrapped.md", "tier": "verified", "how": "apply_patch add, result lists path"},
        # call_8: wrapped, script completed but never printed the patch result.
        {"ts": T(36), "path": "/work/ws/src/unprinted.py", "tier": "heuristic", "how": "apply_patch update, script completed", "unconfirmed": True},
        # call_9: envelope inside a shell command, exit code 0.
        {"ts": T(38), "path": "/work/ws/src/mixed.py", "tier": "verified", "how": "apply_patch update"},
        # call_11: wrapped, a straight-line script whose result is a bare `Script completed`: nothing names the path.
        {"ts": "2026-08-31T10:00:38.500Z", "path": "/work/ws/docs/silent.md", "tier": "heuristic", "how": "apply_patch add, script completed", "unconfirmed": True},
        # call_10: no result record at all.
        {"ts": T(39), "path": "/work/ws/src/cut_off.py", "tier": "heuristic", "how": "apply_patch add, no result", "unconfirmed": True},
        # call_13: decoys in a string and comment are ignored; the real call is confirmed.
        {"ts": T(41), "path": "/work/ws/docs/real-wrapper.md", "tier": "verified", "how": "apply_patch add, result lists path"},
        # call_16: the first call succeeded before the wrapper failed on the second.
        {"ts": T(45), "path": "/work/ws/first-confirmed.py", "tier": "verified", "how": "apply_patch add, result lists path"},
    ]
    m = out["meta"]
    assert m["patch_calls"] == 11
    assert m["patch_deletes"] == 1
    assert m["patch_calls_failed"] == 2  # direct call_6 and only call_16's unconfirmed second patch
    assert m["patch_calls_unmatched"] == 1
    assert m["patch_writes_unconfirmed"] == 3
    assert m["unparsed_patch_calls"] == 1
    assert any(s.startswith("apply_patch verification failed") for s in m["samples"])


def test_fixture_relative_patch_path_is_lexically_normalized_against_cwd():
    out = scan_codex_session(FIXTURE)
    writes = [w for w in out["writes"] if w["ts"] == T(34)]
    assert writes == [
        {"ts": T(34), "path": "/work/ws/docs/wrapped.md", "tier": "verified", "how": "apply_patch add, result lists path"}
    ]
    assert not any("/./" in w["path"] or "/../" in w["path"] for w in writes)


def test_fixture_variable_patch_uses_nearest_executable_preceding_binding():
    out = scan_codex_session(FIXTURE)
    writes = [w for w in out["writes"] if w["ts"] == T(41) and w["how"].startswith("apply_patch")]
    assert writes == [
        {"ts": T(41), "path": "/work/ws/docs/real-wrapper.md", "tier": "verified", "how": "apply_patch add, result lists path"}
    ]
    assert not any("decoy" in w["path"] or "stale" in w["path"] for w in writes)


def test_shell_calls_go_through_bashwrites_with_workdir(fake_a1):
    out = scan_codex_session(FIXTURE)
    shell_writes = [w for w in out["writes"] if w["how"] == "redirect"]
    # exec_command JSON arguments: the call's own workdir wins over the session cwd.
    assert shell_writes == [
        {"ts": T(20), "path": "/work/ws/sub/reviews/verdict.md", "tier": "heuristic", "how": "redirect", "cwd_seen": "/work/ws/sub"},
        {"ts": T(44), "path": "/work/ws/reviews/codex-output.md", "tier": "heuristic", "how": "redirect", "cwd_seen": "/work/ws"},
    ]
    assert out["reads"] == [
        # custom `exec` JS with bare keys.
        {"ts": T(5), "path": "/work/ws/docs/PLAN.md", "tier": "heuristic", "how": "sed", "cwd_seen": "/work/ws"},
        # `["bash","-lc",cmd]` unwrapped, no workdir so the session cwd; the result reported exit 1.
        {"ts": T(25), "path": "/work/ws/notes.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/work/ws", "failed": True},
        # the shell command around an embedded apply_patch envelope is still scanned.
        {"ts": T(38), "path": "/work/ws/after.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/work/ws"},
        # call_12: list-form result of an exec script; the exit code is inside the joined text.
        {"ts": "2026-08-31T10:00:38.700Z", "path": "/work/ws/missing.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/work/ws", "failed": True},
        # call_13: two real call objects retain their own property-order-independent cwd.
        {"ts": T(41), "path": "/work/first/real-one.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/work/first"},
        {"ts": T(41), "path": "/work/second/real-two.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/work/second"},
    ]
    # call_4 (string result, `Process exited with code 1`) and call_12 (list result, embedded `"exit_code":1`).
    assert out["meta"]["shell_calls_failed"] == 3  # plus fixture call_14's invalid cp


def test_unparseable_exec_script_is_counted_not_dropped():
    out = scan_codex_session(FIXTURE)
    m = out["meta"]
    assert m["shell_calls"] == 10  # original six, call_13's two calls, call_14, call_15
    assert m["unparsed_shell_calls"] == 1
    assert any(s.startswith("const cmds = ") for s in m["samples"])
    assert m["records"] == 35
    assert m["origin_missing_session_meta"] == 0


def test_exec_commands_from_js_handles_bare_keys_and_commas_in_strings():
    script = 'const r = await tools.exec_command({cmd:"echo \\"a, b: c\\" > out.txt", workdir:"/w", yield_time_ms:10000}); text(r.output)'
    calls, unparsed = exec_commands_from_js(script)
    assert unparsed == 0
    assert calls == [{"cmd": 'echo "a, b: c" > out.txt', "workdir": "/w", "yield_time_ms": 10000}]
    calls, unparsed = exec_commands_from_js("for (const c of cmds) { await tools.exec_command({cmd: c}); }")
    assert calls == [] and unparsed == 1
    # Single-quoted JS strings, with an escaped quote and a bare double quote inside.
    calls, unparsed = exec_commands_from_js("await tools.exec_command({cmd: 'echo \"it\\'s\" > f', workdir: '/w'})")
    assert unparsed == 0 and calls == [{"cmd": 'echo "it\'s" > f', "workdir": "/w"}]


def test_js_tool_names_in_strings_and_comments_are_not_calls():
    script = (
        'const example = "tools.exec_command({cmd: \'cat string.txt\'})";\n'
        "// await tools.exec_command({cmd: 'cat line-comment.txt'});\n"
        "/* await tools.apply_patch('*** Begin Patch\\n*** Add File: comment.md\\n*** End Patch'); */\n"
        "await tools.exec_command({workdir: '/one', cmd: 'cat real.txt'});\n"
        "await tools.apply_patch('*** Begin Patch\\n*** Add File: real.md\\n*** End Patch');\n"
    )
    calls, shell_unparsed = exec_commands_from_js(script)
    patches, patch_unparsed = patches_from_js(script)
    assert calls == [{"workdir": "/one", "cmd": "cat real.txt"}]
    assert shell_unparsed == patch_unparsed == 0
    assert [patch_paths(p) for p in patches] == [([("real.md", "apply_patch add")], 0)]


def test_exec_call_objects_keep_workdir_when_property_order_reverses():
    calls, unparsed = exec_commands_from_js(
        "await tools.exec_command({cmd: 'cat one.txt', workdir: '/one'}); "
        "await tools.exec_command({workdir: '/two', cmd: 'cat two.txt'});"
    )
    assert unparsed == 0
    assert calls == [
        {"cmd": "cat one.txt", "workdir": "/one"},
        {"workdir": "/two", "cmd": "cat two.txt"},
    ]


def test_patches_from_js_resolves_literals_objects_and_bindings():
    script = (
        'const patch = "*** Begin Patch\\n*** Update File: a.py\\n@@\\n-x\\n+y \\"q\\"\\n*** End Patch";\n'
        "for (let i = 0; i < 2; i++) { await tools.apply_patch(patch); }\n"
        "let del = `*** Begin Patch\\n*** Delete File: b.py\\n*** End Patch`;\n"
        "text(await tools.apply_patch(del));\n"
        "await tools.apply_patch({input: '*** Begin Patch\\n*** Add File: c.py\\n+1\\n*** End Patch'});\n"
        'await tools.apply_patch("*** Begin Patch\\n*** Add File: d.py\\n+1\\n*** End Patch");\n'
        "const r = await tools.exec_command({cmd: 'git diff -R'}); await tools.apply_patch(r.output);\n"
        "await tools.apply_patch(`*** Begin Patch\\n*** Add File: ${name}\\n*** End Patch`);\n"
    )
    patches, unparsed = patches_from_js(script)
    assert unparsed == 2  # r.output and the template literal with a substitution
    # One call per source occurrence: the loop body counts once, not per iteration.
    assert [patch_paths(p) for p in patches] == [
        ([("a.py", "apply_patch update")], 0),
        ([], 1),
        ([("c.py", "apply_patch add")], 0),
        ([("d.py", "apply_patch add")], 0),
    ]
    assert patches[0].endswith('+y "q"\n*** End Patch')


def _exec_rollout(tmp_path, script: str, output) -> Path:
    import json

    return _rollout(
        tmp_path,
        '{"timestamp":"t0","type":"session_meta","payload":{"cwd":"/w","originator":"codex","source":"cli","cli_version":"0.1"}}',
        json.dumps({"timestamp": "t1", "type": "response_item", "payload": {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": script}}),
        json.dumps({"timestamp": "t2", "type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "c1", "output": output}}),
    )


COMPLETED = [{"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"}, {"type": "input_text", "text": "{}"}]


def test_bare_script_completed_leaves_wrapped_patches_heuristic(tmp_path):
    # The shape of nearly every real wrapped call: a `const` bound to the patch, one call, no branches.
    # `Script completed` says the script ran to its end; nothing in the result names the path, so the
    # write stays heuristic (no rule about the script's shape upgrades it) and `how` says why.
    script = (
        'const patch = String.raw`*** Begin Patch\n*** Add File: docs/PRD.md\n+if (x) { try } ? :\n*** End Patch`;\n'
        'const second = "*** Begin Patch\\n*** Update File: /w/src/a.py\\n@@\\n-1\\n+2\\n*** End Patch";\n'
        "await tools.apply_patch(patch);\ntext(await tools.apply_patch(second));\n"
    )
    out = scan_codex_session(_exec_rollout(tmp_path, script, COMPLETED))
    assert out["writes"] == [
        {"ts": "t1", "path": "/w/docs/PRD.md", "tier": "heuristic", "how": "apply_patch add, script completed", "unconfirmed": True},
        {"ts": "t1", "path": "/w/src/a.py", "tier": "heuristic", "how": "apply_patch update, script completed", "unconfirmed": True},
    ]
    assert out["meta"]["patch_calls"] == 2 and out["meta"]["patch_writes_unconfirmed"] == 2
    # The same script whose result lists one path under the banner: that path alone is verified.
    listed = COMPLETED + [{"type": "input_text", "text": "Success. Updated the following files:\nM /w/src/a.py\n"}]
    out = scan_codex_session(_exec_rollout(tmp_path, script, listed))
    assert out["writes"] == [
        {"ts": "t1", "path": "/w/docs/PRD.md", "tier": "heuristic", "how": "apply_patch add, script completed", "unconfirmed": True},
        {"ts": "t1", "path": "/w/src/a.py", "tier": "verified", "how": "apply_patch update, result lists path"},
    ]
    assert out["meta"]["patch_writes_unconfirmed"] == 1


def test_completed_script_with_control_flow_leaves_wrapped_patches_unconfirmed(tmp_path):
    script = (
        'const patch = "*** Begin Patch\\n*** Add File: docs/PRD.md\\n+x\\n*** End Patch";\n'
        "for (const f of files) { await tools.apply_patch(patch); }\n"
    )
    out = scan_codex_session(_exec_rollout(tmp_path, script, COMPLETED))
    assert out["writes"] == [{"ts": "t1", "path": "/w/docs/PRD.md", "tier": "heuristic", "how": "apply_patch add, script completed", "unconfirmed": True}]
    assert out["meta"]["patch_writes_unconfirmed"] == 1
    # A `Script running` yield settles nothing either.
    running = [{"type": "input_text", "text": "Script running with cell ID 3\nWall time 10.0 seconds\nOutput:\n"}]
    out = scan_codex_session(_exec_rollout(tmp_path, 'const p = "*** Begin Patch\\n*** Add File: y.md\\n*** End Patch"; await tools.apply_patch(p);', running))
    assert out["writes"] == [{"ts": "t1", "path": "/w/y.md", "tier": "heuristic", "how": "apply_patch add, script running", "unconfirmed": True}]


def test_exit_code_embedded_in_list_form_result_fails_the_shell_call(tmp_path, fake_a1):
    # The `exec` tool answers with a list of text parts; the wrapped command's exit code is only in the
    # text the script printed. Three forms seen in real rollouts.
    done = "Script completed\nWall time 0.1 seconds\nOutput:\n"
    forms = {
        "result object": '{"chunk_id":"06cc8d","wall_time_seconds":0.002,"exit_code":2,"original_token_count":9}\ncat: notes.txt: No such file\n',
        "exit_code line": "cat: notes.txt: No such file\nexit_code=2\n",
        "nested result": "Chunk ID: 1a2b\nWall time: 0.0 seconds\nProcess exited with code 2\nOutput:\ncat: notes.txt: No such file\n",
    }
    for form, printed in forms.items():
        script = 'const r = await tools.exec_command({cmd:"cat notes.txt", workdir:"/w"}); text(r.output);'
        out = scan_codex_session(_exec_rollout(tmp_path, script, [{"type": "input_text", "text": done}, {"type": "input_text", "text": printed}]))
        assert out["reads"] == [{"ts": "t1", "path": "/w/notes.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/w", "failed": True}], form
        assert out["meta"]["shell_calls_failed"] == 1, form
    # Exit code 0 in the same place: not failed, not counted.
    ok = '{"chunk_id":"06cc8d","wall_time_seconds":0.002,"exit_code":0,"original_token_count":9}\nhello\n'
    out = scan_codex_session(_exec_rollout(tmp_path, script, [{"type": "input_text", "text": done}, {"type": "input_text", "text": ok}]))
    assert out["reads"] == [{"ts": "t1", "path": "/w/notes.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/w"}]
    assert out["meta"]["shell_calls_failed"] == 0
    # A `Script failed` wrapper fails the script's commands even without an exit code.
    out = scan_codex_session(_exec_rollout(tmp_path, script, [{"type": "input_text", "text": "Script failed\nWall time 0.0 seconds\nOutput:\n"}, {"type": "input_text", "text": "Script error:\nboom"}]))
    assert out["reads"][0]["failed"] is True and out["meta"]["shell_calls_failed"] == 1
    # Two wrapped commands, one non-zero code: the output does not say which one failed.
    two = 'const a = await tools.exec_command({cmd:"cat a.txt", workdir:"/w"}); const b = await tools.exec_command({cmd:"cat b.txt", workdir:"/w"}); text(JSON.stringify(a)); text(JSON.stringify(b));'
    printed = '{"chunk_id":"1","exit_code":0,"output":"x"}\n{"chunk_id":"2","exit_code":1,"output":""}\n'
    out = scan_codex_session(_exec_rollout(tmp_path, two, [{"type": "input_text", "text": done}, {"type": "input_text", "text": printed}]))
    assert [(r["path"], r["failed"], r["failed_scope"]) for r in out["reads"]] == [("/w/a.txt", True, "script"), ("/w/b.txt", True, "script")]
    assert out["meta"]["shell_calls_failed"] == 1


def test_embedded_exit_code_does_not_fail_a_wrapped_patch_in_a_completed_script(tmp_path, fake_a1):
    # A script that applies a patch and then runs a failing test: the exit code belongs to the
    # test run, the patch result (listed under the banner) still verifies the write.
    script = (
        'const p = "*** Begin Patch\\n*** Add File: fix.py\\n+1\\n*** End Patch";\n'
        "text(await tools.apply_patch(p));\n"
        'const r = await tools.exec_command({cmd:"cat fix.py", workdir:"/w"}); text(JSON.stringify(r));\n'
    )
    parts = [
        {"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"},
        {"type": "input_text", "text": "Success. Updated the following files:\nA fix.py\n"},
        {"type": "input_text", "text": '{"chunk_id":"9","exit_code":1,"output":"F"}\n'},
    ]
    out = scan_codex_session(_exec_rollout(tmp_path, script, parts))
    assert out["writes"] == [{"ts": "t1", "path": "/w/fix.py", "tier": "verified", "how": "apply_patch add, result lists path"}]
    assert out["reads"] == [{"ts": "t1", "path": "/w/fix.py", "tier": "heuristic", "how": "cat", "cwd_seen": "/w", "failed": True}]
    assert out["meta"]["patch_calls_failed"] == 0 and out["meta"]["shell_calls_failed"] == 1


NO_META_ORIGIN = {"originator": None, "source": None, "cwd": None, "cli_version": None, "missing": ["originator", "source", "cwd", "cli_version"], "tier": "reported", "session_meta": False}


def test_rollout_without_session_meta_returns_null_origin_and_counts_it(tmp_path):
    p = _rollout(
        tmp_path,
        '{"timestamp":"t1","type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch","call_id":"c","input":"*** Begin Patch\\n*** Add File: y.md\\n*** End Patch"}}',
        '{"timestamp":"t2","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c","output":"Success. Updated the following files:\\nA y.md\\n"}}',
    )
    out = scan_codex_session(p)
    assert out["origin"] == NO_META_ORIGIN
    assert out["meta"]["origin_missing_session_meta"] == 1
    assert out["meta"]["records"] == 2
    # The write is still there, relative for want of a cwd.
    assert out["writes"] == [{"ts": "t1", "path": "y.md", "tier": "verified", "how": "apply_patch add", "relative": True}]
    # With a turn_context the cwd is filled from it and marked; the other fields stay missing.
    p2 = _rollout(tmp_path, '{"timestamp":"t0","type":"turn_context","payload":{"cwd":"/from/turn"}}')
    out = scan_codex_session(p2)
    assert out["origin"] == {**NO_META_ORIGIN, "cwd": "/from/turn", "cwd_from": "turn_context", "missing": ["originator", "source", "cli_version"]}
    assert out["meta"]["origin_missing_session_meta"] == 1


def test_wrapped_patch_in_a_failed_script_is_counted_unless_its_path_was_confirmed(tmp_path):
    script = (
        'const a = "*** Begin Patch\\n*** Add File: first.py\\n+1\\n*** End Patch";\n'
        'const b = "*** Begin Patch\\n*** Update File: second.py\\n@@\\n-x\\n+y\\n*** End Patch";\n'
        "text(await tools.apply_patch(a)); text(await tools.apply_patch(b));"
    )
    p = _rollout(
        tmp_path,
        '{"timestamp":"t0","type":"session_meta","payload":{"cwd":"/w","originator":"codex","source":"cli","cli_version":"0.1"}}',
        '{"timestamp":"t1","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c1","name":"exec","input":' + __import__("json").dumps(script) + "}}",
        '{"timestamp":"t2","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c1","output":[{"type":"input_text","text":"Script failed\\nWall time 0.1 seconds\\nOutput:\\n"},{"type":"input_text","text":"Success. Updated the following files:\\nA first.py\\n"},{"type":"input_text","text":"Script error:\\napply_patch verification failed: Failed to find expected lines in /w/second.py"}]}}',
    )
    out = scan_codex_session(p)
    assert out["writes"] == [{"ts": "t1", "path": "/w/first.py", "tier": "verified", "how": "apply_patch add, result lists path"}]
    assert out["meta"]["patch_calls"] == 2
    assert out["meta"]["patch_calls_failed"] == 1
    assert out["meta"]["patch_writes_unconfirmed"] == 0


def test_direct_apply_patch_with_dict_result_and_nonzero_exit_is_not_a_write(tmp_path):
    p = _rollout(
        tmp_path,
        '{"timestamp":"t0","type":"session_meta","payload":{"cwd":"/w","originator":"codex","source":"cli","cli_version":"0.1"}}',
        '{"timestamp":"t1","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c1","name":"apply_patch","input":"*** Begin Patch\\n*** Add File: x.md\\n+hi\\n*** End Patch"}}',
        '{"timestamp":"t2","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c1","output":{"output":"patch rejected","metadata":{"exit_code":1}}}}',
        '{"timestamp":"t3","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c2","name":"apply_patch","input":"*** Begin Patch\\n*** Add File: y.md\\n+hi\\n*** End Patch"}}',
        '{"timestamp":"t4","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c2","output":{"output":"Success. Updated the following files:\\nA y.md\\n","metadata":{"exit_code":0}}}}',
    )
    out = scan_codex_session(p)
    assert out["writes"] == [{"ts": "t3", "path": "/w/y.md", "tier": "verified", "how": "apply_patch add"}]
    assert out["meta"]["patch_calls_failed"] == 1
    assert out["meta"]["samples"] == ["patch rejected"]


def test_patch_paths_reads_headers_only():
    paths, deletes = patch_paths("*** Begin Patch\n*** Update File: a.py\n*** Move to: b.py\n@@\n-*** Add File: not_a_header.py\n*** Delete File: c.py\n*** End Patch")
    assert paths == [("a.py", "apply_patch update"), ("b.py", "apply_patch move")]
    assert deletes == 1


def test_apply_patch_inside_shell_command_is_a_patch_and_the_rest_is_still_scanned(tmp_path, fake_a1):
    p = _rollout(
        tmp_path,
        '{"timestamp":"2026-08-31T10:00:00Z","type":"session_meta","payload":{"cwd":"/w","originator":"codex-tui","cli_version":"0.1"}}',
        '{"timestamp":"2026-08-31T10:00:01Z","type":"response_item","payload":{"type":"function_call","name":"shell","arguments":"{\\"command\\":[\\"apply_patch\\",\\"*** Begin Patch\\\\n*** Add File: x.md\\\\n+hi\\\\n*** End Patch\\"]}","call_id":"c"}}',
        '{"timestamp":"2026-08-31T10:00:02Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c","output":"Success. Updated the following files:\\nA x.md\\n"}}',
        '{"timestamp":"2026-08-31T10:00:03Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\\"command\\":\\"cat before.txt\\\\napply_patch <<\'EOF\'\\\\n*** Begin Patch\\\\n*** Add File: cat > trap.md\\\\n+cat > trap2.md\\\\n*** End Patch\\\\nEOF\\\\ncat > note.md <<\'X\'\\\\nhi\\\\nX\\"}","call_id":"d"}}',
    )
    out = scan_codex_session(p)
    assert out["writes"] == [
        {"ts": "2026-08-31T10:00:01Z", "path": "/w/x.md", "tier": "verified", "how": "apply_patch add"},
        # No result record for `d`: the patch write is an unconfirmed attempt.
        {"ts": "2026-08-31T10:00:03Z", "path": "/w/cat > trap.md", "tier": "heuristic", "how": "apply_patch add, no result", "unconfirmed": True},
        # The shell write after the envelope is still seen; nothing inside the envelope is.
        {"ts": "2026-08-31T10:00:03Z", "path": "/w/note.md", "tier": "heuristic", "how": "redirect", "cwd_seen": "/w"},
    ]
    assert out["reads"] == [{"ts": "2026-08-31T10:00:03Z", "path": "/w/before.txt", "tier": "heuristic", "how": "cat", "cwd_seen": "/w"}]
    assert out["origin"]["missing"] == ["source"]
    assert out["origin"]["source"] is None


def test_cwd_falls_back_to_turn_context(tmp_path):
    p = _rollout(
        tmp_path,
        '{"timestamp":"2026-08-31T10:00:00Z","type":"session_meta","payload":{"originator":"codex_exec","source":"exec","cli_version":"0.1"}}',
        '{"timestamp":"2026-08-31T10:00:01Z","type":"turn_context","payload":{"cwd":"/from/turn"}}',
        '{"timestamp":"2026-08-31T10:00:02Z","type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch","call_id":"c","input":"*** Begin Patch\\n*** Add File: y.md\\n*** End Patch"}}',
        '{"timestamp":"2026-08-31T10:00:03Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c","output":"Success. Updated the following files:\\nA y.md\\n"}}',
    )
    out = scan_codex_session(p)
    assert out["origin"]["cwd"] == "/from/turn"
    assert out["origin"]["cwd_from"] == "turn_context"
    assert out["origin"]["missing"] == []
    assert out["writes"] == [{"ts": "2026-08-31T10:00:02Z", "path": "/from/turn/y.md", "tier": "verified", "how": "apply_patch add"}]


def test_relative_path_without_cwd_is_flagged(tmp_path):
    p = _rollout(tmp_path, '{"timestamp":"2026-08-31T10:00:02Z","type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch","input":"*** Begin Patch\\n*** Add File: y.md\\n*** End Patch"}}')
    out = scan_codex_session(p)
    assert out["origin"] == NO_META_ORIGIN
    assert out["meta"]["origin_missing_session_meta"] == 1
    assert out["writes"] == [{"ts": "2026-08-31T10:00:02Z", "path": "y.md", "tier": "heuristic", "how": "apply_patch add, no result", "relative": True, "unconfirmed": True}]
    assert out["meta"]["patch_calls_unmatched"] == 1


def test_codex_output_paths_reads_only_the_codex_exec_invocation():
    swarm = 'cd /Users/alice/Workspace/dagr-swarm-b && codex exec -m gpt-5.6-sol -c model_reasoning_effort="max" -s read-only --color never -o /Users/alice/reviews/swarm-b-sol.md - < /Users/alice/REVIEW-PROMPT.md > /Users/alice/log.txt 2>&1'
    assert codex_output_paths(swarm) == [["/Users/alice/reviews/swarm-b-sol.md"]]
    # Unrelated `-o` options in other segments are not codex writes.
    assert codex_output_paths("gcc -o a.out main.c && codex exec -o out.md 'p'; curl -o dl.bin http://x") == [["out.md"]]
    assert codex_output_paths("codex exec 'review this' | tee -o log.txt") == [[]]
    assert codex_output_paths("sort -o sorted.txt in.txt") == []
    # Forms: `-oFILE`, `--output-last-message FILE`, `--output-last-message=FILE`, quoted, `codex e`, a path to the binary.
    assert codex_output_paths("/opt/bin/codex exec -oout.md p") == [["out.md"]]
    assert codex_output_paths('codex e --output-last-message "my out.md" p') == [["my out.md"]]
    assert codex_output_paths("codex exec --output-last-message=/abs/o.md p") == [["/abs/o.md"]]
    # A quoted string holding `&&` is one argument, not a separator; a newline is.
    assert codex_output_paths('codex exec "a && b -o not.md" -o yes.md\nsort -o no.md x') == [["yes.md"]]
    assert codex_output_paths("codex exec -o a.md p1 && codex exec -o b.md p2") == [["a.md"], ["b.md"]]


def test_launch_command_output_file_is_a_verified_write():
    out = scan_codex_session(FIXTURE, launch_command="codex exec --model gpt-5 -o reviews/A.md 'review this' && echo done")
    o = [w for w in out["writes"] if w["how"] == "codex -o"]
    # Timestamped at the rollout's last record: the CLI writes the file when the session ends.
    assert o == [{"ts": T(46), "path": "/work/ws/reviews/A.md", "tier": "verified", "how": "codex -o"}]
    assert output_writes_from_launch('codex exec --output-last-message="/abs/out.md" p', "t") == [{"ts": "t", "path": "/abs/out.md", "tier": "verified", "how": "codex -o"}]
    assert output_writes_from_launch("codex exec -o $OUT p", "t", "/w") == [{"ts": "t", "path": "/w/$OUT", "tier": "verified", "how": "codex -o", "unresolved": True}]
    assert output_writes_from_launch("claude -p x -o f", "t") == []
    assert output_writes_from_launch("echo codex && gcc -o a.out x.c", "t") == []
    # Two launches in one command: the command alone does not say which session wrote which file.
    assert output_writes_from_launch("codex exec -o /a.md p && codex exec -o /b.md q", "t") == [
        {"ts": "t", "path": "/a.md", "tier": "heuristic", "how": "codex -o", "ambiguous": True},
        {"ts": "t", "path": "/b.md", "tier": "heuristic", "how": "codex -o", "ambiguous": True},
    ]


def test_path_only_scan_attributes_only_output_path_evidenced_in_rollout(fake_a1):
    out = scan_codex_session(FIXTURE)
    output = [w for w in out["writes"] if w["path"] == "/work/ws/reviews/codex-output.md"]
    assert output == [{"ts": T(44), "path": "/work/ws/reviews/codex-output.md", "tier": "heuristic", "how": "redirect", "cwd_seen": "/work/ws"}]
    # No launch command was supplied, so the scan does not invent a verified codex -o event.
    assert not any(w["how"] == "codex -o" for w in out["writes"])


def test_bashwrite_exclusions_are_flattened_into_integer_meta(fake_a1):
    out = scan_codex_session(FIXTURE)
    assert out["meta"]["bashwrites_excluded_invalid_cp_mv"] == 1
    assert isinstance(out["meta"]["bashwrites_excluded_invalid_cp_mv"], int)


def test_scan_is_callable_with_the_path_alone_and_stable():
    a = scan_codex_session(FIXTURE)
    b = scan_codex_session(FIXTURE)
    assert a == b
    assert set(a) == {"writes", "reads", "artifact_facts", "origin", "meta"}


def test_missing_file_yields_empties_with_zero_records(tmp_path):
    out = scan_codex_session(tmp_path / "nope.jsonl")
    assert out["writes"] == [] and out["reads"] == []
    # No file, no session_meta: every origin field present and null, and the absence counted.
    assert out["origin"] == NO_META_ORIGIN
    assert out["meta"]["records"] == 0
    assert out["meta"]["origin_missing_session_meta"] == 1


# ---------------------------------------------------------------- code map, regex against the loop


def _reference_code_map(script: str) -> bytearray:
    """The character loop `_js_code_map` replaced (0.1.0), kept as its specification."""
    code = bytearray(b"\x01") * len(script)
    i = 0
    state = "code"
    quote = ""
    while i < len(script):
        c = script[i]
        nxt = script[i + 1] if i + 1 < len(script) else ""
        if state == "code":
            if c in "\"'`":
                state, quote = "string", c
                code[i] = 0
            elif c == "/" and nxt == "/":
                state = "line_comment"
                code[i:i + 2] = b"\x00\x00"
                i += 1
            elif c == "/" and nxt == "*":
                state = "block_comment"
                code[i:i + 2] = b"\x00\x00"
                i += 1
        elif state == "string":
            code[i] = 0
            if c == "\\" and i + 1 < len(script):
                code[i + 1] = 0
                i += 1
            elif c == quote:
                state = "code"
        elif state == "line_comment":
            code[i] = 0
            if c in "\r\n":
                state = "code"
        else:
            code[i] = 0
            if c == "*" and nxt == "/":
                code[i + 1] = 0
                i += 1
                state = "code"
        i += 1
    return code


@pytest.mark.parametrize("script", [
    "", "a", "'", "\\", "'\\", "'a\\'b'c", "`x${y}`z", "a // c\nb", "a // c\rb", "a /* c */ b", "/*/ x", "/**/x",
    "a /* never closed", "'never closed", "\"a\\\"b\" c", "x = '//not a comment'; tools.exec_command({cmd: 'ls'})",
    "// tools.apply_patch(p)\ntools.apply_patch(p)", "'\\\n' tools", "`multi\nline` code",
])
def test_code_map_matches_the_character_loop(script):
    codexio_parse._code_map_memo.clear()
    assert codexio_parse._js_code_map(script) == _reference_code_map(script)


def test_code_map_matches_the_character_loop_on_random_scripts():
    import random

    rng = random.Random(7)
    alphabet = ['"', "'", "`", "/", "*", "\\", "\n", "\r", "a", " "]
    for _ in range(20000):
        script = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20)))
        codexio_parse._code_map_memo.clear()
        assert codexio_parse._js_code_map(script) == _reference_code_map(script), script
