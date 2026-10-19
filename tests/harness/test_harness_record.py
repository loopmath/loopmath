"""The harness record (`loopmath.harness`, `spec/harness.md`): digest, factors, layouts.

Fixture trees are written here from dicts. The harness2 trees follow its code, read and not run
(google-research/harness2 at 1cfe254): the eight components and their files are cited per file below.
"""

from __future__ import annotations

import json
import os
import unicodedata

import pytest

from loopmath import harness as H

SHA = "sha256:" + "0" * 64


def write_tree(root, files: dict[str, str | bytes]):
    root.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            p.write_bytes(body)
        else:
            p.write_text(body, encoding="utf-8", newline="")
    return root


def sha8(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------- harness2 layouts

# The OpenCode surface, harness2/substrate/opencode.py:44-63: always-on systemprompt.md, guardrails.md,
# LongTermMEMORY.md and tool_descriptions/tool_guidance.md (:44-49), skills under skills/ (:52), plugins under
# plugin/ (:62). The paths on a surface, harness2/core/tree.py:66-74 and 92-110: skills/<n>/SKILL.md and
# skills/<n>/scripts/<file>, sub_agents/<n>/AGENT.md, scripts/<n>.py, plugin/<n>.ts, names [a-z0-9][a-z0-9_-]*
# (tree.py:37). Component ids per file: harness2/core/prompts.py:424-431. Not harness parts: the evidence
# tree under context/ (harness2/core/context.py:17-23) and the README stubs (tree.py:141).
H2_OPENCODE = {
    "systemprompt.md": "You are a careful analyst.\n",
    "guardrails.md": "Never read the rubric.\n",
    "LongTermMEMORY.md": "Lesson: check units.\n",
    "tool_descriptions/tool_guidance.md": "Prefer grep before read.\n",
    "sub_agents/checker/AGENT.md": "---\nmode: subagent\n---\nCheck the table.\n",
    "skills/xlsx-audit/SKILL.md": "---\nname: xlsx-audit\ndescription: audit sheets\n---\nTrigger: a sheet.\n",
    "skills/xlsx-audit/scripts/scan.py": "print('scan')\n",
    "scripts/fetch_rates.py": "print('rates')\n",
    "plugin/no-rubric.ts": "export default {}\n",
    "context/task.md": "The task text.\n",
    "scripts/README.md": "Stub written by write_tree.\n",
}


def test_harness2_opencode_tree_maps_the_eight_components(tmp_path):
    root = write_tree(tmp_path / "tree", H2_OPENCODE)
    rec = H.record_from_tree(root, source="harness2")
    assert rec["schema"] == "loopmath.harness/0.1"
    assert rec["source"] == "harness2"
    assert rec["digest"].startswith("sha256:") and len(rec["digest"]) == 71
    assert rec["parent"] is None
    assert H.factors(rec) == tuple(sorted([
        f"prompt:{sha8(H2_OPENCODE['systemprompt.md'])}",
        f"guardrails:{sha8(H2_OPENCODE['guardrails.md'])}",
        "memory:LongTermMEMORY",
        f"tool-guidance:{sha8(H2_OPENCODE['tool_descriptions/tool_guidance.md'])}",
        "subagent:checker",
        "skill:xlsx-audit",
        "script:fetch_rates",
        "plugin:no-rubric",
    ]))
    assert "harness2" in rec["layouts"]


def test_harness2_evidence_files_are_not_in_the_digest(tmp_path):
    a = write_tree(tmp_path / "a", H2_OPENCODE)
    b = write_tree(tmp_path / "b", {**H2_OPENCODE, "context/task.md": "Another task.\n",
                                    "context/probe_1/trajectory.txt": "[ACTION] ls\n"})
    assert H.digest(a) == H.digest(b)


# The Codex surface, harness2/substrate/codex.py:41-74: AGENTS.md is the only always-on file (:41), skills under
# .agents/skills (:47), hooks in .codex/hooks.json (:58) with the events at :59-72, command hooks only
# (harness2/core/tree.py:245). The schema of a hooks entry: tree.py:216-251.
H2_CODEX = {
    "AGENTS.md": "Work in small steps.\n",
    ".agents/skills/csv-join/SKILL.md": "---\nname: csv-join\n---\nTrigger: two csv files.\n",
    ".codex/hooks.json": json.dumps({"hooks": {
        "PreToolUse": [{"matcher": "shell", "hooks": [{"type": "command", "command": "python3 guard.py"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "python3 lessons.py"}]}],
    }}),
}


def test_harness2_codex_tree(tmp_path):
    rec = H.record_from_tree(write_tree(tmp_path / "t", H2_CODEX), source="harness2")
    assert H.factors(rec) == tuple(sorted([
        f"prompt:{sha8(H2_CODEX['AGENTS.md'])}", "skill:csv-join", "hook:PreToolUse:shell", "hook:Stop",
    ]))
    assert "codex" in rec["layouts"]


# ---------------------------------------------------------------- Claude Code and Codex layouts

CLAUDE_REPO = {
    "CLAUDE.md": "Run the tests before you finish.\n",
    ".claude/skills/release-notes/SKILL.md": "---\nname: release-notes\n---\nWrite notes.\n",
    ".claude/skills/release-notes/template.md": "## Notes\n",
    ".claude/agents/code-reviewer.md": "---\nname: code-reviewer\n---\nReview diffs.\n",
    ".claude/commands/ship.md": "Ship it.\n",
    ".claude/settings.json": json.dumps({
        "permissions": {"allow": ["Bash(pytest:*)"]},
        "hooks": {"PostToolUse": [{"matcher": "Edit|Write",
                                   "hooks": [{"type": "command", "command": "ruff format"}]}]},
        "enabledPlugins": {"formatter@acme-tools": True, "old@acme-tools": False},
    }),
    ".mcp.json": json.dumps({"mcpServers": {"github": {"command": "gh-mcp"}, "postgres": {"command": "pg"}}}),
    "src/app.py": "print('app')\n",
    "README.md": "# app\n",
    "scripts/deploy.sh": "echo deploy\n",
    "skills/not-a-harness/SKILL.md": "a repo folder, not a harness part outside the harness2 layout\n",
}


def test_claude_code_layout(tmp_path):
    rec = H.record_from_tree(write_tree(tmp_path / "repo", CLAUDE_REPO), source="hand")
    assert H.factors(rec) == tuple(sorted([
        f"prompt:{sha8(CLAUDE_REPO['CLAUDE.md'])}",
        "skill:release-notes",
        "subagent:code-reviewer",
        "command:ship",
        "hook:PostToolUse:Edit|Write",
        "plugin:formatter@acme-tools",
        "mcp:github",
        "mcp:postgres",
    ]))
    assert "claude-code" in rec["layouts"] and "harness2" not in rec["layouts"]


def test_repo_code_is_not_in_the_digest(tmp_path):
    a = write_tree(tmp_path / "a", CLAUDE_REPO)
    b = write_tree(tmp_path / "b", {**CLAUDE_REPO, "src/app.py": "print('changed')\n", "src/new.py": "x = 1\n",
                                    "scripts/deploy.sh": "echo changed\n"})
    assert H.digest(a) == H.digest(b)


CODEX_REPO = {
    "AGENTS.md": "Use uv.\n",
    "AGENTS.override.md": "Local override.\n",
    ".agents/skills/bisect/SKILL.md": "---\nname: bisect\n---\nBisect a regression.\n",
    ".codex/config.toml": '[mcp_servers.linear]\ncommand = "linear-mcp"\n\n[mcp_servers.docs]\nurl = "http://127.0.0.1:9"\n',
}


def test_codex_layout(tmp_path):
    rec = H.record_from_tree(write_tree(tmp_path / "repo", CODEX_REPO))
    assert rec["source"] == "hand"  # the default source
    assert H.factors(rec) == tuple(sorted([
        f"prompt:{sha8(CODEX_REPO['AGENTS.md'])}", f"prompt:{sha8(CODEX_REPO['AGENTS.override.md'])}",
        "skill:bisect", "mcp:linear", "mcp:docs",
    ]))
    assert "codex" in rec["layouts"]


def test_empty_tree_has_no_factors_and_a_fixed_digest(tmp_path):
    a = H.record_from_tree(write_tree(tmp_path / "a", {"src/x.py": "1\n"}))
    b = H.record_from_tree(write_tree(tmp_path / "b", {}))
    assert H.factors(a) == () and a["digest"] == b["digest"]
    assert a["layouts"] == []


# ---------------------------------------------------------------- digest normalization

def test_digest_ignores_line_endings_and_a_bom(tmp_path):
    a = write_tree(tmp_path / "a", {"CLAUDE.md": "one\ntwo\n"})
    b = write_tree(tmp_path / "b", {"CLAUDE.md": "one\r\ntwo\r\n"})
    c = write_tree(tmp_path / "c", {"CLAUDE.md": b"\xef\xbb\xbfone\rtwo\r"})
    assert H.digest(a) == H.digest(b) == H.digest(c)
    # and the prompt factor hashes the normalized text too
    assert H.factors(H.record_from_tree(a)) == H.factors(H.record_from_tree(b)) == H.factors(H.record_from_tree(c))


def test_digest_changes_with_content_but_the_factor_name_does_not(tmp_path):
    a = write_tree(tmp_path / "a", CLAUDE_REPO)
    b = write_tree(tmp_path / "b", {**CLAUDE_REPO, ".claude/skills/release-notes/template.md": "## Changed\n"})
    ra, rb = H.record_from_tree(a), H.record_from_tree(b)
    assert ra["digest"] != rb["digest"]
    assert H.factors(ra) == H.factors(rb)


def test_digest_does_not_depend_on_creation_order_or_unicode_form(tmp_path):
    name = "café"
    a = write_tree(tmp_path / "a", {f".claude/skills/{unicodedata.normalize('NFC', name)}/SKILL.md": "x\n",
                                    "CLAUDE.md": "p\n"})
    b = write_tree(tmp_path / "b", {"CLAUDE.md": "p\n",
                                    f".claude/skills/{unicodedata.normalize('NFD', name)}/SKILL.md": "x\n"})
    assert H.digest(a) == H.digest(b)
    assert f"skill:{unicodedata.normalize('NFC', name)}" in H.factors(H.record_from_tree(b))


def test_ignored_folders_and_os_litter(tmp_path):
    a = write_tree(tmp_path / "a", {".claude/skills/s/SKILL.md": "x\n"})
    b = write_tree(tmp_path / "b", {".claude/skills/s/SKILL.md": "x\n", ".claude/skills/s/.DS_Store": b"\0\1",
                                    ".claude/skills/s/__pycache__/m.cpython-313.pyc": b"\0",
                                    ".claude/skills/s/node_modules/dep/index.js": "x\n"})
    assert H.digest(a) == H.digest(b)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_symlinks_are_hashed_as_links_and_never_followed(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret one\n")
    a = write_tree(tmp_path / "a", {})
    (a / "CLAUDE.md").symlink_to(outside)
    d1 = H.digest(a)
    outside.write_text("secret two\n")
    assert H.digest(a) == d1  # the target's content is never read
    b = write_tree(tmp_path / "b", {})
    (b / "CLAUDE.md").symlink_to(tmp_path / "elsewhere.md")
    assert H.digest(b) != d1  # the link text is hashed


def test_digest_takes_a_layout(tmp_path):
    tree = write_tree(tmp_path / "t", {"scripts/a.py": "1\n", "systemprompt.md": "p\n"})
    assert H.digest(tree) == H.digest(tree, layout="auto")
    assert "script:a" in H.factors(H.record_from_tree(tree))  # harness2 markers found: scripts/ count
    assert "script:a" not in H.factors(H.record_from_tree(tree, layout="agents"))
    assert H.digest(tree, layout="agents") != H.digest(tree, layout="harness2")


# ---------------------------------------------------------------- factors() on any record

def test_factors_keep_unknown_kinds_and_any_source():
    rec = {"schema": "loopmath.harness/0.1", "digest": SHA, "parent": None, "source": "meta-harness",
           "factors": ["widget:x", "skill:b", "skill:a", "skill:a", "bare-name", "", 7, None, " mcp:pad "],
           "future_field": {"kept": True}}
    assert H.factors(rec) == ("mcp:pad", "other:bare-name", "skill:a", "skill:b", "widget:x")
    norm = H.normalize_record(rec)
    assert norm["source"] == "meta-harness" and norm["future_field"] == {"kept": True}
    assert norm["factors"] == list(H.factors(rec))


def test_factors_of_a_later_schema_version_are_still_read():
    rec = {"schema": "loopmath.harness/0.7", "digest": SHA, "factors": ["skill:a"], "new": 1}
    assert H.factors(rec) == ("skill:a",)


def test_factors_of_something_that_is_not_a_record():
    assert H.factors(None) == ()
    assert H.factors({"factors": "skill:a"}) == ()
    assert H.factors([]) == ()


def test_record_carries_parent_and_tool(tmp_path):
    root = write_tree(tmp_path / "t", H2_OPENCODE)
    tool = {"name": "harness2", "version": "0.1.0", "run_id": "r1", "task_id": "jb-007", "stage": "agg1"}
    rec = H.record_from_tree(root, source="harness2", parent=SHA, tool=tool)
    assert rec["parent"] == SHA and rec["tool"] == tool
    assert json.loads(json.dumps(rec)) == rec


def test_record_from_run_document():
    rec = {"schema": "loopmath.harness/0.1", "digest": SHA, "factors": ["skill:a"], "source": "hand"}
    doc = {"run": {"ext": {H.EXT_KEY: rec}}}
    assert H.EXT_KEY == "dev.loopmath.harness"
    assert H.run_record(doc)["factors"] == ["skill:a"]
    assert H.run_record({"run": {"ext": {"harness": rec}}})["factors"] == ["skill:a"]  # the bare key reads too
    assert H.run_record({"run": {"ext": {}}}) is None
    assert H.run_record({"run": {}}) is None


def test_a_harness2_codex_tree_needs_the_layout_named(tmp_path):
    # harness2's Codex install copies AGENTS.md, .agents, scripts and .codex (harness2/substrate/codex.py:106,
    # 132-139); AGENTS.md is no harness2 marker, so an importer names the layout
    tree = write_tree(tmp_path / "t", {**H2_CODEX, "scripts/tally.py": "print(1)\n",
                                       "sub_agents/checker/AGENT.md": "---\nmode: subagent\n---\nx\n"})
    assert "script:tally" not in H.factors(H.record_from_tree(tree))
    rec = H.record_from_tree(tree, source="harness2", layout="harness2")
    assert {"script:tally", "subagent:checker", "skill:csv-join", "hook:Stop"} <= set(H.factors(rec))
    assert rec["layouts"] == ["codex", "harness2"]


def test_only_the_named_claude_folders_are_read(tmp_path):
    a = write_tree(tmp_path / "a", {".claude/settings.json": "{}"})
    b = write_tree(tmp_path / "b", {".claude/settings.json": "{}", ".claude/worktrees/w1/CLAUDE.md": "other\n",
                                    ".claude/worktrees/w1/src/app.py": "1\n", ".claude/todos/t.json": "[]"})
    assert H.digest(a) == H.digest(b)
    assert H.factors(H.record_from_tree(b)) == ()


def test_a_layout_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(ValueError):
        H.digest(write_tree(tmp_path / "t", {}), layout="opencode")
    with pytest.raises(NotADirectoryError):
        H.digest(tmp_path / "missing")
