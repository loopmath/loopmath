"""Prompt views and deterministic prompt construction for the labeler."""

from __future__ import annotations

# Compatibility facade: keep every name the former monolith bound at import time.
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .labeler_common import (
    COMMIT_LIMIT,
    PARENT_COMMAND_LIMIT,
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    LabelerError,
)
from .labeler_prompt_build import (
    PARENT_EDGE_KINDS,
    _PROMPT_V1_HEAD,
    _PROMPT_V2_HEAD,
    _PROMPT_V3_HEAD,
    _PROMPT_V3_STRICT_HEAD,
    _PROMPT_V4_HEAD,
    _ROLE_GUARD,
    _candidate_parent_result,
    _candidate_summary,
    _build_prompt_snapshot,
    _prompt_metadata,
    build_prompt,
    candidate_parent_metadata,
    candidate_parents,
)
from .labeler_prompt_evidence import (
    _MISSING,
    _TIMESTAMP_RE,
    _attempt_evidence_result,
    _attempt_ledger_result,
    _cause_fields,
    _feat,
    _field,
    _normalization,
    _valid_timestamp,
    attempt_evidence,
    node_view,
)
from .labeler_prompt_session import (
    _COMMAND_WRAPPER_RE,
    _DECISION_LIST_HALF,
    _DECISION_LIST_LIMIT,
    _DECISION_TEXT_RE,
    _DISPOSITION_TOOL_RE,
    _EVIDENCE_TEXT_HALF,
    _EVIDENCE_TEXT_LIMIT,
    _SESSION_LIST_HALF,
    _SESSION_LIST_LIMIT,
    _SYSTEM_USER_TEXT_RE,
    _bounded_rows,
    _bounded_text,
    _command_from_arguments,
    _content_text,
    _human_span,
    _read_session_records,
    _reason_maps,
    _record_accounting,
    _session_evidence_result,
    _session_evidence_summary,
    _session_record_fallback_reason,
    _text_span,
    _tool_calls,
    _user_text_exclusion,
    session_evidence,
)
from .labeler_prompt_workflow import (
    _dispatch_text,
    _task_dispatches,
    _verified_children,
    _workflow_evidence_result,
    _workflow_record_fallback_reason,
    workflow_evidence,
)
