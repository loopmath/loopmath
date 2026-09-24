"""One token-completeness definition shared by graph writers and readers."""

from __future__ import annotations


# OCP field first, Graph token key second.  Completeness means all six mapped
# streams are present with a value other than None; zero is a measured value.
TOKEN_STREAMS = (
    ("input_tokens", "in"),
    ("cached_input_tokens", "cache_read"),
    ("cache_creation_tokens", "cache_write"),
    ("cache_creation_5m_tokens", "cache_write_5m"),
    ("cache_creation_1h_tokens", "cache_write_1h"),
    ("output_tokens", "out"),
)


def missing_token_streams(tokens: object) -> tuple[str, ...]:
    """Return OCP names for every absent or None token stream."""
    if not isinstance(tokens, dict):
        return tuple(ocp_name for ocp_name, _graph_name in TOKEN_STREAMS)
    return tuple(
        ocp_name
        for ocp_name, graph_name in TOKEN_STREAMS
        if tokens.get(graph_name) is None
    )
