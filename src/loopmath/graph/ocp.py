"""Compatibility exports for graph-to-OCP conversion.

The mapping helpers and emitter are split into sibling modules; public names
remain available from ``loopmath.graph.ocp``.
"""

from .ocp_support import *  # noqa: F401,F403
from .ocp_support import _EM_DASH_RE, _fmt_ts, _parse_ts
from .ocp_emit import to_ocp
