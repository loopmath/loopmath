"""Compatibility facade for heuristic Bash file recognition.

The parser and the two cohesive recognizers live in sibling modules. These explicit
re-exports preserve the original public import path.
"""

from .bashread_ops import reads_from_command
from .bashwrite_ops import writes_from_command

__all__ = ("reads_from_command", "writes_from_command")
