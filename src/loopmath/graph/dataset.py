"""Compatibility exports for the graph evaluation dataset builder.

Implementation is organized by label loading, node features, edge labeling, and CLI/reporting.
"""
import sys

from .dataset_core import *
from .dataset_core import _pick
from .dataset_nodes import *
from .dataset_edges import *
from .dataset_build import *


if __name__ == "__main__":
    sys.exit(main())
