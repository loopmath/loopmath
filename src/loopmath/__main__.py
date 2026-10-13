"""`python -m loopmath` runs the CLI, as the `loopmath` command does."""

import sys

from .cli import main

sys.exit(main())
