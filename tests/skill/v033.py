"""loopmath 0.3.3's installer as it shipped, for the upgrade tests.

fixtures/skill-0.3.3/ holds 0.3.3's `src/loopmath/skill/install.py` and the packaged texts it
installed (`skills/`), byte for byte from the v0.3.3 tag. `installer()` loads that module with its
texts read from the fixture, so a test makes a 0.3.3 install with 0.3.3's own code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skill-0.3.3"
NAME = "loopmath_skill_install_0_3_3"


def installer() -> ModuleType:
    if NAME in sys.modules:
        return sys.modules[NAME]
    spec = importlib.util.spec_from_file_location(NAME, FIXTURE / "install.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = mod  # its dataclass looks its module up while the class is built
    spec.loader.exec_module(mod)
    mod._package_file = lambda *parts: FIXTURE.joinpath("skills", *parts).read_text(encoding="utf-8")
    return mod
