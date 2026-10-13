"""The namespace an adapter writes its ext under, and documents written before 0.2.7 (shared by the adapter tests).

From 0.2.7 every adapter writes `dev.loopmath.adapter.<name>`. Adapters emit OCP v0.2, where the pre-rename
`dev.dagr.` namespace draws no warning, but a document migrated to v0.3 drew W182 for each such key. A document
an adapter wrote before 0.2.7 still migrates and validates, with W182 and no error, and migrates to the same
document apart from the key.
"""

from __future__ import annotations

import copy
import json

from loopmath.ocp.conformance import validate_doc
from loopmath.ocp.migrate import migrate_doc


def check_namespace(doc: dict, name: str) -> None:
    key, old_key = f"dev.loopmath.adapter.{name}", f"dev.dagr.adapter.{name}"
    text = json.dumps(doc)
    assert "dev.dagr." not in text and f'"{key}"' in text
    new = migrate_doc(copy.deepcopy(doc))
    assert not [f for f in validate_doc(new) if f.code == "W182" or f.level == "error"]
    old = migrate_doc(json.loads(text.replace(f'"{key}"', f'"{old_key}"')))
    found = validate_doc(old)
    assert not [f for f in found if f.level == "error"]
    assert any(f.code == "W182" for f in found)
    assert json.loads(json.dumps(old).replace(f'"{old_key}"', f'"{key}"')) == new
