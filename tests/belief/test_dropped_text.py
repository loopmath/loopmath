"""The plain words `fit` prints beside a dropped reason (new-user test: text only)."""

from __future__ import annotations

from loopmath.belief import fit as F


def test_a_shipped_copy_is_said_plainly():
    text = F.dropped_text({F.SHIPPED_COPY: 3})
    assert text == ("dropped 3: shipped copy of a stored run "
                    "(3: this run ships with loopmath and is also in your store, so it counts once, as yours)")
    assert "run id" not in text and "your copy is used" not in text
