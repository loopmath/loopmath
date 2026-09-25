"""The suite stays out of the user's store.

tests/conftest.py points LOOPMATH_HOME and LOOPMATH_CACHE_DIR at a session temp
folder, lists ~/.loopmath (paths, mtimes, sizes) before any test runs, and moves
this file to the end of the run. The last test lists it again and fails on any
path that appeared, changed or went away. Nothing here writes or deletes under
~/.loopmath.
"""

import os

from loopmath import output
from loopmath.ingest import base


def test_the_session_store_and_cache_are_a_temp_folder(session_store_folder):
    home = output.home()
    cache = base.cache_dir()
    assert home == session_store_folder / "home"
    assert cache == session_store_folder / "cache"
    user_store = os.path.expanduser("~/.loopmath")
    for path in (home, cache):
        assert not str(path).startswith(user_store + os.sep), path


def test_the_listing_sees_a_new_a_changed_and_a_removed_path(tmp_path, store_listing, user_store_changes):
    root = tmp_path / "store"
    (root / "runs").mkdir(parents=True)
    (root / "runs" / "kept.json").write_text("{}")
    (root / "runs" / "edited.json").write_text("{}")
    (root / "runs" / "gone.json").write_text("{}")
    before = store_listing(root)
    assert user_store_changes(root, before) == ([], [], [])
    (root / "runs" / "new.json").write_text("{}")
    (root / "runs" / "edited.json").write_text('{"a": 1}')
    (root / "runs" / "gone.json").unlink()
    added, changed, removed = user_store_changes(root, before)
    assert added == [str(root / "runs" / "new.json")]
    assert str(root / "runs" / "edited.json") in changed
    assert str(root / "runs" / "kept.json") not in changed
    assert removed == [str(root / "runs" / "gone.json")]


def test_the_suite_left_the_users_store_alone(user_store_changes):
    added, changed, removed = user_store_changes()
    lines = []
    for label, paths in (("added", added), ("changed", changed), ("removed", removed)):
        if paths:
            shown = ", ".join(paths[:10]) + (f", and {len(paths) - 10} more" if len(paths) > 10 else "")
            lines.append(f"{len(paths)} {label}: {shown}")
    assert not lines, "the suite touched ~/.loopmath: " + "; ".join(lines)
