# Release policy

The distribution published to the Python Package Index is named `loopmath`.
Installing it provides the Python import package `loopmath` and the `loopmath` console
command, plus `loop`, a short alias for the same command. If another `loop` is
already on the PATH, use `loopmath`.

## Version meaning

loopmath versions follow PEP 440 and use Semantic Versioning intent. A final release
has the form `MAJOR.MINOR.PATCH`. Before 1.0, a minor release may contain an
incompatible change because the public Python and command-line interfaces are
still evolving. A patch release remains backward compatible and contains fixes
only. Starting with 1.0, MAJOR changes for an incompatible public-interface
change, MINOR changes for backward-compatible functionality, and PATCH changes
for backward-compatible fixes.

A `.devN` suffix identifies an unreleased development snapshot, and `aN`, `bN`,
and `rcN` identify alpha, beta, and release-candidate builds. For example,
`0.1.0.dev0` is development work leading toward `0.1.0`; it is not the final
`0.1.0` release. Local version identifiers are not published to the public
index. The version in `pyproject.toml` and `loopmath.__version__` must always match
so package metadata and `loopmath --version` report the same value.

## When the version moves

The version changes only on an intentional release transition, not on every
merged change. Release preparation replaces the development or prerelease
suffix with the chosen final version after the release scope is frozen. After
that version is published, development advances to the next planned release
with a `.dev0` suffix. Further development snapshots increment `N` only when a
snapshot is intentionally distributed. Because an index version cannot be
replaced, any correction after publication receives a new patch version rather
than rebuilding a different artifact under an existing version.

## Release requirements

A release comes from a clean, reviewed commit. The release owner updates
`pyproject.toml` and `loopmath.__version__` together, confirms that the version is
not a development or prerelease version for a final release, and verifies that
the root `README.md` and all required package data are present. An explicitly
approved SPDX license expression must be declared in `pyproject.toml`, and the
corresponding root license file must be included in both distribution
artifacts. The full test suite must pass in the release worktree.

`scripts/release-check.sh` builds the source distribution and then the wheel
from it with `python -m build`, requires `twine check --strict dist/*` to pass,
checks that the wheel holds every file under `src/loopmath`, and installs the
wheel, not the source tree or an editable install, into a genuinely fresh
virtual environment under `/tmp`. In that environment `loopmath --version`,
`loop --version`, `loopmath --help`, `loopmath task-types --json`,
`loopmath skill install` into a temporary home, `loopmath skill show`,
`loopmath doctor --json` and `loopmath analyze` on the committed synthetic
fixture must succeed, and so must the loop on a temporary store:
`loopmath workflows list --json`, `loopmath prior show --json`,
`loopmath ocp validate` on the OCP v0.3 examples in `spec/examples/v0.3/`,
`loopmath fit --json`, `loopmath recommend` with `--json` and with `--html`,
`loopmath runs --html`, `loopmath posterior --html` and `loopmath status --json`.
This check catches a missing console entry point or package data, including
`loopmath/prices.toml`, `loopmath/skill/SKILL.md`, the OCP schemas, the
workflow catalog, the prior bundle and the view pages.
The environment holds the wheel's declared core dependencies and nothing else:
no extra, and nothing borrowed from another environment, so these checks show
that a plain `pip install loopmath` runs the loop. The check also confirms that
the extras' packages (PyMC, PyTensor, ArviZ, Matplotlib) are absent and that
`loopmath research fit` names the missing `[bayes]` extra.
The check makes no network requests. uv resolves the dependencies offline from
its cache when it can. Otherwise the build interpreter's installed copies of the
declared core dependencies, and of their own dependencies, are repacked as
wheels in a temporary folder, and pip resolves the wheel against that folder
alone (`--no-index`). The versions installed are then those copies, not the
newest on the index.

Right after an upload, `scripts/release-check.sh --online testpypi` (then
`--online pypi`) installs the published `loopmath==VERSION` into a fresh
virtual environment with pip, the network and no cache, as a user would, and
runs the same checks on it, so the real published dependency closure is what
is tested. It builds nothing. `--version X` picks the version (default: this
tree's), and `--online URL` takes any simple index. For TestPyPI the
dependencies come from PyPI. The report lists every distribution pip installed
and the hosts it came from.

The artifacts must be built from the exact commit tagged `vMAJOR.MINOR.PATCH`,
and the tag, artifact filenames, and embedded metadata must agree. Any failed
check, dirty worktree, version mismatch, missing artifact, or unexpected file
in an artifact blocks publication. Only the release owner uploads the verified
artifacts with the owner's publishing credentials. Automation and contributors
must never upload to PyPI or TestPyPI on the owner's behalf.
