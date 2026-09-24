#!/usr/bin/env bash
# Release check (docs/release.md): build the sdist and wheel, twine check them, install the
# wheel into a fresh venv under /tmp, and run the installed commands from outside the tree.
#
# No network. The venv gets the wheel's declared core dependencies and nothing else: uv
# resolves them offline from its cache when it can; otherwise the build interpreter's
# installed copies of those dependencies (and theirs) are repacked as wheels in a temp folder
# and pip resolves the wheel against that folder alone (--no-index). No extra is installed and
# nothing is borrowed, so the loop checks prove a plain `pip install loopmath` runs the loop.
#
# Online mode, after an upload: --online installs the published loopmath==VERSION into a fresh
# venv with pip and the network (no cache), so the real published closure is what the same
# checks run on. Nothing is built.
#
# Usage: scripts/release-check.sh [--python PY] [--keep] [--online pypi|testpypi|URL [--version X]]
#   PY builds and supplies the dependencies: $LOOPMATH_PY, then python3. It needs `build`,
#   `twine` and the core dependencies installed (online: only a Python with venv and pip).
#   --online pypi or testpypi (dependencies from PyPI), or a simple index URL.
#   --version defaults to the version in this tree's pyproject.toml.
#   --keep leaves the temp folder in place. Exit 0 only when every check passes.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
py="${LOOPMATH_PY:-python3}"
keep=0
online=""
version=""
while [ $# -gt 0 ]; do
  case "$1" in
    --python) py="$2"; shift 2 ;;
    --keep) keep=1; shift ;;
    --online) online="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "release-check.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ -n "$online" ]; then
  case "$online" in
    pypi) index=(--index-url https://pypi.org/simple/) ;;
    # TestPyPI holds loopmath only; its dependencies come from PyPI.
    testpypi) index=(--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/) ;;
    http://*|https://*|file://*) index=(--index-url "$online") ;;
    *) echo "release-check.sh: --online takes pypi, testpypi or an index URL" >&2; exit 2 ;;
  esac
  [ -n "$version" ] || version="$("$py" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' "$root/pyproject.toml")"
elif [ -n "$version" ]; then
  echo "release-check.sh: --version goes with --online" >&2; exit 2
fi

work="$(cd "$(mktemp -d /tmp/loopmath-release-check.XXXXXX)" && pwd -P)"
[ "$keep" = 1 ] || trap 'rm -rf "$work"' EXIT
dist="$work/dist"
venv="$work/venv"
home="$work/home"
mkdir -p "$home"
fixture="$root/tests/fixtures/graph/packaging_smoke/packaging-smoke.jsonl"
failed=0
deps="not verified"
house="$work/wheelhouse"

step() { printf '%-62s' "$1"; }
pass() { echo "PASS${1:+ ($1)}"; }
fail() { echo "FAIL${1:+ ($1)}"; failed=$((failed + 1)); }

# check TITLE CMD...: run CMD from $work with the clean env; PASS on exit 0.
check() {
  local title="$1"; shift
  step "$title"
  if out="$(cd "$work" && env -i PATH="$venv/bin:/usr/bin:/bin" HOME="$home" LOOPMATH_HOME="$work/store" \
            LOOPMATH_CACHE_DIR="$work/cache" NO_COLOR=1 "$@" 2>&1)"; then
    pass
  else
    fail "exit $?"
    printf '%s\n' "$out" | tail -5 | sed 's/^/    /'
  fi
}

echo "loopmath release check"
echo "tree: $root ($(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "python: $("$py" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "temp: $work"
[ -z "$online" ] || echo "online: loopmath==$version from ${index[*]}"
echo

if [ -z "$online" ]; then
# The wheel is built from the sdist, in a temp folder: that proves the sdist is complete and
# leaves no build/ in the tree.
step "build sdist, then wheel from it (no isolation, no network)"
if out="$("$py" -m build --no-isolation --outdir "$dist" "$root" 2>&1)"; then
  pass "$(cd "$dist" && ls | tr '\n' ' ')"
else
  fail; printf '%s\n' "$out" | tail -8 | sed 's/^/    /'
  echo; echo "release check: build failed"; exit 1
fi

step "twine check dist/*"
if out="$("$py" -m twine check --strict "$dist"/* 2>&1)"; then pass; else fail; printf '%s\n' "$out" | tail -5 | sed 's/^/    /'; fi

wheel="$(ls "$dist"/*.whl | head -1)"
step "wheel holds every file under src/loopmath but READMEs"
missing="$("$py" - "$root/src" "$wheel" <<'PY'
import sys, zipfile
from pathlib import Path
src, wheel = Path(sys.argv[1]), sys.argv[2]
names = set(zipfile.ZipFile(wheel).namelist())
want = [p.relative_to(src).as_posix() for p in sorted((src / "loopmath").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        and p.name not in (".DS_Store", "README.md")]
lost = [w for w in want if w not in names]
print(f"{len(lost)} of {len(want)} missing: " + " ".join(lost[:8]) if lost else "")
PY
)"
if [ -z "$missing" ]; then pass; else fail "$missing"; fi
fi

step "fresh venv under /tmp"
if "$py" -m venv "$venv" >/dev/null 2>&1; then pass; else fail; echo "release check: no venv"; exit 1; fi

if [ -n "$online" ]; then
  # As a user would: pip, the network, no cache, no config file. pip's install report names
  # every distribution it chose and where from.
  step "pip install loopmath==$version from the index (no cache)"
  if out="$(PIP_CONFIG_FILE=/dev/null "$venv/bin/python" -m pip install --no-input --disable-pip-version-check \
            --quiet --no-cache-dir "${index[@]}" --report "$work/install-report.json" "loopmath==$version" 2>&1)"; then
    pass
    deps="verified online: $("$venv/bin/python" - "$work/install-report.json" <<'PY'
import json, sys
from urllib.parse import urlsplit
items = json.load(open(sys.argv[1]))["install"]
hosts = sorted({urlsplit(i["download_info"]["url"]).netloc or "local" for i in items})
print(f"{len(items)} distributions from {', '.join(hosts)}: "
      + " ".join(f"{i['metadata']['name']}=={i['metadata']['version']}" for i in items))
PY
)"
  else
    fail; printf '%s\n' "$out" | tail -5 | sed 's/^/    /'
    echo; echo "release check: loopmath==$version did not install from $online"; exit 1
  fi
else
step "install the wheel and its declared core dependencies"
uv_out="uv not found"
if command -v uv >/dev/null 2>&1 && uv_out="$(uv pip install --offline --quiet --python "$venv/bin/python" "$wheel" 2>&1)"; then
  deps="verified: uv resolved the declared core dependencies offline from its cache"
  pass "uv cache"
else
  echo "NOT BY UV (it cannot resolve offline; next: local copies)"
  printf '%s\n' "$uv_out" | grep -v '^[[:space:]]*$' | tail -2 | sed 's/^[[:space:]]*/    uv offline: /'

  # The wheel's Requires-Dist (no extra) and their own requirements, walked through the build
  # interpreter's installed distributions; each one found is repacked as a wheel from its
  # RECORD. A missing or out-of-range distribution stops here. pip then does its own
  # resolution against these wheels only.
  step "repack the declared core closure as local wheels"
  if house_out="$("$py" - "$wheel" "$house" 2>&1 <<'PY'
import base64, email, hashlib, re, sys, zipfile
from importlib import metadata
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

wheel, house = sys.argv[1], Path(sys.argv[2])
house.mkdir(parents=True)
with zipfile.ZipFile(wheel) as z:
    meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    top = email.message_from_bytes(z.read(meta)).get_all("Requires-Dist") or []

def needs(reqs, extras):
    for text in reqs:
        req = Requirement(text)
        if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in ("", *extras)):
            yield req

seen, queue = {}, list(needs(top, ()))
while queue:
    req = queue.pop(0)
    name = canonicalize_name(req.name)
    try:
        dist = metadata.distribution(req.name)
    except metadata.PackageNotFoundError:
        sys.exit(f"{req} is declared but not installed in {sys.executable}")
    if not req.specifier.contains(dist.version, prereleases=True):
        sys.exit(f"{req} is declared but {sys.executable} has {dist.version}")
    extras = set(req.extras) | seen.get(name, (None, set()))[1]
    if name in seen and extras == seen[name][1]:
        continue
    seen[name] = (dist, extras)
    queue.extend(needs(dist.requires or [], tuple(extras)))

def b64(data):
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

for name, (dist, _) in sorted(seen.items()):
    files = [f for f in dist.files or [] if f.parts[0] != ".." and "__pycache__" not in f.parts]
    info = next(f.parts[0] for f in files if f.parts[0].endswith(".dist-info") and f.name == "METADATA")
    tags = [line.split(":", 1)[1].strip() for line in dist.read_text("WHEEL").splitlines() if line.startswith("Tag:")]
    tag = "-".join(".".join(sorted({t.split("-")[i] for t in tags})) for i in range(3))
    out = house / f"{re.sub(r'[-_.]+', '_', name)}-{dist.version}-{tag}.whl"
    record = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            rel = f.as_posix()
            if f.parts[0] == info and f.name in ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json"):
                continue
            data = Path(dist.locate_file(f)).read_bytes()
            z.writestr(rel, data)
            record.append(f"{rel},sha256={b64(data)},{len(data)}")
        record.append(f"{info}/RECORD,,")
        z.writestr(f"{info}/RECORD", "\n".join(record) + "\n")
print(f"{len(seen)} wheels: " + " ".join(f"{d.metadata['Name']}=={d.version}" for d, _ in
                                          (seen[n] for n in sorted(seen))))
PY
)"; then
    pass "$house_out"
    step "pip install the wheel from those wheels only (--no-index)"
    if out="$(PIP_CONFIG_FILE=/dev/null "$venv/bin/python" -m pip install --no-input --disable-pip-version-check \
              --quiet --no-index --only-binary :all: --find-links "$house" "$wheel" 2>&1)"; then
      deps="verified: pip resolved the declared core dependencies with no index, from wheels repacked"
      deps="$deps from $py's installed copies (versions are those, not an index's newest)"
      pass
    else
      fail; printf '%s\n' "$out" | tail -5 | sed 's/^/    /'
    fi
  else
    fail; printf '%s\n' "$house_out" | tail -5 | sed 's/^/    /'
  fi
fi
fi

# Nothing beyond loopmath's declared core closure: the extras' packages are absent.
step "venv lacks the extras (pymc, pytensor, arviz, matplotlib)"
extra_found="$(cd "$work" && env -i PATH="$venv/bin:/usr/bin:/bin" HOME="$home" "$venv/bin/python" -c '
import importlib.util
print(" ".join(m for m in ("pymc", "pytensor", "arviz", "matplotlib", "h5netcdf") if importlib.util.find_spec(m)))' 2>&1)"
if [ -z "$extra_found" ]; then pass; else fail "found: $extra_found"; fi

step "loopmath imports from the venv, not the tree"
where="$(cd "$work" && "$venv/bin/python" -c 'import loopmath; print(loopmath.__file__)' 2>&1)"
case "$where" in "$venv"/*) pass ;; *) fail "$where" ;; esac

check "loopmath --version" loopmath --version
check "loop --version (the alias)" loop --version
check "loopmath --help" loopmath --help
check "loopmath task-types --json" loopmath task-types --json
check "loopmath skill install --target both (temp HOME)" loopmath skill install --target both --json
check "loopmath skill show" loopmath skill show
check "loopmath doctor --json" loopmath doctor --json
check "loopmath analyze on the committed fixture" loopmath analyze --logs "$fixture" --all --no-cache

# The loop, from the installed wheel: its packaged data (OCP schemas, the workflow catalog,
# the prior bundle, the view pages) must ship. The store is a temp folder.
check "loopmath workflows list --json" loopmath workflows list --json
check "loopmath prior show --json" loopmath prior show --json
check "loopmath ocp validate the spec examples" loopmath ocp validate --json "$root"/spec/examples/v0.3/*.ocp.json
check "loopmath fit --json (the prior, an empty store)" loopmath fit --json
check "loopmath recommend --json" loopmath recommend --type feature --repo acme/web --feature size=m --json
check "loopmath recommend --html" loopmath recommend --type feature --repo acme/web --feature size=m --html
check "loopmath runs --html" loopmath runs --html
check "loopmath posterior --html" loopmath posterior --html
check "loopmath status --json" loopmath status --json

# The research verbs need the [bayes] extra, which this venv lacks: they must say so.
step "loopmath research fit names the missing [bayes] extra"
out="$(cd "$work" && env -i PATH="$venv/bin:/usr/bin:/bin" HOME="$home" LOOPMATH_HOME="$work/store" \
       LOOPMATH_CACHE_DIR="$work/cache" NO_COLOR=1 loopmath research fit 2>&1)"
code=$?
case "$code:$out" in 0:*) fail "exit 0" ;; *"loopmath[bayes]"*) pass "exit $code" ;; *) fail "exit $code: $(printf '%s' "$out" | tail -1)" ;; esac

echo
echo "dependencies: $deps"
if [ "$failed" -eq 0 ]; then
  echo "release check: all checks passed"
else
  echo "release check: $failed failed"
fi
[ "$failed" -eq 0 ]
