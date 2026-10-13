#!/usr/bin/env bash
# Release check (docs/release.md): build the sdist and wheel with one fixed time and no owner
# in their headers, twine check them, install the wheel into a fresh venv under /tmp, and run
# the installed commands from outside the tree.
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
# In both modes the installed prior (priors/bundle/ and priors/benchmarks.toml) must be this
# checkout's, byte for byte, with the same runs per source: the package carries the prior we fit
# with. --prior-only DIR runs that comparison alone on an installed `loopmath` folder.
#
# Usage: scripts/release-check.sh [--python PY] [--keep] [--online pypi|testpypi|URL [--version X]]
#        scripts/release-check.sh [--python PY] --prior-only DIR
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
prior_only=""
while [ $# -gt 0 ]; do
  case "$1" in
    --python) py="$2"; shift 2 ;;
    --keep) keep=1; shift ;;
    --online) online="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --prior-only) prior_only="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
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
if [ -n "$prior_only" ] && [ -n "$online" ]; then
  echo "release-check.sh: --prior-only checks an installed folder; it does not go with --online" >&2; exit 2
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

# prior_parity PKG: the prior the installed `loopmath` folder PKG carries is this checkout's. Every
# file of priors/bundle/ and priors/benchmarks.toml has the checkout's sha256 and none is on one
# side only; the installed manifest's runs per source are the checkout's, and each installed source
# file is the one its manifest names (sha256) with that many runs. One line per file group.
prior_parity() {
  local rows verdict title detail
  if ! rows="$("$py" - "$root/src/loopmath/priors" "$1/priors" 2>&1 <<'PY'
import gzip, hashlib, json, sys
from pathlib import Path

tree, inst = Path(sys.argv[1]), Path(sys.argv[2])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listing(folder):
    if not folder.is_dir():
        return {}
    return {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != ".DS_Store"}


def diff(mine, theirs):
    """How the installed files differ from the checkout's, as phrases."""
    out = []
    for words, names in (("differ", [n for n in sorted(set(mine) & set(theirs)) if mine[n] != theirs[n]]),
                         ("missing from the install", sorted(set(mine) - set(theirs))),
                         ("only in the install", sorted(set(theirs) - set(mine)))):
        if names:
            out.append(f"{words}: {', '.join(names)}")
    return out


def sources(folder):
    try:
        return json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("sources") or {}
    except (OSError, ValueError):
        return {}


def line(problems, title, ok):
    print("\t".join(("FAIL" if problems else "PASS", title, "; ".join(problems) if problems else ok)))


mine, theirs = listing(tree / "bundle"), listing(inst / "bundle")
problems = [] if theirs else [f"no bundle files in {inst / 'bundle'}"]
problems += diff(mine, theirs)
want = {name: e.get("runs") for name, e in sources(tree / "bundle").items()}
got = {name: e.get("runs") for name, e in sources(inst / "bundle").items()}
if got != want:
    problems.append("runs per source: installed " + (", ".join(f"{k} {v}" for k, v in got.items()) or "none")
                    + "; checkout " + (", ".join(f"{k} {v}" for k, v in want.items()) or "none"))
for name, e in sources(inst / "bundle").items():
    path = inst / "bundle" / str(e.get("file"))
    try:
        if sha(path) != e.get("sha256"):
            problems.append(f"{name}: {path.name} is not the file its manifest names")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            n = sum(1 for text in fh if text.strip())
        if n != e.get("runs"):
            problems.append(f"{name}: {n} runs in {path.name}, {e.get('runs')} in its manifest")
    except Exception as exc:  # a missing or damaged file is a finding, not a crash
        problems.append(f"{name}: {path.name} does not read ({type(exc).__name__})")
line(problems, "prior bundle equals the checkout",
     ", ".join(f"{k} {v}" for k, v in want.items()) + f"; {len(mine)} files")

toml = "benchmarks.toml"
mine = {toml: sha(tree / toml)} if (tree / toml).is_file() else {}
theirs = {toml: sha(inst / toml)} if (inst / toml).is_file() else {}
problems = diff(mine, theirs) if mine or theirs else [f"no {toml} in the checkout or the install"]
line(problems, f"{toml} equals the checkout",
     f"{(tree / toml).read_text(encoding='utf-8').count('[[benchmark]]') if mine else 0} benchmark entries")
PY
)"; then
    step "prior equals the checkout"; fail "the comparison stopped"
    printf '%s\n' "$rows" | tail -5 | sed 's/^/    /'
    return
  fi
  while IFS=$'\t' read -r verdict title detail; do
    step "$title"
    if [ "$verdict" = PASS ]; then pass "$detail"; else fail "$detail"; fi
  done <<< "$rows"
}

echo "loopmath release check"
echo "tree: $root ($(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "python: $("$py" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "temp: $work"
[ -z "$online" ] || echo "online: loopmath==$version from ${index[*]}"
[ -z "$prior_only" ] || echo "installed: $prior_only"
echo

if [ -n "$prior_only" ]; then
  prior_parity "$prior_only"
  echo
  if [ "$failed" -eq 0 ]; then echo "release check: the installed prior equals the checkout"; else echo "release check: $failed failed"; fi
  [ "$failed" -eq 0 ]
  exit
fi

if [ -z "$online" ]; then
# The upload files carry one fixed time and no owner: SOURCE_DATE_EPOCH is the release date of
# this version in CHANGELOG.md (midnight UTC) unless it is set. The wheel build honours it; the
# sdist is rewritten below, since setuptools stamps the builder's user, ids and file times.
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
  SOURCE_DATE_EPOCH="$("$py" - "$root" <<'PY'
import datetime, re, sys, tomllib
from pathlib import Path
root = Path(sys.argv[1])
version = tomllib.load(open(root / "pyproject.toml", "rb"))["project"]["version"]
m = re.search(rf"^## {re.escape(version)} \((\d{{4}}-\d\d-\d\d)\)", (root / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
day = datetime.date.fromisoformat(m.group(1)) if m else datetime.date(1980, 1, 1)
print(int(datetime.datetime(day.year, day.month, day.day, tzinfo=datetime.timezone.utc).timestamp()))
PY
)"
fi
export SOURCE_DATE_EPOCH

# The wheel is built from the sdist, in a temp folder: that proves the sdist is complete and
# leaves no build/ in the tree.
step "build sdist, then wheel from it (no isolation, no network)"
if out="$("$py" -m build --no-isolation --outdir "$dist" "$root" 2>&1)"; then
  pass "$(cd "$dist" && ls | tr '\n' ' ')"
else
  fail; printf '%s\n' "$out" | tail -8 | sed 's/^/    /'
  echo; echo "release check: build failed"; exit 1
fi

# Every tar header: uid and gid 0, no user or group name, the fixed time, no PAX record but a
# long path; the gzip header: no file name and the fixed time. Member bytes are unchanged.
step "sdist rewritten with no owner and one fixed time"
if out="$("$py" - "$dist" "$SOURCE_DATE_EPOCH" <<'PY' 2>&1
import gzip, io, os, sys, tarfile
from pathlib import Path
dist, epoch = Path(sys.argv[1]), int(sys.argv[2])
sdist = next(dist.glob("*.tar.gz"))
def members(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        return [(m.name, m.type, m.mode, m.linkname, t.extractfile(m).read() if m.isfile() else None) for m in t.getmembers()]
before = sdist.read_bytes()
buf = io.BytesIO()
with tarfile.open(fileobj=io.BytesIO(before), mode="r:gz") as src, \
        tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as out:
    for m in src.getmembers():
        data = src.extractfile(m).read() if m.isfile() else None
        m.uid = m.gid = 0
        m.uname = m.gname = ""
        m.mtime = epoch
        m.pax_headers = {}
        out.addfile(m, io.BytesIO(data) if data is not None else None)
raw = io.BytesIO()
with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as gz:
    gz.write(buf.getvalue())
after = raw.getvalue()
if members(after) != members(before):
    sys.exit("member names, modes or bytes changed")
tmp = sdist.with_name(sdist.name + ".tmp")
tmp.write_bytes(after)
os.replace(tmp, sdist)
print(f"{len(members(after))} members")
PY
)"; then pass "$out"; else fail; printf '%s\n' "$out" | tail -5 | sed 's/^/    /'; fi

step "archive headers name no one and carry the fixed time"
if out="$("$py" - "$dist" "$SOURCE_DATE_EPOCH" <<'PY' 2>&1
import sys, tarfile, time, zipfile
from pathlib import Path
dist, epoch = Path(sys.argv[1]), int(sys.argv[2])
when = time.gmtime(max(epoch, 315532800))[:6]
bad = []
for sdist in sorted(dist.glob("*.tar.gz")):
    head = sdist.read_bytes()[:10]
    if head[3] & 0x08:
        bad.append(f"{sdist.name}: the gzip header names a file")
    if int.from_bytes(head[4:8], "little") != epoch:
        bad.append(f"{sdist.name}: the gzip header time is not the fixed time")
    with tarfile.open(sdist) as t:
        for m in t.getmembers():
            if m.uid or m.gid or m.uname or m.gname:
                bad.append(f"{sdist.name}: {m.name}: an owner in the tar header")
            if m.mtime != epoch:
                bad.append(f"{sdist.name}: {m.name}: a tar header time that is not the fixed time")
            if set(m.pax_headers) - {"path", "linkpath"}:
                bad.append(f"{sdist.name}: {m.name}: PAX records " + ", ".join(sorted(set(m.pax_headers) - {"path", "linkpath"})))
for wheel in sorted(dist.glob("*.whl")):
    with zipfile.ZipFile(wheel) as z:
        if z.comment:
            bad.append(f"{wheel.name}: a zip comment")
        for i in z.infolist():
            if i.date_time != when:
                bad.append(f"{wheel.name}: {i.filename}: a zip time that is not the fixed time")
            if i.extra or i.comment:
                bad.append(f"{wheel.name}: {i.filename}: zip extra fields or a comment")
print("\n".join(bad[:20]) if bad else f"fixed time {time.strftime('%Y-%m-%d', time.gmtime(epoch))}")
sys.exit(1 if bad else 0)
PY
)"; then pass "$out"; else fail; printf '%s\n' "$out" | tail -20 | sed 's/^/    /'; fi

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

# The package carries the prior we fit with: this checkout's bundle and benchmark file.
prior_parity "$(dirname "$where")"

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
