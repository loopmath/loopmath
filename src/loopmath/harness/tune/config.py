"""The config file `loopmath-harness.toml`: load, merge flags and `--set`, validate, render, freeze.

Precedence is flag, then file, then default (`schema.FIELDS`); there is no environment layer apart from
`LOOPMATH_HOME`. Unknown keys in the file are kept, never refused, and written back. `run` freezes the resolved
config into the tune folder as `config.toml` plus its sha256 (`freeze`), and the event stream carries that digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import CONFIG_SCHEMA
from .contracts import Config, ConfigError, Task, TuneDir
from .schema import (EDITORS, TABLES, Field, by_key, defaults, editor_defaults, parse_duration, parse_value,
                     table_fields)

__all__ = ["DEFAULT_PATH", "load", "resolve", "validate", "render", "freeze", "load_frozen", "apply_set",
           "set_editors", "task_of", "TEST_WRITER_TOOLS"]

DEFAULT_PATH = "loopmath-harness.toml"
TEST_WRITER_TOOLS = ("Read", "Glob", "Grep", "Write", "Edit")
FULL_MAX_EDITS = 4
PB12_MAX_EDITS = 10


def _merge(base: dict, over: Mapping) -> dict:
    """`over` laid on `base`: tables merge key by key; `editors` replaces the list as a whole, each entry laid on
    the per-editor defaults of its place (focus cycling procedure, domain, ...; unnamed entries are named from
    their focus later); anything else replaces."""
    out = dict(base)
    for key, value in over.items():
        if key == "editors" and isinstance(value, list):
            eds = []
            for i, ed in enumerate(value, start=1):
                start = {**editor_defaults(i), "name": ""}
                eds.append({**start, **(dict(ed) if isinstance(ed, Mapping) else {"_bad": ed})})
            out["editors"] = eds
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"no config file at {path}; `loopmath harness init` writes one") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from None


def _typed(data: dict) -> dict:
    """Every known field parsed to its type (ConfigError on the first bad one); unknown keys kept as they are."""
    out = dict(data)
    for table in TABLES:
        if table == "editors":
            eds = out.get("editors")
            if not isinstance(eds, list):
                raise ConfigError("editors: an array of tables ([[editors]])", "editors")
            fixed = []
            for i, ed in enumerate(eds, start=1):
                if not isinstance(ed, Mapping) or "_bad" in ed:
                    raise ConfigError(f"editors.{i}: a table", f"editors.{i}")
                ed = dict(ed)
                for f in table_fields("editors"):
                    if f.name in ed:
                        ed[f.name] = _parse(f, ed[f.name], f"editors.{i}.{f.name}")
                fixed.append(ed)
            out["editors"] = fixed
            continue
        tab = out.get(table)
        if not isinstance(tab, Mapping):
            raise ConfigError(f"[{table}]: a table", table)
        tab = dict(tab)
        for f in table_fields(table):
            if f.name in tab:
                tab[f.name] = _parse(f, tab[f.name], f.key)
        out[table] = tab
    return out


def _parse(f: Field, value: Any, key: str) -> Any:
    try:
        return parse_value(f, value)
    except ValueError as exc:
        raise ConfigError(str(exc).replace(f.key, key, 1), key) from None


def _name_editors(eds: list[dict]) -> None:
    """Give an unnamed editor its focus as its name, numbered when taken; refuse two editors of one name."""
    seen: set[str] = set()
    for i, ed in enumerate(eds, start=1):
        if not ed.get("name"):
            base = ed.get("focus") or "procedure"
            name, k = base, 1
            while name in seen:
                k += 1
                name = f"{base}-{k}"
            ed["name"] = name
        if ed["name"] in seen:
            raise ConfigError(f"editors.{i}.name: {ed['name']!r} is taken by an earlier editor", f"editors.{i}.name")
        if not all(c.isalnum() or c in "-_" for c in ed["name"]):
            raise ConfigError(f"editors.{i}.name: letters, digits, - and _ only, not {ed['name']!r}",
                              f"editors.{i}.name")
        seen.add(ed["name"])


def set_editors(data: dict, n: int) -> dict:
    """`--editors N`: keep the first N editors and fill new ones with the defaults (focus cycling)."""
    if n < 0:
        raise ConfigError("--editors: at least 0", "editors")
    eds = [dict(e) for e in data.get("editors") or []][:n]
    while len(eds) < n:
        i = len(eds) + 1
        eds.append(dict(EDITORS[i - 1]) if i <= len(EDITORS) else editor_defaults(i, [e.get("name") for e in eds]))
    return {**data, "editors": eds}


def apply_set(data: dict, assignment: str) -> dict:
    """One `--set KEY=VALUE`: a dotted key (`budget.usd=6`, `editors.2.model=claude-sonnet-5-5`); the value is read
    as TOML when it parses (numbers, booleans, quoted text, lists), else as plain text, then parsed to the field's
    type. An unknown key is kept, as the file keeps one."""
    key, sep, text = assignment.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"--set {assignment!r}: give KEY=VALUE", key)
    raw = text.strip()
    value = _toml_value(raw)
    parts = key.split(".")
    out = {**data}
    try:
        f = by_key(key)
    except KeyError:
        f = None
    if f is not None:
        if f.type == "str" and not isinstance(value, str):
            value = raw
        value = _parse(f, value, key)
    if parts[0] == "editors":
        if len(parts) != 3 or not parts[1].isdigit() or int(parts[1]) < 1:
            raise ConfigError(f"--set {key}: an editor key is editors.N.KEY, N from 1", key)
        i = int(parts[1])
        out = set_editors(out, max(i, len(out.get("editors") or [])))
        eds = [dict(e) for e in out["editors"]]
        eds[i - 1][parts[2]] = value
        out["editors"] = eds
        return out
    cur = out
    for part in parts[:-1]:
        nxt = cur.get(part)
        nxt = dict(nxt) if isinstance(nxt, Mapping) else {}
        cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return out


def _toml_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def resolve(path: str | Path | None = None, *, flags: Mapping[str, Any] | None = None,
            sets: Iterable[str] = (), editors: int | None = None, editor_model: str | None = None,
            editor_effort: str | None = None, data: Mapping | None = None, check: bool = True) -> Config:
    """The config from the defaults, the file at `path` (or `data`, already read), then the flags.

    `flags` maps dotted keys to values (`{"executor.model": "claude-opus-5-5"}`, as a flag gives them; None
    means the flag was not given). `editors` is `--editors N`; `editor_model` and `editor_effort` set every editor;
    `sets` are `--set KEY=VALUE`, applied last. ConfigError on any bad value; with `check`, `validate` too.
    """
    merged = defaults()
    src = ""
    if data is not None:
        merged = _merge(merged, data)
    elif path is not None:
        p = Path(path).expanduser()
        merged = _merge(merged, _read(p))
        src = str(p)
    schema_id = merged.get("schema", CONFIG_SCHEMA)
    if schema_id != CONFIG_SCHEMA:
        raise ConfigError(f"schema {schema_id!r}: this loopmath reads {CONFIG_SCHEMA}", "schema")
    if editors is not None:
        merged = set_editors(merged, editors)
    for key, value in (flags or {}).items():
        if value is not None:
            merged = apply_set(merged, f"{key}={value if isinstance(value, str) else _toml_text(value)}")
    for ed in merged.get("editors") or []:
        if editor_model is not None:
            ed["model"] = editor_model
        if editor_effort is not None:
            ed["effort"] = editor_effort
    for s in sets:
        merged = apply_set(merged, s)
    typed = _typed(merged)
    _name_editors(typed["editors"])
    cfg = Config(data=typed, path=src)
    if check:
        errs = validate(cfg)
        if errs:
            raise ConfigError(errs[0][1], errs[0][0])
    return cfg


def load(path: str | Path = DEFAULT_PATH, **kw: Any) -> Config:
    """`resolve` on the file at `path`."""
    return resolve(path, **kw)


def validate(cfg: Config) -> list[tuple[str, str]]:
    """[(key, reason)] for rules across fields; empty when the config can run."""
    g = cfg.get
    errs: list[tuple[str, str]] = []
    sources = [bool(g("task.prompt")), bool(g("task.prompt_file")), bool(g("task.set"))]
    if sum(sources) > 1:
        errs.append(("task.prompt", "give one of task.prompt, task.prompt_file and task.set"))
    has_command = bool(g("score.command"))
    judge = g("judge.kind")
    if not has_command and judge == "none":
        errs.append(("score.command", "nothing scores the runs: give score.command or a judge"))
    combine = g("score.combine")
    if combine in ("gate", "mean") and not (has_command and judge != "none"):
        errs.append(("score.combine", f"combine = {combine} needs both a score command and a judge"))
    if combine == "command" and not has_command:
        errs.append(("score.combine", "combine = command needs score.command"))
    if combine == "judge" and judge == "none":
        errs.append(("score.combine", "combine = judge needs a judge"))
    if g("score.kind") == "junit" and has_command and not g("score.junit_path"):
        errs.append(("score.junit_path", "score.kind = junit needs score.junit_path"))
    if judge == "rubric" and not g("judge.rubric_file"):
        errs.append(("judge.rubric_file", "judge.kind = rubric needs judge.rubric_file"))
    if g("edit.enabled") and not cfg.editors:
        errs.append(("editors", "edit.enabled needs at least one editor"))
    if not g("edit.enabled") and not g("edit.from"):
        errs.append(("edit.from", "edit.enabled = false needs edit.from (your own edit folders)"))
    for i, ed in enumerate(cfg.editors, start=1):
        if ed.get("focus") == "file" and not ed.get("focus_file"):
            errs.append((f"editors.{i}.focus_file", "focus = file needs focus_file"))
    design, m = g("screen.design"), g("screen.max_edits")
    if design == "pb12" and m > PB12_MAX_EDITS:
        errs.append(("screen.max_edits", f"the pb12 design takes at most {PB12_MAX_EDITS} edits"))
    if design == "full" and m > FULL_MAX_EDITS:
        errs.append(("screen.max_edits", f"the full design takes at most {FULL_MAX_EDITS} edits"))
    if design == "fold-over" and g("screen.pairs") % 2:
        errs.append(("screen.pairs", "the fold-over design needs an even number of pairs"))
    rule = g("deliver.rule")
    if g("final.runs") == 0 and rule in ("final-vs-base-mean", "gate"):
        errs.append(("final.runs", f"final.runs = 0: deliver.rule {rule} reads the final run; use fit or "
                                   "always-build"))
    if g("loop.overlap"):
        errs.append(("loop.overlap", "loop.overlap = true is not supported in this release: the next round's "
                                     "editors start after the reflector ends; set loop.overlap = false"))
    try:
        parse_duration(g("budget.wall"))
    except ValueError as exc:
        errs.append(("budget.wall", str(exc)))
    if g("auth.mode") == "api-key-env" and not g("auth.api_key_env"):
        errs.append(("auth.api_key_env", "auth.mode = api-key-env needs auth.api_key_env"))
    return errs


# ---------------------------------------------------------------------------------------------------- render


def _toml_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ConfigError(f"{value!r} has no TOML form")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_text(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_key(k)} = {_toml_text(v)}" for k, v in value.items()) + "}"
    raise ConfigError(f"{value!r} has no TOML form")


def _key(k: str) -> str:
    return k if k and all(c.isalnum() or c in "-_" for c in k) else json.dumps(k)


def _line(key: str, value: Any, f: Field | None, comments: bool) -> str:
    text = f"{_key(key)} = {_toml_text(value)}"
    if comments and f is not None:
        note = f.help + (f" ({' | '.join(map(str, f.choices))})" if f.choices and f.type != "str-list" else "")
        text = f"{text:<27} # {note}"
    return text


def render(data: Mapping | Config, comments: bool = True) -> str:
    """The config as TOML, tables in FIELDS order, one comment per known field when `comments`; unknown keys
    and tables follow, kept as they are. `init` writes `render(schema.defaults())`."""
    if isinstance(data, Config):
        data = data.data
    lines = [f"schema = {_toml_text(data.get('schema', CONFIG_SCHEMA))}"]
    scalars = [(k, v) for k, v in data.items() if k not in TABLES and k != "schema" and not isinstance(v, Mapping)
               and not (isinstance(v, list) and v and all(isinstance(x, Mapping) for x in v))]
    for k, v in scalars:
        lines.append(_line(k, v, None, comments))
    for table in list(TABLES) + [k for k, v in data.items() if k not in TABLES and isinstance(v, Mapping)]:
        if table not in data:
            continue
        if table == "editors":
            for ed in data["editors"]:
                lines += ["", "[[editors]]"]
                known = {f.name: f for f in table_fields("editors")}
                for k in list(known) + [k for k in ed if k not in known]:
                    if k in ed:
                        lines.append(_line(k, ed[k], known.get(k), comments))
            continue
        tab = data[table]
        lines += ["", f"[{_key(table)}]"]
        known = {f.name: f for f in table_fields(table)}
        for k in list(known) + [k for k in tab if k not in known]:
            if k in tab:
                lines.append(_line(k, tab[k], known.get(k), comments))
    return "\n".join(lines) + "\n"


def freeze(cfg: Config, tune: TuneDir) -> Config:
    """Write the resolved config into the tune folder as config.toml and its sha256; the digest is returned on the
    config, and `tune.start` carries it."""
    from ...store.lock import atomic_write_text

    text = render(cfg, comments=False)
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    tune.root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(tune.config, text)
    atomic_write_text(tune.config_digest, digest + "\n")
    return Config(data=cfg.data, path=str(tune.config), digest=digest)


def load_frozen(tune: TuneDir) -> Config:
    """The frozen config of a tune (resume): ConfigError when config.toml no longer matches its sha256."""
    text = tune.config.read_text(encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    want = tune.config_digest.read_text(encoding="utf-8").strip() if tune.config_digest.exists() else digest
    if digest != want:
        raise ConfigError(f"{tune.config} changed after the tune started (sha256 {digest} is not {want})")
    cfg = resolve(data=tomllib.loads(text), check=False)
    return Config(data=cfg.data, path=str(tune.config), digest=digest)


# ---------------------------------------------------------------------------------------------------- the task


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigError(f"git {' '.join(args)} in {repo}: {exc}") from None
    if out.returncode != 0:
        raise ConfigError(f"git {' '.join(args)} in {repo}: {(out.stderr or out.stdout).strip()[:300]}")
    return out.stdout.strip()


def task_of(cfg: Config, *, base_dir: Path | None = None, git: bool = True) -> Task:
    """The single task the config names: its prompt (task.prompt, or task.prompt_file read), its repo and base
    commit (`git rev-parse`), and its stable id. A task set is read by the set runner, one Task per line."""
    root = Path(base_dir) if base_dir is not None else (Path(cfg.path).parent if cfg.path else Path.cwd())
    repo = (root / Path(cfg.get("task.repo")).expanduser()).resolve()
    prompt = cfg.get("task.prompt") or ""
    if not prompt and cfg.get("task.prompt_file"):
        pf = (root / Path(cfg.get("task.prompt_file")).expanduser())
        try:
            prompt = pf.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"task.prompt_file {pf}: {exc}", "task.prompt_file") from None
    if not prompt.strip():
        raise ConfigError("no task: give task.prompt, task.prompt_file or task.set", "task.prompt")
    ref = cfg.get("task.base_ref") or "HEAD"
    commit = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}") if git else ""
    tid = cfg.get("task.id") or hashlib.sha256(f"{prompt}\0{commit}".encode("utf-8")).hexdigest()[:12]
    return Task(id=tid, prompt=prompt, repo=repo, base_ref=ref, base_commit=commit,
                score_command=cfg.get("score.command") or "")
