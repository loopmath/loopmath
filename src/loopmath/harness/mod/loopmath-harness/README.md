# loopmath-harness

A Claude Code mod for `loopmath harness`: configure, plan, start and watch a harness tune without leaving the
session. The mod runs the `loopmath` CLI by argv and folds its event stream. It starts no Claude turn and makes
no model call itself; the tune's own sessions run in the engine, which the CLI starts detached.

## Install

    pip install loopmath
    loopmath harness mod install            # prints the two claude plugin commands, then asks

`--print` only prints them; `--scope project` installs for one repo. `loopmath harness mod uninstall` reverses
both. The mod is copied to `$LOOPMATH_HOME/mods/loopmath-harness-<version>/` first, so it works offline.

## Use

- `/harness` opens the configure pane. Each section (Task, Score, Loop, Models, Screen, Budget, Run) draws its
  fields from `loopmath harness schema --json`; a change runs `loopmath harness config set KEY VALUE`, so the
  TOML file stays the record and the CLI checks every value. An empty value unsets the key.
- **Plan** runs the dry run (no model call) and shows sessions, the cost point and range, the worst-case time,
  the design's residual degrees of freedom and any warning or failed check.
- **Start** asks once more ("Spend up to X USD"); the second press runs `loopmath harness run --detach --yes`.
- `/harness watch [ID]` opens the progress pane: stages, sessions, edits, the effects table and the delivery,
  with Stop, Show and Apply (a dry run first, then "Write these files"). A band above the prompt and the status
  line show the live tune in one line.
- `/harness stop [ID]`, `/harness show [ID]` and `/harness apply [ID]` (add `--yes` to write) run the matching
  CLI verbs.

Closing the session or reloading the mod leaves the tune running; the progress feed picks it up again.

## Options

- `loopmath_bin`: the loopmath executable (default `loopmath`).
- `config_path`: the config file (default `loopmath-harness.toml` in the session's folder).

## Check

    loopmath harness mod validate --test

runs `claude plugin validate --strict` and `claude plugin test` under a throwaway `CLAUDE_CONFIG_DIR`.
