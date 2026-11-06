# Update the loopmath fit

Refit on everything in the store, then show the results page. No question.

Use exactly the commands below. Flags and fields are in `loopmath guide reference`. Never run `--help`.

## 1. Fit

```sh
loopmath fit --json
```

Add options only when the user asked for them:
- "without the shipped data", "only my runs": `--no-prior`;
- "without RQ1" (or `e0`, `sweep`, `benchmark`, `shared`): `--without rq1`, repeatable.

A new fit replaces the one `recommend` and the pages read. Read `fit.id`, `fit.n_runs.user`, `fit.n_runs.prior`, `runs_by_source`, `options`, `seconds`, `dropped`, `shipped_overlap`.

`shipped_overlap` counts, per shipped source, the user's runs that the shipped prior also holds (the same runs); the fit uses the user's copies. When it has counts, the summary says one line, once: "N of your runs are also in the shipped SOURCE prior (same runs): fits use your copies", with the total for N ("1 of your runs is also in the shipped SOURCE prior (same run): fits use your copy" for one run), and with several sources "the shipped prior (rq1 40, e0 4)" for "the shipped SOURCE prior". Its `dropped` reason, "shipped copy of a stored run", is that same overlap: do not name it again.

If it exits 4 (another fit holds the lock), wait 30 seconds and run it once more. If `fit.n_runs.user` is 0 and the user expected runs, say so and suggest onboarding (`loopmath guide onboard`) or bringing existing runs (`loopmath guide import`).

## 2. Show the results page

```sh
loopmath posterior --html
```

It prints the page path. Open it (`open PATH` on macOS, `xdg-open PATH` on Linux); if that fails, give the path.

When the user asked about one workflow, add `--workflow CFG` (a `cfg_` id from their runs or a recommendation). For one task type and repo, add `--type TYPE --repo REPO`.

## 3. Summary

End with at most 5 lines, then stop:

> Fit fit_20260924181210 in 3.4 s: 52 of your runs plus 1,469 shipped runs (store level; options: none).
> 44 of your runs are also in the shipped rq1 prior (same runs): fits use your copies.
> Left out: 12 attempts without usable cost.
> Results page: /Users/me/.loopmath/views/posterior-20260924-181214.html

Name each option used, the overlap line when `shipped_overlap` has counts, and each other `dropped` reason with its count. `options.level` is the fit's level, not an option: say "store level" when its `level` is `store` (the default) and "shared level" when it is `shared` or `options` has no `level` (0.3.5's fit).
