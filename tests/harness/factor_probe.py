"""Single-seed probes of harness factor recovery, on the simulation of `test_factor_recovery.py`.

    python tests/harness/factor_probe.py [--runs 1200] [--seeds 100-103] [--scale 1 0.3] [--extras none prompt]
                                         [--form hand] [--out FILE]

For each seed, base scale and choice of extras: one fit, then per head the hfactor scale, the slope of the named
estimates on their truths (nonzero effects), the error against the no-learning error, the mean posterior sd over
the factor scale, the root mean square z, and the 80 and 95% hits. `--extras` takes `none`, `all`, or a comma list
of `hook`, `prompt` and `pq`. `--scale` multiplies the prior scale of the effects the simulation draws (base nodes
and prompts), not the named truths. `--form` is `hand` (the hand-set truths, the default), `prior` or `control`.
Every choice draws the same random numbers, so the runs and the named parts of each tree are shared. `--out`
writes every result, with each named estimate, as JSON. Fits run here, so run it through the heavy-job wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_factor_recovery as R  # noqa: E402


def seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def extras(text: str) -> tuple[str, ...]:
    if text == "none":
        return ()
    if text == "all":
        return R.EXTRAS
    got = tuple(text.split(","))
    bad = set(got) - set(R.EXTRAS)
    if bad:
        raise argparse.ArgumentTypeError(f"unknown extras {sorted(bad)}; use none, all or {','.join(R.EXTRAS)}")
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=1200)
    ap.add_argument("--seeds", type=seeds, default=seeds("100-103"))
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--extras", type=extras, nargs="+", default=[()])
    ap.add_argument("--form", choices=R.FORMS, default="hand")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    results = []
    print("seed scale extras head phi_hfactor slope mse_ratio sd_over_scale z_rms hit80 hit95")
    with tempfile.TemporaryDirectory() as tmp:
        for seed in a.seeds:
            for scale in a.scale:
                for ex in a.extras:
                    r = R.replicate(seed, a.runs, Path(tmp) / f"s{seed}-{scale}-{'-'.join(ex)}", a.form,
                                    extras=ex, scale=scale)
                    results.append(r)
                    for head in ("cost", "success"):
                        ln, c = r["learning"][head], r["heads"][head]
                        n = c["zero"]["n"] + c["nonzero"]["n"]
                        print(f"{seed} {scale:g} {','.join(ex) or 'none'} {head} {ln['phi']:.3f} {ln['slope']:.3f} "
                              f"{ln['mse_ratio']:.3f} {ln['sd_over_scale']:.3f} {ln['z_rms']:.2f} "
                              f"{c['zero']['hit80'] + c['nonzero']['hit80']}/{n} "
                              f"{c['zero']['hit95'] + c['nonzero']['hit95']}/{n}", flush=True)
    if a.out:
        a.out.write_text(json.dumps({"argv": sys.argv[1:] if argv is None else argv, "results": results}, indent=1),
                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
