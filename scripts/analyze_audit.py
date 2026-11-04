"""Read scripts/ft_audit.py runs against a fixed equilibration rule (stated below), set before the runs.

Per state (seed lineages pooled; the first half of each file's chains started cold, the second half resized) and per
observable (E, m2, dx, dy, minority), on the snapshots after the midpoint (sweeps 110-200 of 200): the weighted mean
(guided runs carry logw) with a chain bootstrap error, cold vs resized, the first vs the second half of that window, and
lag-1/lag-2 autocorrelations of the 10-sweep snapshots. Equilibrated iff every observable passes both 2-sigma tests.

usage: python scripts/analyze_audit.py NAME=FILE[,FILE...] [...] [--json OUT]
"""
import argparse
import json

import numpy as np

OBS = ("E", "m2", "dx", "dy", "minority")
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("runs", nargs="+")
p.add_argument("--boot", type=int, default=1000)
p.add_argument("--json")
args = p.parse_args()
rng = np.random.default_rng(0)


def load(files):
    d = [np.load(f) for f in files]
    meta = [json.loads(str(z["meta"])) for z in d]
    snaps = min(z["E"].shape[0] for z in d)
    data = {k: np.concatenate([z[k][:snaps] for z in d], axis=1) for k in OBS + ("logw",)}
    cold = np.concatenate([np.arange(z["E"].shape[1]) < z["E"].shape[1] // 2 for z in d])
    sweeps = meta[0]["every"] * np.arange(1, snaps + 1)
    return data, cold, sweeps, meta


def chain_mean(w, o, chains, t):
    """Weighted mean over the given chains and snapshots, and its chain-bootstrap error."""
    sw, swo = w[t][:, chains].sum(0), (w[t] * o[t])[:, chains].sum(0)
    boot = [(lambda i: swo[i].sum() / sw[i].sum())(rng.integers(0, len(sw), len(sw))) for _ in range(args.boot)]
    return float(swo.sum() / sw.sum()), float(np.std(boot))


def autocorr(o, t, lag):
    x = o[t] - o[t].mean()
    return float((x[:-lag] * x[lag:]).mean() / (x * x).mean())


out = {}
for spec in args.runs:
    name, files = spec.split("=")
    data, cold, sweeps, meta = load(files.split(","))
    w = np.exp(data["logw"])
    window = np.nonzero(sweeps > sweeps[-1] / 2)[0]
    early, late = window[: len(window) // 2], window[len(window) // 2:]
    allc, coldc, warmc = np.ones(len(cold), bool), cold, ~cold
    res = {"files": files.split(","), "chains": int(len(cold)), "sweeps": int(sweeps[-1]),
           "window": [int(sweeps[window[0]]), int(sweeps[window[-1]])], "guide_floor": meta[0]["guide_floor"],
           "observables": {}}
    ok = True
    print(f"== {name}: {len(cold)} chains, sweeps {sweeps[window[0]]}-{sweeps[window[-1]]} of {sweeps[-1]}"
          f"{', guided' if meta[0]['guide_floor'] else ''}")
    for k in OBS:
        o = data[k]
        m, e = chain_mean(w, o, allc, window)
        c_, ce = chain_mean(w, o, coldc, window)
        r_, re = chain_mean(w, o, warmc, window)
        a_, ae = chain_mean(w, o, allc, early)
        b_, be = chain_mean(w, o, allc, late)
        z_cw, z_el = (c_ - r_) / np.hypot(ce, re), (a_ - b_) / np.hypot(ae, be)
        passed = bool(abs(z_cw) < 2 and abs(z_el) < 2)
        ok &= passed
        trace = [chain_mean(w, o, allc, [t])[0] for t in range(len(sweeps))]
        res["observables"][k] = {"mean": m, "se": e, "cold": [c_, ce], "resized": [r_, re], "z_cold_resized": z_cw,
                                 "early": [a_, ae], "late": [b_, be], "z_early_late": z_el, "pass": passed,
                                 "rho1": autocorr(o, window, 1), "rho2": autocorr(o, window, 2), "trace": trace}
        f = ".6f" if k == "E" else ".5f"
        print(f"  {k:8s} {m:{f}} ± {e:{f}}   cold {c_:{f}} resized {r_:{f}} ({z_cw:+.1f}σ)   early {a_:{f}} late {b_:{f}}"
              f" ({z_el:+.1f}σ)   ρ1 {res['observables'][k]['rho1']:+.2f} ρ2 {res['observables'][k]['rho2']:+.2f}"
              f"   {'pass' if passed else 'FAIL'}")
    res["equilibrated"] = bool(ok)
    print(f"  -> {'equilibrated' if ok else 'NOT equilibrated'} by the frozen rule")
    out[name] = res
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
