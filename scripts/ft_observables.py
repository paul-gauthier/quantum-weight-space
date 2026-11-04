"""Free sanity checks on saved guided Lanczos shards (scripts/ft_lanczos.py): samples, weights, diag and x only.

Per size, weighted by w = |psi/psi_g|^2, with sample-bootstrap errors:
- zz share <diag> / <E_loc>: exactly 1/3 in an S = 0 state;
- bond modulation: NN sigma^z sigma^z inside a 2x2 patch minus across a patch edge, x and y bonds (0 if translation
  invariant by one site);
- m2 = S^zz(pi, pi) / N, S^zz(q) over the zone (its largest values), and zz correlations C(r) along the axis and the
  diagonal from each in-patch origin, staggered, r <= L/2 (looking for structure near r = 5 left from the 10x10 torus);
- the two-point zero-variance extrapolation of (var, E) before and after the Lanczos step (lanczos.summary's formula,
  weighted moments).

usage: PYTHONPATH=. python scripts/ft_observables.py [--root results/lanczos] --output OUT.json
"""
import argparse
import glob
import json
import os

import numpy as np

from scripts.lanczos import at_moments, best_alpha_m, moments

# results/sanity/observables.json also has "18_300" (shards ft18g_s* of the 300-step 18x18 state, not included here)
GROUPS = {"12": "ft12g_s*", "14": "ft14g_s*", "16": "ft16g_s*", "18_600": "ft18cg_s*"}
EXACT = {12: -0.49729, 14: -0.49707, 16: -0.49695, 18: -0.49688}  # estimated exact energies (±3e-5 systematic)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--root", default="results/lanczos")
p.add_argument("--boot", type=int, default=300)
p.add_argument("--output")
args = p.parse_args()
rng = np.random.default_rng(0)


def load(pattern):
    seen, d = set(), []
    for f in sorted(glob.glob(f"{args.root}/**/{pattern}.npz", recursive=True)):
        if os.path.basename(f) not in seen:
            seen.add(os.path.basename(f))
            d.append(np.load(f))
    return d


def stat(w, f):
    """Weighted mean of per-sample rows f (n, ...) and its bootstrap error."""
    mean = lambda i: np.tensordot(w[i], f[i], axes=(0, 0)) / w[i].sum()
    boot = np.array([mean(rng.integers(0, len(w), len(w))) for _ in range(args.boot)])
    return mean(np.arange(len(w))), boot.std(0)


out = {}
for name, pattern in GROUPS.items():
    d = load(pattern)
    s = np.concatenate([z["samples"] for z in d]).astype(np.float64)
    w = np.exp(np.concatenate([z["logw"] for z in d]))
    c = float(d[0]["c"])
    x = np.concatenate([z["x"] for z in d])
    y = np.concatenate([z["y"] for z in d])
    diag = np.concatenate([z["diag"] for z in d])
    n, N = s.shape
    l = int(round(np.sqrt(N)))
    g = s.reshape(n, l, l)
    per = 1 / (4 * N)
    res = {"samples": n, "shards": len(d)}

    share = lambda i: (w[i] * diag[i]).sum() / (w[i] * (x.real[i] + c)).sum()
    res["zz_share"] = float(share(np.arange(n)))
    res["zz_share_se"] = float(np.std([share(rng.integers(0, n, n)) for _ in range(args.boot)]))

    hx, hy = g * np.roll(g, -1, axis=2), g * np.roll(g, -1, axis=1)
    mod = np.stack([hx[:, :, 0::2].mean((1, 2)) - hx[:, :, 1::2].mean((1, 2)),
                    hy[:, 0::2, :].mean((1, 2)) - hy[:, 1::2, :].mean((1, 2))], 1)
    m, se = stat(w, mod)
    res["modulation_x"], res["modulation_y"] = [float(m[0]), float(se[0])], [float(m[1]), float(se[1])]

    f = np.fft.fft2(g) / N  # sum_r e^{-i q r} sigma_r / N
    sq, sq_se = stat(w, np.abs(f) ** 2 * N)  # S^zz(q) (Pauli units)
    res["m2"] = [float(sq[l // 2, l // 2] / N), float(sq_se[l // 2, l // 2] / N)]
    res["S_pipi"], res["S_pi0"] = float(sq[l // 2, l // 2]), float((sq[l // 2, 0] + sq[0, l // 2]) / 2)
    order = np.argsort(sq.ravel())[::-1][:4]
    res["S_top"] = [[f"({2 * i / l:.3g}π, {2 * j / l:.3g}π)", float(sq[i, j])]
                    for i, j in (np.unravel_index(k, sq.shape) for k in order)]

    rs = np.arange(1, l // 2 + 1)
    corr = {}
    for label, (di, dj) in {"axis": (0, 1), "diag": (1, 1)}.items():
        for oi in (0, 1):
            for oj in (0, 1):
                rows = [(g[:, oi::2, oj::2] * np.roll(np.roll(g, -r * di, 1), -r * dj, 2)[:, oi::2, oj::2]).mean((1, 2))
                        for r in rs]
                m, se = stat(w, np.stack(rows, 1))
                sign = (-1.0) ** (rs * (di + dj))
                corr[f"{label}_origin{oi}{oj}"] = [[float(a), float(b)] for a, b in zip(sign * m, se)]
    res["C_staggered"] = corr

    mom = moments(x, y)
    point = lambda i: (lambda mm: (lambda a: (mm[0], mm[1] - mm[0] ** 2, *at_moments(mm, a)))(best_alpha_m(mm)))(
        np.tensordot(w[i], mom[i], axes=(0, 0)) / w[i].sum())
    e0, v0, e1, v1 = point(np.arange(n))
    ext = lambda e0_, v0_, e1_, v1_: (e1_ - v1_ * (e0_ - e1_) / (v0_ - v1_) + c) * per
    boot = [ext(*point(rng.integers(0, n, n))) for _ in range(args.boot)]
    res["zero_variance"] = {"E0": float((e0 + c) * per), "E1_insample": float((e1.real + c) * per),
                            "var0_over_N": float(v0.real * per ** 2 * N), "var1_over_N": float(v1.real * per ** 2 * N),
                            "E_extrapolated": float(ext(e0, v0, e1, v1).real), "se": float(np.std(np.real(boot))),
                            "exact_estimate": EXACT[l]}
    out[name] = res
    print(f"{name:7s} zz share {res['zz_share']:.5f} ± {res['zz_share_se']:.5f}   modulation x {res['modulation_x'][0]:+.4f}"
          f" ± {res['modulation_x'][1]:.4f}  y {res['modulation_y'][0]:+.4f} ± {res['modulation_y'][1]:.4f}   m2 "
          f"{res['m2'][0]:.5f}   S(pi,pi) {res['S_pipi']:.2f}  S(pi,0) {res['S_pi0']:.3f}", flush=True)
    print(f"        top S(q): {res['S_top']}")
    for key in ("axis_origin00", "axis_origin01", "axis_origin10", "axis_origin11", "diag_origin00", "diag_origin11"):
        print(f"        C {key:14s} " + " ".join(f"{a:+.4f}" for a, _ in corr[key]))
    z = res["zero_variance"]
    print(f"        zero-variance: E0 {z['E0']:.6f} (var {z['var0_over_N']:.2e}) -> E1 {z['E1_insample']:.6f} (var "
          f"{z['var1_over_N']:.2e}) -> E(var=0) {z['E_extrapolated']:.5f} ± {z['se']:.5f}  vs exact est. {EXACT[l]}",
          flush=True)
if args.output:
    json.dump(out, open(args.output, "w"), indent=1)
