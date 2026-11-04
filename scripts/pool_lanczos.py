"""Pool ft_lanczos.py shards (same c) into one cross-fitted E1 with a bootstrap error, against a record energy.

The cross-fitted E1 is averaged over random half splits (alpha fitted on one half, scored on the other); its error is the
bootstrap spread of that estimator. Prints one line: samples, E0, E1, margin to the record in sigma.

usage: PYTHONPATH=. python scripts/pool_lanczos.py 'GLOB' --sites 196 --record -0.49674 [--boot 300] [--json OUT]
"""
import argparse
import glob
import json

import numpy as np

from scripts.lanczos import at_moments, best_alpha_m, moments

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("pattern")
p.add_argument("--sites", type=int, required=True)
p.add_argument("--record", type=float, required=True)
p.add_argument("--boot", type=int, default=300)
p.add_argument("--alpha-files", type=int, default=0,
               help="fit alpha on the first K shards and score E1 at that fixed alpha on the rest (stable with heavy tails)")
p.add_argument("--json")
args = p.parse_args()
files = sorted(set(glob.glob(args.pattern)))
d = [np.load(f) for f in files]
cs = {round(float(z["c"]), 6) for z in d}
assert len(cs) == 1, f"shards have different offsets c: {cs}"
c, per = cs.pop(), 1 / (4 * args.sites)
floors = {float(z["guide_floor"]) if "guide_floor" in z.files else 0.0 for z in d}
assert len(floors) == 1, f"guided and unguided shards mixed: {floors}"
x = np.concatenate([z["x"] for z in d])
y = np.concatenate([z["y"] for z in d])
w = np.exp(np.concatenate([z["logw"] if "logw" in z.files else np.zeros(len(z["x"])) for z in d]))
m = moments(x, y) * w[:, None]  # guided sampling (ft_lanczos.py --guide-floor): weight w = |psi/psi_g|^2 per sample


def wmean(idx):
    return m[idx].sum(0) / w[idx].sum()


def crossfit(idx, rng, splits=1):
    vals = []
    for _ in range(splits):
        h = rng.permutation(idx)
        a, b = h[: len(h) // 2], h[len(h) // 2:]
        for fit, ev in ((a, b), (b, a)):
            vals.append(at_moments(wmean(ev), best_alpha_m(wmean(fit)))[0])
    return (np.mean(vals) + c) * per


if args.alpha_files:  # alpha from independent shards; E1 at a fixed alpha is a valid variational energy with a stable error
    sizes = [len(z["x"]) for z in d]
    k = sum(sizes[: args.alpha_files])
    fit, ev = np.arange(k), np.arange(k, len(x))
    alpha = best_alpha_m(wmean(fit))
    e1f = (at_moments(wmean(ev), alpha)[0] + c) * per
    e0f = (wmean(ev)[0] + c) * per
    rng = np.random.default_rng(0)
    bs = np.array([(at_moments(wmean(rng.choice(ev, len(ev))), alpha)[0] + c) * per for _ in range(args.boot)])
    se = bs.std()
    print(f"alpha {alpha:.4f} from {args.alpha_files} shard(s) ({k} samples); scored on {len(ev)}: E0 {e0f:.6f}  "
          f"E1 {e1f:.6f} ± {se:.6f}  vs record {args.record}: {(args.record - e1f) * 1e5:+.1f}e-5 = {(args.record - e1f) / se:.2f} sigma")
    if args.json:
        json.dump({"alpha": float(alpha), "alpha_samples": int(k), "samples": int(len(ev)), "E0": e0f, "E1": e1f, "E1_se": se,
                   "record": args.record, "sigma": (args.record - e1f) / se, "files": files}, open(args.json, "w"), indent=1)
    raise SystemExit
rng = np.random.default_rng(0)
n = len(x)
e1 = crossfit(np.arange(n), rng, splits=10)
e0 = (wmean(np.arange(len(x)))[0] + c) * per
boot = np.array([crossfit(rng.integers(0, n, n), rng) for _ in range(args.boot)])
se = boot.std()
out = {"shards": len(files), "samples": n, "guide_floor": floors.pop(), "ess_fraction": float(w.sum() ** 2 / (w ** 2).sum() / n), "E0": e0, "E1_crossfit": e1, "E1_se": se, "record": args.record,
       "margin": args.record - e1, "sigma": (args.record - e1) / se}
print(f"{len(files)} shards, {n} samples: E0 {e0:.6f}  E1 {e1:.6f} ± {se:.6f}  vs record {args.record}: "
      f"{(args.record - e1) * 1e5:+.1f}e-5 = {(args.record - e1) / se:.2f} sigma")
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
