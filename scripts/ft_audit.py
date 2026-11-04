"""Equilibration audit for a fine-tuned network: long mixed-start chains that track slow observables, not just E.

Like scripts/ft_eval.py (half the chains cold, half from resized released samples), but every --every sweeps each
chain records its local energy and cheap observables, so cold/resized agreement and drift can be judged per
observable (scripts/analyze_audit.py):
- E: local energy per site;
- m2: S^zz(pi, pi) / N = (sum (-1)^(i+j) sigma / N)^2;
- dx, dy: NN sigma^z sigma^z on bonds inside a 2x2 patch minus bonds across a patch edge (x and y bonds);
- minority: unit 3 on its minority side of the node (cos b3 > 0), the network's sign-flip sector.
With --guide-floor the chains sample the guide of scripts/ft_lanczos.py (unit 3's node floored) and each sample
carries logw = log |psi/psi_g|^2. With --zero-shot L the untrained transfer (start "ours") is audited instead.
The last snapshot's samples are saved for scripts/ft_s2.py and scripts/ft_symmetry.py.

usage: PYTHONPATH=. python -u scripts/ft_audit.py runs/ft/NAME.json --chains 8192 --sweeps 200 --every 10
       [--guide-floor 0.02] [--seed 7] --output runs/audit/NAME.npz
       PYTHONPATH=. python -u scripts/ft_audit.py --zero-shot 14 ... --output runs/audit/zs14.npz
"""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from wsp import fast, vit
from wsp.system import make, make_state
from wsp.transfer import Settings, resize_samples, transfer

STARTS = {"ours": Settings(fill="uniform", fill_heads="nonlocal", local=0.7, window=True, window_avg_units=(3,),
                           sign_rule=True)}  # as in scripts/finetune.py

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("report", type=Path, nargs="?")
p.add_argument("--zero-shot", type=int, metavar="L", help="audit the untrained transfer on an L x L lattice")
p.add_argument("--chains", type=int, default=8192)
p.add_argument("--sweeps", type=int, default=200)
p.add_argument("--every", type=int, default=10)
p.add_argument("--chunk", type=int, default=16384)
p.add_argument("--guide-floor", type=float, default=0.0)
p.add_argument("--fp32", action="store_true", help="cast the weights to fp32 (shifts E/N by ~1e-7)")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
jax.config.update("jax_default_matmul_precision", "float32")
if args.zero_shot:
    l, start = args.zero_shot, "ours"
    params = fast.cast(transfer(vit.load_params("main"), l, STARTS[start]), jnp.float64)
else:
    r = json.load(open(args.report))
    l, start = r["l"], r["start"]
    saved = np.load(args.report.with_name(args.report.stem + "_params.npz"))
    leaves, tree = jax.tree_util.tree_flatten_with_path(transfer(vit.load_params("main"), l, STARTS[start]))
    params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved["/".join(str(k.key) for k in path)])
                                                 for path, _ in leaves])
if args.fp32:
    params = fast.cast(params, jnp.float32)
system = make(l)
N = system.N


def lift(a3, b3):
    """log |psi_g / psi| >= 0 for the guide of scripts/ft_lanczos.py: |cosh(a3 + i b3)| floored at --guide-floor."""
    return jnp.maximum(0.0, jnp.log(args.guide_floor) - 0.5 * jnp.log(jnp.sinh(a3) ** 2 + jnp.cos(b3) ** 2))


def log_psi_g(variables, spins):
    lp, a3, b3 = fast._copy_diagnostics(variables["params"], jnp.atleast_2d(spins), jnp.arange(N), jnp.int8(1))
    return lp + lift(a3, b3)


rng = np.random.default_rng(args.seed)
half = args.chains // 2
cold = np.stack([rng.permutation(np.repeat(np.array([-1, 1], np.int8), N // 2)) for _ in range(half)])
resized = resize_samples(vit.load_samples("main"), l)
warm = resized[rng.permutation(len(resized))[np.arange(args.chains - half) % len(resized)]]
vs = make_state(log_psi_g if args.guide_floor else vit.log_psi, params, np.concatenate([cold, warm]),
                chunk_size=args.chains, seed=args.seed)
ident = (np.arange(N, dtype=np.int32)[None], np.ones(1, np.int8))
ii, jj = np.meshgrid(np.arange(l), np.arange(l), indexing="ij")
stag = ((-1.0) ** (ii + jj)).reshape(-1)


def observables(s):
    g = s.reshape(-1, l, l).astype(np.float64)
    hx, hy = g * np.roll(g, -1, axis=2), g * np.roll(g, -1, axis=1)  # bonds (i, j)-(i, j+1) and (i, j)-(i+1, j)
    return {"m2": (s.astype(np.float64) @ stag / N) ** 2,
            "dx": hx[:, :, 0::2].mean((1, 2)) - hx[:, :, 1::2].mean((1, 2)),
            "dy": hy[:, 0::2, :].mean((1, 2)) - hy[:, 1::2, :].mean((1, 2))}


names = ("E", "m2", "dx", "dy", "minority", "logw")
series = {k: [] for k in names}
meta = {"report": str(args.report), "zero_shot": args.zero_shot, "start": start, "l": l, "chains": args.chains,
        "every": args.every, "guide_floor": args.guide_floor, "seed": args.seed, "fp32": args.fp32}
print(json.dumps(meta), flush=True)
t0 = time.time()
for k in range(1, args.sweeps // args.every + 1):
    vs.n_discard_per_chain = args.every - 1
    s = np.asarray(vs.sample()).reshape(-1, N).astype(np.int8)
    t = fast.copy_terms(params, s, *ident, chunk=args.chunk, diagnostics=True)
    a3, b3 = t["a3"][0], t["b3"][0]
    row = {"E": (t["A"][0] / t["B"][0]).real * t["per_site"], **observables(s),
           "minority": (np.cos(b3) > 0).astype(np.float64),
           "logw": -2 * np.asarray(lift(a3, b3)) if args.guide_floor else np.zeros(len(s))}
    for key in names:
        series[key].append(row[key])
    w = np.exp(row["logw"])
    text = []
    for g, idx in (("cold", slice(0, half)), ("resized", slice(half, None))):
        m = {key: (w[idx] * row[key][idx]).sum() / w[idx].sum() for key in ("E", "m2", "dy", "minority")}
        text.append(f"{g} E {m['E']:.6f} m2 {m['m2']:.5f} dy {m['dy']:+.5f} minority {m['minority']:.4f}")
    print(f"sweep {k * args.every:3d}  " + "  |  ".join(text) + f"  ({time.time() - t0:.0f}s)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{key: np.stack(v) for key, v in series.items()}, samples=s, sweeps=k * args.every,
             meta=json.dumps(meta))
print("saved", args.output, flush=True)
