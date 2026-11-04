"""One Lanczos step on a fine-tuned network (scripts/finetune.py), with scripts/lanczos.py's evaluator and analysis.

The frozen weights are sampled on fresh chains (resized starts, which agree with cold starts for our fine-tuned
states, scripts/ft_eval.py), then x = (H' psi)/psi and y = (H'^2 psi)/psi are computed per sample exactly as in
lanczos.py, and E1 = E(alpha*) comes from its summary (two-fold cross-fitted alpha, bootstrap errors).

usage: PYTHONPATH=. python -u scripts/ft_lanczos.py runs/ft/NAME.json [--chains 2048] [--sweeps 40]
       [--output runs/lanczos/ft_NAME.npz]
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from scripts.lanczos import evaluate, summary
from wsp import fast, vit
from wsp.system import make, make_state
from wsp.transfer import Settings, resize_samples, transfer

STARTS = {"ours": Settings(fill="uniform", fill_heads="nonlocal", local=0.7, window=True, window_avg_units=(3,),
                           sign_rule=True),
          "naive": Settings(fill="zero", sign_rule=True, sign_units=()),
          "scratch": Settings(fill="zero", sign_rule=True, sign_units=())}  # as in scripts/finetune.py

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("report", type=Path)
p.add_argument("--chains", type=int, default=2048)
p.add_argument("--sweeps", type=int, default=40)
p.add_argument("--chunk", type=int, default=32768)
p.add_argument("--block", type=int, default=8)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--limit", type=int, default=0, help="use only this many samples (smoke tests)")
p.add_argument("--c", type=float, help="fixed energy offset (NetKet units), so that shards can be pooled")
p.add_argument("--guide-floor", type=float, default=0.0,
               help="sample from psi_g: unit 3's |cosh(a3 + i b3)| floored at this value; samples carry log w = log|psi/psi_g|^2")
p.add_argument("--output", type=Path)
args = p.parse_args()
jax.config.update("jax_default_matmul_precision", "float32")
r = json.load(open(args.report))
l, start = r["l"], r["start"]
system = make(l)
saved = np.load(args.report.with_name(args.report.stem + "_params.npz"))
leaves, tree = jax.tree_util.tree_flatten_with_path(transfer(vit.load_params("main"), l, STARTS[start]))
params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved["/".join(str(k.key) for k in path)]) for path, _ in leaves])
rng = np.random.default_rng(args.seed)
resized = resize_samples(vit.load_samples("main"), l)
def node_lift(params, spins):
    """log|psi_g / psi| >= 0: lifts unit 3's amplitude node, |cosh(a3 + i b3)| -> max(|cosh|, floor) (its inputs are
    the window-averaged ones with window_avg_units=(3,)). Sampling |psi_g|^2 and weighting by |psi/psi_g|^2 is unbiased
    and bounds the near-node samples whose (H psi)/psi and (H^2 psi)/psi otherwise blow up (heavy-tailed y)."""
    lp, a3, b3 = fast._copy_diagnostics(params, spins, jnp.arange(spins.shape[-1]), jnp.int8(1))
    return lp, jnp.maximum(0.0, jnp.log(args.guide_floor) - 0.5 * jnp.log(jnp.sinh(a3) ** 2 + jnp.cos(b3) ** 2))


def log_psi_g(variables, spins):
    lp, lift = node_lift(variables["params"], jnp.atleast_2d(spins))
    return lp + lift


init = resized[rng.permutation(len(resized))[np.arange(args.chains) % len(resized)]]
vs = make_state(log_psi_g if args.guide_floor else vit.log_psi, params, init, chunk_size=args.chains, seed=args.seed)
vs.n_discard_per_chain = args.sweeps - 1
samples = np.asarray(vs.sample()).reshape(-1, system.N).astype(np.int8)
if args.limit:
    samples = samples[:args.limit]
logw = np.zeros(len(samples))
if args.guide_floor:
    lift_fn = jax.jit(lambda p_, s_: node_lift(p_, s_)[1])
    logw = -2 * np.concatenate([np.asarray(lift_fn(params, jnp.asarray(samples[i:i + 4096])))
                                for i in range(0, len(samples), 4096)])
    print(f"guided sampling: floor {args.guide_floor}, {np.mean(logw < 0):.2%} of samples lifted, "
          f"ESS fraction {np.exp(logw).sum() ** 2 / np.exp(2 * logw).sum() / len(logw):.3f}", flush=True)
ident = (np.arange(l * l, dtype=np.int32)[None], np.ones(1, np.int8))
e = fast.energy(fast.copy_terms(params, samples, *ident, chunk=args.chunk))
c = args.c if args.c is not None else float(np.round(e["mean"] * 4 * system.N, 3))  # conditioning offset
print(json.dumps({"report": str(args.report), "start": start, "l": l, "samples": len(samples), "E0_fast": e["mean"],
                  "c": c}), flush=True)
out = args.output or Path("runs/lanczos") / f"ft_{args.report.stem}.npz"
out.parent.mkdir(parents=True, exist_ok=True)


def done(count, part):
    np.savez(out, samples=samples[:count], c=c, l=l, source=str(args.report), logw=logw[:count],
             guide_floor=args.guide_floor, **part)
    if count >= 64 and count % 256 < args.block and not args.guide_floor:
        s = summary(part["x"], part["y"], c, system.N, n_boot=200)
        print(f"  {count}: E0 {s['E0']:.6f}  E1 {s['E1']:.6f} +/- {s['E1_se']:.6f}  dE {s['dE']:+.6f} +/- "
              f"{s['dE_se']:.6f}  V0 {s['V0']:.4f} V1 {s['V1']:.4f}  ESS {s['ess_fraction']:.2f}", flush=True)


x, y, diag, ref = evaluate(params, samples, *ident, c, args.chunk, args.block, verbose=False, done=done)
if args.guide_floor:  # the weighted estimate is pooled by scripts/pool_lanczos.py; record the weighted E0 here
    w = np.exp(logw)
    print(f"FINAL guided: weighted E0 {(np.sum(w * x.real) / w.sum() + c) / (4 * system.N):.6f}", flush=True)
    print("saved", out, flush=True)
    raise SystemExit
s = summary(x, y, c, system.N)
plain = {k: (None if v is None else float(v)) for k, v in s.items() if v is None or isinstance(v, (int, float, np.number))}
print("FINAL " + json.dumps(plain), flush=True)
json.dump({"report": str(args.report), "samples": len(samples), "c": c, "E0_fast": e["mean"], "E0_fast_err": e["err"],
           "summary": plain}, open(out.with_suffix(".json"), "w"), indent=1)
print("saved", out, flush=True)
