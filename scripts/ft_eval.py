"""Equilibration check for a fine-tuned network (scripts/finetune.py): frozen weights on mixed starts.

Half the chains start cold (random S^z = 0 states), half from resized released samples. A state whose cold and
resized groups disagree is not equilibrated, and its energy from one kind of start is not its variational energy.
Energies come from the fast evaluator, per group, every --every sweeps.

usage: PYTHONPATH=. python -u scripts/ft_eval.py runs/ft/NAME.json [--chains 16384] [--sweeps 60] [--every 10]
       [--output runs/ft/NAME_mixed.json]
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from wsp import fast, vit
from wsp.system import make, make_state
from wsp.transfer import Settings, resize_samples, transfer

# the starts of scripts/finetune.py (scratch has naive's shapes)
STARTS = {"ours": Settings(fill="uniform", fill_heads="nonlocal", local=0.7, window=True, window_avg_units=(3,),
                           sign_rule=True),
          "naive": Settings(fill="zero", sign_rule=True, sign_units=()),
          "scratch": Settings(fill="zero", sign_rule=True, sign_units=())}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("report", type=Path)
p.add_argument("--chains", type=int, default=16384)
p.add_argument("--sweeps", type=int, default=60)
p.add_argument("--every", type=int, default=10)
p.add_argument("--chunk", type=int, default=16384)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--output", type=Path)
args = p.parse_args()
jax.config.update("jax_default_matmul_precision", "float32")
r = json.load(open(args.report))
l, start = r["l"], r["start"]
system = make(l)
saved = np.load(args.report.with_name(args.report.stem + "_params.npz"))
template = transfer(vit.load_params("main"), l, STARTS[start])
leaves, tree = jax.tree_util.tree_flatten_with_path(template)
params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved["/".join(str(k.key) for k in path)]) for path, _ in leaves])
rng = np.random.default_rng(args.seed)
half = args.chains // 2
cold = np.stack([rng.permutation(np.repeat(np.array([-1, 1], np.int8), system.N // 2)) for _ in range(half)])
resized = resize_samples(vit.load_samples("main"), l)
warm = resized[rng.permutation(len(resized))[np.arange(args.chains - half) % len(resized)]]
vs = make_state(vit.log_psi, params, np.concatenate([cold, warm]), chunk_size=args.chains, seed=args.seed)
groups = {"cold": slice(0, half), "resized": slice(half, args.chains)}
ident = (np.arange(l * l, dtype=np.int32)[None], np.ones(1, np.int8))
out = {"report": str(args.report), "start": start, "l": l, "chains": args.chains, "trace": []}
print(json.dumps({k: out[k] for k in ("report", "start", "l", "chains")}), flush=True)
for k in range(1, args.sweeps // args.every + 1):
    vs.n_discard_per_chain = args.every - 1
    s = np.asarray(vs.sample()).reshape(-1, system.N).astype(np.int8)
    t = fast.copy_terms(params, s, *ident, chunk=args.chunk)
    row = {"sweeps": k * args.every, "all": fast.energy(t)}
    for g, idx in groups.items():
        row[g] = fast.energy(t, idx=idx)
    out["trace"].append({key: ({m: float(v) for m, v in val.items()} if isinstance(val, dict) else val)
                         for key, val in row.items()})
    print(f"sweep {k * args.every:3d}  all {row['all']['mean']:.6f} +- {row['all']['err']:.6f}  cold "
          f"{row['cold']['mean']:.6f}  resized {row['resized']['mean']:.6f}", flush=True)
dest = args.output or args.report.with_name(args.report.stem + "_mixed.json")
dest.write_text(json.dumps(out, indent=1))
print("saved", dest, flush=True)
