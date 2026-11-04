"""Symmetry sector of a fine-tuned one-copy network, priced on saved samples.

For each of the 64 elements g (4 in-patch shifts x 8 point ops about the lattice centre x spin flip; translations
by whole patches are exact in the architecture), R_g = E_{|psi|^2}[psi(g s) / psi(s)] = <psi|T_g^-1|psi> / <psi|psi>.
The expected ground-state sector (zero momentum, even under spin flip, fully symmetric point group) has every
R_g = +1 for an exactly symmetric state; R_g near -1 would mean the wrong sector. 1 - Re R_g = |psi - T psi|^2 / 2
is the asymmetry, and the mean of R_g over the group is the state's weight in the symmetric sector. Guided samples are
weighted by exp(logw); errors are a sample bootstrap.

usage: PYTHONPATH=. python -u scripts/ft_symmetry.py runs/ft/NAME.json SAMPLES.npz [...] [--limit 2048] --output OUT.json
"""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from wsp import fast, native, vit
from wsp.transfer import Settings, transfer

STARTS = {"ours": Settings(fill="uniform", fill_heads="nonlocal", local=0.7, window=True, window_avg_units=(3,),
                           sign_rule=True)}  # as in scripts/finetune.py

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("report", type=Path)
p.add_argument("samples", nargs="+")
p.add_argument("--limit", type=int, default=2048)
p.add_argument("--chunk", type=int, default=1024)
p.add_argument("--output", type=Path)
args = p.parse_args()
jax.config.update("jax_default_matmul_precision", "float32")
r = json.load(open(args.report))
l = r["l"]
saved = np.load(args.report.with_name(args.report.stem + "_params.npz"))
leaves, tree = jax.tree_util.tree_flatten_with_path(transfer(vit.load_params("main"), l, STARTS[r["start"]]))
params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved["/".join(str(k.key) for k in path)]) for path, _ in leaves])
data = [np.load(f) for f in args.samples]
samples = np.concatenate([d["samples"].reshape(-1, l * l) for d in data]).astype(np.int8)
logw = np.concatenate([d["logw"][-1] if d["logw"].ndim == 2 else d["logw"] for d in data])
sel = np.random.default_rng(0).permutation(len(samples))[:args.limit]
samples, w = samples[sel], np.exp(logw[sel])
perms, flips = native.copies("symm_trxy_ising", l)  # shift-major: 4 shifts x 8 point ops, then the same spin-flipped
labels = [f"shift{dr}{dc}_{'rot%d' % (90 * k) + ('T' if t else '')}{'_flip' if f < 0 else ''}"
          for f in (1, -1) for dr in (0, 1) for dc in (0, 1) for k in range(4) for t in (0, 1)]
t0 = time.time()
lp = np.empty((len(perms), len(samples)), np.complex128)
for g, (perm, flip) in enumerate(zip(perms, flips)):
    for idx, k in fast._blocks(len(samples), min(args.chunk, len(samples))):
        lp[g, idx[:k]] = np.asarray(fast._copy_logpsi(params, jnp.asarray(samples[idx]), perm, flip))[:k]
assert labels[0] == "shift00_rot0" and np.allclose(lp[0], lp[0])
R = np.exp(lp - lp[0])
mean = lambda idx: (w[idx] * R[:, idx]).sum(1) / w[idx].sum()
rng = np.random.default_rng(1)
boot = np.array([mean(rng.integers(0, len(w), len(w))) for _ in range(500)])
m, se = mean(np.arange(len(w))), boot.std(0)
sector = lambda b: b.mean(-1).real
out = {"report": str(args.report), "l": l, "samples": len(samples), "seconds": time.time() - t0,
       "symmetric_weight": float(sector(m)), "symmetric_weight_se": float(sector(boot).std()),
       "max_imag": float(np.abs(m.imag).max()),
       "ops": {lab: {"R": float(m[g].real), "se": float(se[g].real), "abs_ratio_spread": float(np.abs(R[g]).std())}
               for g, lab in enumerate(labels)}}
print(f"{l}x{l}: weight in the symmetric sector {out['symmetric_weight']:.5f} ± {out['symmetric_weight_se']:.5f}, "
      f"min R {m.real.min():.5f}, max |Im R| {out['max_imag']:.1e}  ({out['seconds']:.0f}s)", flush=True)
for g in np.argsort(m.real)[:6]:
    print(f"  {labels[g]:22s} R {m[g].real:.5f} ± {se[g].real:.5f}", flush=True)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
