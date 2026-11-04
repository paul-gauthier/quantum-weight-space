"""Fine-tune a network on an l x l lattice with MinSR (NetKet's VMC_SR, use_ntk), from one of three starts.

The question is not whether fine-tuning helps, but whether our training-free transfer is a better place to start
than what a practitioner would do without it. Every start has a fixed sign, so every fine-tune trains the amplitude:
- ours:    the released Plain weights through our transfer: attention rule (uniform fill on nonlocal heads, cut 0.7),
           window head with unit 3 window-averaged, and the sign rule (Marshall + unit 3's flips);
- naive:   the released weights dropped onto the larger lattice (zero-padded J, kappa = 1, the global head), with
           Marshall's sign pi * M_A instead of the head's phase;
- scratch: the same architecture as naive, randomly initialized, with Marshall's sign.
After training, the frozen weights are measured on fresh chains with the fast evaluator.

usage: PYTHONPATH=. python -u scripts/finetune.py START L --steps 300 --lr 0.003 --output runs/ft/NAME.json
       [--samples 4096] [--diag-shift 1e-4] [--therm 20] [--eval-chains 16384] [--eval-sweeps 30] [--fp64]
"""
import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import netket as nk

from wsp import fast, vit
from wsp.system import make, make_state
from wsp.transfer import Settings, resize_samples, transfer

STARTS = {
    "ours": Settings(fill="uniform", fill_heads="nonlocal", local=0.7, window=True, window_avg_units=(3,),
                     sign_rule=True),
    "naive": Settings(fill="zero", sign_rule=True, sign_units=()),
}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("start", choices=["ours", "naive", "scratch"])
p.add_argument("l", type=int)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=0.003)
p.add_argument("--diag-shift", type=float, default=1e-4)
p.add_argument("--clip", type=float, default=5.0, help="local energies clipped to median +- clip * MAD in the update")
p.add_argument("--samples", type=int, default=4096, help="chains = samples per step (one sweep apart)")
p.add_argument("--therm", type=int, default=20, help="sweeps before the first step")
p.add_argument("--eval-chains", type=int, default=16384)
p.add_argument("--eval-sweeps", type=int, default=30, help="sweeps before the final fresh-chain measurement")
p.add_argument("--chunk", type=int, default=512, help="samples per forward chunk (local energies)")
p.add_argument("--chunk-bwd", type=int, default=512, help="samples per Jacobian chunk")
p.add_argument("--eval-chunk", type=int, default=16384, help="configs per forward batch in the final measurement")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fp64", action="store_true")
p.add_argument("--init-params", type=Path, help="continue from a saved *_params.npz of the same start")
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
jax.config.update("jax_default_matmul_precision", "float32")  # full fp32, no TF32
l, t0 = args.l, time.time()
system = make(l)
rng = np.random.default_rng(args.seed)
P = vit.load_params("main")


def random_params(template, key):
    """Same shapes as template: kernels ~ N(0, 1/fan_in), biases 0, LayerNorm scale 1, J ~ N(0, 1/tokens)."""
    leaves, tree = jax.tree_util.tree_flatten_with_path(template)
    keys = jax.random.split(key, len(leaves))
    out = []
    for (path, leaf), k in zip(leaves, keys):
        name, shape = path[-1].key, leaf.shape
        if name == "kernel":
            out.append(jax.random.normal(k, shape, leaf.dtype) / math.sqrt(shape[0]))
        elif name == "J":
            out.append(jax.random.normal(k, shape, leaf.dtype) / math.sqrt(shape[-1]))
        elif name == "scale":
            out.append(jnp.ones(shape, leaf.dtype))
        elif name == "sign_rule":
            out.append(leaf)
        else:
            out.append(jnp.zeros(shape, leaf.dtype))
    return jax.tree_util.tree_unflatten(tree, out)


if args.start == "scratch":
    params = random_params(transfer(P, l, STARTS["naive"]), jax.random.PRNGKey(args.seed))
else:
    params = transfer(P, l, STARTS[args.start])
if args.init_params:  # continue a run: same tree, saved leaves
    saved = np.load(args.init_params)
    leaves, tree = jax.tree_util.tree_flatten_with_path(params)
    params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved["/".join(str(k.key) for k in path)]) for path, _ in leaves])
params = fast.cast(params, jnp.float64 if args.fp64 else jnp.float32)
n_params = sum(x.size for x in jax.tree.leaves(params))


def initial_states(n):
    if args.start == "scratch" or l < vit.L:
        return np.stack([rng.permutation(np.repeat(np.array([-1, 1], np.int8), system.N // 2)) for _ in range(n)])
    resized = resize_samples(vit.load_samples("main"), l)
    return resized[rng.permutation(len(resized))[np.arange(n) % len(resized)]]


RAW = {}  # unclipped energy statistics of the last step (the driver is a frozen pytree: no new attributes)


class ClippedSR(nk.driver.VMC_SR):
    """VMC_SR whose update uses local energies clipped to median +- clip * MAD, the same for every start (a naive
    transfer has local-energy outliers of 1e2-1e8 per site that make one unclipped step diverge). The logged energy
    is the unclipped estimate."""
    clip = 5.0

    def compute_loss_and_update(self):
        state, estimators = self.state, self.state.local_estimators

        def clipped(op, **kwargs):
            e = estimators(op, **kwargs)
            RAW["stats"] = nk.stats.statistics(e)
            med = jnp.median(e.real)
            mad = jnp.median(jnp.abs(e.real - med))
            return jnp.clip(e.real, med - self.clip * mad, med + self.clip * mad) + 1j * e.imag

        state.local_estimators = clipped
        try:
            return super().compute_loss_and_update()
        finally:
            del state.local_estimators


vs = make_state(vit.log_psi, params, initial_states(args.samples), chunk_size=args.chunk, seed=args.seed)
vs.n_discard_per_chain = args.therm
vs.sample()  # thermalize the chains once; each step then advances every chain by one sweep
vs.n_discard_per_chain = 0
driver = ClippedSR(system.hamiltonian, nk.optimizer.Sgd(learning_rate=args.lr), diag_shift=args.diag_shift,
                   variational_state=vs, use_ntk=True, chunk_size_bwd=args.chunk_bwd)
ClippedSR.clip = args.clip
report = {"start": args.start, "l": l, "settings": None if args.start == "scratch" else vars(STARTS[args.start]),
          "n_params": int(n_params), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
          "devices": [str(d) for d in jax.devices()], "trace": [], "status": "training"}
args.output.parent.mkdir(parents=True, exist_ok=True)
print(json.dumps({k: report[k] for k in ("start", "l", "n_params", "args")}), flush=True)


def save():
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(args.output)


t_train = time.time()


def callback(step, log_data, driver):
    e = RAW["stats"]  # unclipped energy of the samples the step was computed on, before the update
    row = {"step": int(step), "E": float(e.mean.real) * system.PER_SITE, "err": float(e.error_of_mean) * system.PER_SITE,
           "var": float(e.variance) * system.PER_SITE ** 2 * system.N,
           "E_clipped": float(driver._loss_stats.mean.real) * system.PER_SITE, "seconds": time.time() - t_train}
    report["trace"].append(row)
    if step % 10 == 0 or step == args.steps - 1:
        print(f"step {step:4d}  E/N {row['E']:.5f} +- {row['err']:.5f}  N*var/N^2 {row['var']:.2e}"
              f"  ({row['seconds']:.0f}s)", flush=True)
        save()
    return bool(np.isfinite(row["E_clipped"]))


driver.run(n_iter=args.steps, callback=callback, out=None, show_progress=False)
trained = vs.parameters
np.savez(args.output.with_name(args.output.stem + "_params.npz"),
         **{"/".join(str(k.key) for k in path): np.asarray(x)
            for path, x in jax.tree_util.tree_flatten_with_path(trained)[0]})

# Fresh chains on the frozen weights; energies from the fast evaluator, as for every other reported energy.
report["status"] = "evaluating"
save()
ev = make_state(vit.log_psi, trained, initial_states(args.eval_chains), chunk_size=args.eval_chains, seed=args.seed + 1)
ident = (np.arange(l * l, dtype=np.int32)[None], np.ones(1, np.int8))
report["eval"] = []
every = args.eval_sweeps // 3
for k in range(1, 4):
    ev.n_discard_per_chain = every - 1
    s = np.asarray(ev.sample()).reshape(-1, system.N).astype(np.int8)
    e = fast.energy(fast.copy_terms(trained, s, *ident, chunk=args.eval_chunk))
    report["eval"].append({"sweeps": k * every, "E": e["mean"], "err": e["err"], "var": e["var"],
                           "vscore": system.N * e["var"] / e["mean"] ** 2})
    print(f"eval sweep {k * every:3d}  E/N {e['mean']:.6f} +- {e['err']:.6f}  V {report['eval'][-1]['vscore']:.4f}",
          flush=True)
report["status"], report["seconds"] = "complete", time.time() - t0
save()
print("saved", args.output, flush=True)
