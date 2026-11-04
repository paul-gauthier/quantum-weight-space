"""<S^2> of a fine-tuned network, from saved samples (scripts/ft_audit.py or scripts/ft_lanczos.py snapshots).

At S^z = 0, S^2 = N/2 + sum over unordered antiparallel pairs {i, j} of the exchange P_ij, so the local estimator is
    S2_loc(s) = N/2 + sum_{i up, j down} psi(s^ij) / psi(s),
(N/2)^2 extra forward passes per sample. Exact for an eigenstate: 0 for a singlet, S(S+1) in general. Guided samples
are weighted by exp(logw). The estimator itself is checked against exact diagonalization by --selftest (4x4).

usage: PYTHONPATH=. python -u scripts/ft_s2.py runs/ft/NAME.json SAMPLES.npz [...] [--limit 2048] --output OUT.json
       PYTHONPATH=. python -u scripts/ft_s2.py --selftest
"""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def pairs(samples):
    """rows, i, j for every (up, down) pair of every sample."""
    out = []
    for r, s in enumerate(samples):
        up, dn = np.nonzero(s > 0)[0], np.nonzero(s < 0)[0]
        i, j = np.repeat(up, len(dn)), np.tile(dn, len(up))
        out.append((np.full(len(i), r, np.int32), i.astype(np.int32), j.astype(np.int32)))
    return tuple(np.concatenate(v) for v in zip(*out))


def s2_local(log_psi, samples, chunk=65536):
    """S2_loc per sample for any batched log_psi(spins) (jax)."""
    @jax.jit
    def ratio_sum(s_dev, rows, i, j, mask, ref):
        s = s_dev[rows]
        ar = jnp.arange(s.shape[0])
        s = s.at[ar, i].multiply(-1).at[ar, j].multiply(-1)
        val = mask * jnp.exp(log_psi(s) - ref[rows])
        return jax.ops.segment_sum(val, rows, num_segments=ref.shape[0], indices_are_sorted=True)

    s_dev = jnp.asarray(samples)
    ref = log_psi(s_dev)
    rows, i, j = pairs(samples)
    acc = 0
    for start in range(0, len(rows), chunk):
        idx = np.arange(start, start + chunk)
        k = min(chunk, len(rows) - start)
        idx[k:] = start + k - 1
        acc = acc + ratio_sum(s_dev, jnp.asarray(rows[idx]), jnp.asarray(i[idx]), jnp.asarray(j[idx]),
                              jnp.asarray((np.arange(chunk) < k).astype(np.float64)), ref)
    return samples.shape[1] / 2 + np.asarray(acc).real


def selftest():
    """4x4 J1-J2 (J2 = 0.5): the ground state must give 0, and a random S^z = 0 state its exact <S^2>."""
    import netket as nk
    from wsp.system import make
    system = make(4)
    hi = system.hilbert
    states = np.asarray(hi.all_states())
    h = system.hamiltonian.to_sparse()
    e, v = np.linalg.eigh(h.toarray())
    s2op = sum(nk.operator.spin.sigmax(hi, a) @ nk.operator.spin.sigmax(hi, b)
               + nk.operator.spin.sigmay(hi, a) @ nk.operator.spin.sigmay(hi, b)
               + nk.operator.spin.sigmaz(hi, a) @ nk.operator.spin.sigmaz(hi, b)
               for a in range(16) for b in range(16)) * 0.25
    s2 = s2op.to_sparse().toarray().real
    rng = np.random.default_rng(0)
    for name, vec in (("ground state", v[:, 0]), ("random", rng.normal(size=len(states)))):
        vec = vec / np.linalg.norm(vec)
        table = jnp.asarray(np.log(np.abs(vec) + 1e-300) + 1j * np.pi * (vec < 0))
        keys = jnp.asarray((states > 0) @ (2 ** np.arange(16)))
        order = jnp.argsort(keys)

        def log_psi(s):
            code = (s > 0).astype(jnp.int64) @ (2 ** jnp.arange(16))
            return table[order[jnp.searchsorted(keys[order], code)]]

        local = s2_local(log_psi, states.astype(np.int8), chunk=4096)
        p_ = np.abs(vec) ** 2
        exact = vec @ s2 @ vec
        print(f"{name}: estimator {np.sum(p_ * local):.10f}  exact {exact:.10f}  (E0/N {e[0] / 64:.6f})")
        assert abs(np.sum(p_ * local) - exact) < 1e-8


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path, nargs="?")
    p.add_argument("samples", nargs="*")
    p.add_argument("--limit", type=int, default=2048)
    p.add_argument("--block", type=int, default=64, help="samples per block")
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest()
        raise SystemExit
    jax.config.update("jax_default_matmul_precision", "float32")
    from wsp import vit
    from wsp.transfer import Settings, transfer
    STARTS = {"ours": Settings(fill="uniform", fill_heads="nonlocal", local=0.7, window=True, window_avg_units=(3,),
                               sign_rule=True)}  # as in scripts/finetune.py
    r = json.load(open(args.report))
    saved = np.load(args.report.with_name(args.report.stem + "_params.npz"))
    leaves, tree = jax.tree_util.tree_flatten_with_path(transfer(vit.load_params("main"), r["l"], STARTS[r["start"]]))
    params = jax.tree_util.tree_unflatten(tree, [jnp.asarray(saved["/".join(str(k.key) for k in path)])
                                                 for path, _ in leaves])
    data = [np.load(f) for f in args.samples]
    samples = np.concatenate([d["samples"].reshape(-1, r["l"] ** 2) for d in data]).astype(np.int8)
    logw = np.concatenate([d["logw"][-1] if d["logw"].ndim == 2 else d["logw"] for d in data])
    sel = np.random.default_rng(0).permutation(len(samples))[:args.limit]
    samples, w = samples[sel], np.exp(logw[sel])
    log_psi = lambda s: vit.log_psi({"params": params}, s)
    t0, local = time.time(), []
    for start in range(0, len(samples), args.block):
        local.append(s2_local(log_psi, samples[start:start + args.block], args.chunk))
        print(f"  {start + len(local[-1])}/{len(samples)}  running <S2> "
              f"{np.sum(w[:start + len(local[-1])] * np.concatenate(local)) / w[:start + len(local[-1])].sum():.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    local = np.concatenate(local)
    mean = lambda idx: float((w[idx] * local[idx]).sum() / w[idx].sum())
    rng = np.random.default_rng(1)
    boot = [mean(rng.integers(0, len(w), len(w))) for _ in range(1000)]
    out = {"report": str(args.report), "l": r["l"], "samples": len(samples), "S2": mean(np.arange(len(w))),
           "S2_se": float(np.std(boot)), "S2_loc_std": float(local.std()), "seconds": time.time() - t0}
    print(json.dumps(out), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1))
