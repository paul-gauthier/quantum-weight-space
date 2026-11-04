"""Fast local energies for GPU runs: any float dtype, with neighbours built on the device.

Only (sample, bond) indices cross the host boundary. Each sample's exchange neighbours are
built on the device, run through the network in large fixed-size batches, and summed per
sample there.

For copies psi_h(s) = psi(flip_h * s[perm_h]) of a base network, `copy_terms` returns, per
sample s,
    B_h(s) = psi_h(s) / psi_ref(s),   A_h(s) = (H psi_h)(s) / psi_ref(s),
where psi_ref is the state the samples were drawn from. For any weights c (a subset of
copies has c_h in {0, 1}), the state psi_c = sum_h c_h psi_h then has
    E[psi_c] = sum_s conj(B c)(A c) / sum_s |B c|^2,
so one pass prices every subset of copies with no further forward passes.

The lattice size is read from the samples; L, N, PER_SITE and BONDS below are the released 10x10.
"""

import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import vit
from .energy import summarize

# JAX runs float32 matmuls in TF32 on NVIDIA GPUs unless told otherwise.
jax.config.update("jax_default_matmul_precision", "float32")

L = vit.L
N = L * L
J2 = 0.5
PER_SITE = 1.0 / (4 * N)  # NetKet's Pauli convention -> E/N with S = sigma/2
CHUNK = 65536


def cast(params, dtype):
    return jax.tree.map(lambda x: x.astype(dtype), params)


@functools.lru_cache
def bonds(l=L, j2=J2):
    """J1 then J2 bonds of the periodic l x l square lattice, as (i, j, J) arrays."""
    idx = np.arange(l * l).reshape(l, l)
    ii, jj, js = [], [], []
    for (dr, dc), j in [((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), j2), ((1, -1), j2)]:
        ii.append(idx.reshape(-1))
        jj.append(np.roll(np.roll(idx, -dr, 0), -dc, 1).reshape(-1))
        js.append(np.full(l * l, j))
    return np.concatenate(ii).astype(np.int32), np.concatenate(jj).astype(np.int32), np.concatenate(js)


BONDS = bonds()


def neighbours(samples, bond=None):
    """Exchange neighbours as (rows, i, j, mel), plus the diagonal, in NetKet's Pauli units."""
    bi, bj, bJ = bonds(math.isqrt(samples.shape[1])) if bond is None else bond
    sisj = samples[:, bi].astype(np.int32) * samples[:, bj]
    rows, cols = np.nonzero(sisj < 0)
    return rows.astype(np.int32), bi[cols], bj[cols], 2.0 * bJ[cols], (sisj * bJ).sum(1)


def released_copies(revision):
    """(perms, flips) of the copies a released revision sums over; copy 0 is the identity."""
    if revision == "main":
        perms = np.arange(N)[None]
    elif revision == "symm_t":
        perms = vit.SHIFTS
    elif revision == "symm_trxy_ising":
        perms = np.loadtxt(vit.CKPT / revision / "symm_indexes.txt", dtype=np.int32)
        return np.concatenate([perms, perms]), np.repeat(np.array([1, -1], np.int8), len(perms))
    else:
        raise ValueError(revision)
    return perms.astype(np.int32), np.ones(len(perms), np.int8)


def log_psi_copies(perms, flips):
    """NetKet-style apply_fun for log sum_h psi(flip_h * s[perm_h])."""
    perms, flips = jnp.asarray(perms), jnp.asarray(flips)

    def apply(variables, spins):
        spins = jnp.atleast_2d(spins)
        out = jax.lax.map(lambda pf: vit.log_psi(variables, spins[..., pf[0]] * pf[1]), (perms, flips))
        m = out.real.max(0)
        return m + jnp.log(jnp.exp(out - m).sum(0))

    return apply


@jax.jit
def _copy_logpsi(params, spins, perm, flip):
    return vit.log_psi({"params": params}, spins[..., perm] * flip)


@jax.jit
def _copy_diagnostics(params, spins, perm, flip):
    s = spins[..., perm] * flip
    x = vit.tokens(params, s)
    out = params["output"]
    z = vit.layer_norm(x.sum(axis=1), out["out_layer_norm"])
    a = vit.layer_norm(vit.dense(z, out["output_layer0"]), out["norm2"])
    b = vit.layer_norm(vit.dense(z, out["output_layer1"]), out["norm3"])
    if "window" in out and "avg" in out["window"]:  # units read from window-averaged inputs (unit 3)
        w = out["window"]
        aw, bw = vit.window_inputs(w, x)
        a, b = a + w["avg"] * (aw.mean(1) - a), b + w["avg"] * (bw.mean(1) - b)
    return vit.head(params, x, s), a[:, 3], b[:, 3]


@jax.jit
def _neighbour_sum(params, samples, rows, bi, bj, mel, perm, flip, ref):
    """sum over a block of neighbours s' of mel * psi_h(s') / psi_ref(s), per sample s."""
    s = samples[rows]
    ar = jnp.arange(s.shape[0])
    s = s.at[ar, bi].multiply(-1).at[ar, bj].multiply(-1)
    lp = vit.log_psi({"params": params}, s[..., perm] * flip)
    val = mel * jnp.exp(lp - ref[rows])
    return jax.ops.segment_sum(val, rows, num_segments=ref.shape[0], indices_are_sorted=True)


def _blocks(m, size):
    """Fixed-size index blocks covering range(m), padded with the last valid index."""
    for start in range(0, m, size):
        idx = np.arange(start, start + size)
        n = min(size, m - start)
        idx[n:] = start + n - 1
        yield idx, n


def copy_terms(params, samples, perms, flips, ref=None, chunk=CHUNK, verbose=False, diagnostics=False):
    """Per-copy terms A, B (copies x samples) and each copy's log psi on the samples.

    ref: log amplitude of the sampled state. None means the samples come from the sum of the
    given copies; "first" means they come from copy 0 alone; or pass an array.
    """
    samples = np.asarray(samples, dtype=np.int8)
    perms, flips = np.asarray(perms, np.int32), np.asarray(flips, np.int8)
    n, t0 = len(samples), time.time()
    s_dev = jnp.asarray(samples)

    size = min(chunk, n)
    lp = np.empty((len(perms), n), np.complex128)
    a3 = np.empty(lp.shape) if diagnostics else None
    b3 = np.empty(lp.shape) if diagnostics else None
    for g, (perm, flip) in enumerate(zip(perms, flips)):
        for idx, k in _blocks(n, size):
            if diagnostics:
                l, a, b = _copy_diagnostics(params, s_dev[idx], perm, flip)
                lp[g, idx[:k]], a3[g, idx[:k]], b3[g, idx[:k]] = (
                    np.asarray(value)[:k] for value in (l, a, b))
            else:
                lp[g, idx[:k]] = np.asarray(_copy_logpsi(params, s_dev[idx], perm, flip))[:k]
    if ref is None:
        m = lp.real.max(0)
        ref = m + np.log(np.exp(lp - m).sum(0))
    elif isinstance(ref, str) and ref == "first":
        ref = lp[0]
    B = np.exp(lp - ref)

    rows, bi, bj, mel, diag = neighbours(samples)
    size = min(chunk, -(-len(rows) // 4096) * 4096)  # small runs: don't pad to a full chunk
    blocks = [
        tuple(jnp.asarray(a) for a in (rows[idx], bi[idx], bj[idx], mel[idx] * (np.arange(size) < k)))
        for idx, k in _blocks(len(rows), size)
    ]
    ref_dev = jnp.asarray(ref)
    A = np.empty_like(B)
    for g, (perm, flip) in enumerate(zip(perms, flips)):
        acc = sum(_neighbour_sum(params, s_dev, *blk, perm, flip, ref_dev) for blk in blocks)
        A[g] = np.asarray(acc) + diag * B[g]
        if verbose:
            print(f"  copy {g + 1}/{len(perms)}  {time.time() - t0:.0f}s", flush=True)
    passes = len(perms) * (n + len(rows))
    result = {"A": A, "B": B, "lp": lp, "passes": passes, "seconds": time.time() - t0,
              "per_site": 1.0 / (4 * samples.shape[1])}
    if diagnostics:
        result.update(a3=a3, b3=b3)
    return result


def energy(terms, c=None, idx=slice(None)):
    """E/N and IID ratio-estimator uncertainty; no division by individual amplitudes."""
    A, B = terms["A"][:, idx], terms["B"][:, idx]
    per_site = terms.get("per_site", PER_SITE)  # caches saved before this key are 10x10
    c = np.ones(len(A)) if c is None else np.asarray(c, dtype=np.float64)
    a, b = c @ A, c @ B
    num, den = np.conj(b) * a * per_site, np.abs(b) ** 2
    mean = num.sum() / den.sum()
    influence = (num.real - mean.real * den) / den.mean()
    var = (np.abs(a * per_site - mean * b) ** 2).sum() / den.sum()
    return {"mean": mean.real, "imag": mean.imag,
            "err": influence.std(ddof=1) / np.sqrt(len(den)),
            "var": var, "ess": den.sum() ** 2 / (den ** 2).sum()}
