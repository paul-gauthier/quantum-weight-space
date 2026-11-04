"""Local energies and energy estimates, evaluating only nonzero connected configs."""

import jax
import jax.numpy as jnp
import numpy as np

from .system import for_samples

BATCH = 4096


def batched(fn, batch=BATCH):
    """Apply a jitted (variables, spins) -> logpsi in fixed-size padded batches."""
    jfn = jax.jit(fn)

    def run(variables, spins):
        spins = np.asarray(spins)
        out = np.empty(spins.shape[0], dtype=np.complex128)
        for i in range(0, spins.shape[0], batch):
            chunk = spins[i : i + batch]
            n = chunk.shape[0]
            if n < batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], batch - n, axis=0)])
            out[i : i + n] = np.asarray(jfn(variables, jnp.asarray(chunk)))[:n]
        return out

    return run


def local_energies(logpsi, variables, samples, logpsi_samples=None):
    """E_loc(s) = sum_s' H_ss' psi(s')/psi(s), in the S = sigma/2, per-site convention."""
    samples = np.asarray(samples)
    system = for_samples(samples)
    conn, mels = system.hamiltonian.get_conn_padded(jnp.asarray(samples))
    conn, mels = np.asarray(conn), np.asarray(mels)
    if logpsi_samples is None:
        logpsi_samples = logpsi(variables, samples)
    diag = (conn == samples[:, None, :]).all(-1)
    off = (mels != 0) & ~diag
    eloc = (mels * diag).sum(1).astype(np.complex128)
    rows, cols = np.nonzero(off)
    lp = logpsi(variables, conn[rows, cols])
    np.add.at(eloc, rows, mels[rows, cols] * np.exp(lp - logpsi_samples[rows]))
    return eloc * system.PER_SITE, logpsi_samples


def summarize(eloc, weights=None):
    """Mean, naive standard error, and variance per site of local energies."""
    e = np.asarray(eloc)
    if weights is None:
        w = np.ones(len(e))
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mean = (w * e).sum()
    var = (w * np.abs(e - mean) ** 2).sum()
    ess = 1.0 / (w**2).sum()
    return {"mean": mean.real, "imag": mean.imag, "err": np.sqrt(var / ess), "var": var, "ess": ess}
