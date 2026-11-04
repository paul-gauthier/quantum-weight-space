"""One Lanczos step, psi_1 = (1 + alpha (H - c)) psi, priced on a saved snapshot with no new sampling.

A snapshot (scripts/ft_lanczos.py) holds one sample per independent chain, drawn from |psi|^2.
Per sample s, with H' = H - c in NetKet's Pauli units,
    x(s) = (H' psi)(s) / psi(s),    y(s) = (H'^2 psi)(s) / psi(s).
Reweighting to psi_alpha = (1 + alpha H') psi gives
    E(alpha) = mean conj(1 + alpha x)(x + alpha y) / mean |1 + alpha x|^2,
    <H'^2>(alpha) = mean |x + alpha y|^2 / mean |1 + alpha x|^2,
and the best alpha is the lowest generalized eigenvector of the 2 x 2 Krylov pencil (`pencil`).
c (the snapshot's mean energy) only keeps the numbers well conditioned.

y needs psi on every configuration up to two exchanges away:
    (H'^2 psi)(s) = d'(s) (H' psi)(s) + sum_b m_b d'(s_b) psi(s_b) + sum_{b, b'} m_b m_b' psi(s_bb'),
with d' = diagonal - c and m = 2 J for an antialigned bond. Two disjoint exchanges commute, so each
such pair is evaluated once with weight 2; pairs sharing a site (b' = b returns to s) are evaluated
as they come. That is ~n^2 / 2 configurations per sample for n ~ 2.26 N exchanges.

Used as a library by scripts/ft_lanczos.py and scripts/pool_lanczos.py.

usage: PYTHONPATH=. python -u scripts/lanczos.py --check       (exact 4x4 test: sparse H applied twice)
"""
import argparse
import math
import time

import numpy as np


def site_bonds(l, bond):
    """(N, 8) bond indices at each site: 4 J1 and 4 J2 bonds."""
    bi, bj, _ = bond
    inc = [[] for _ in range(l * l)]
    for b, (i, j) in enumerate(zip(bi, bj)):
        inc[i].append(b)
        inc[j].append(b)
    return np.array(inc, dtype=np.int32)


def configurations(samples, c, bond, inc):
    """Configurations behind x and y for a block of samples, as flips of four sites (a site flipped twice is
    unchanged) with weights (w_x, w_y), plus the diagonal d of each sample. Rows are sorted by sample."""
    bi, bj, bJ = bond
    out, diag = [], np.empty(len(samples))
    for r, s in enumerate(samples.astype(np.int32)):
        sisj = s[bi] * s[bj]
        a = np.nonzero(sisj < 0)[0]
        n, m = len(a), 2.0 * bJ[a]
        diag[r] = (sisj * bJ).sum()
        # bonds touching exactly one site of each exchange (14 of them; the exchange itself appears twice)
        touch = np.concatenate([inc[bi[a]], inc[bj[a]]], axis=1)
        touch = touch[touch != a[:, None]].reshape(n, 14)
        d_b = diag[r] - 2 * (sisj[touch] * bJ[touch]).sum(1)  # their sign flips under the exchange
        # one exchange (second pair (i, i) cancels): weights m for x, m d' for y
        parts = [(bi[a], bj[a], bi[a], bi[a], m, m * (d_b - c))]
        # exchange b, then b' touching it and antialigned afterwards, i.e. aligned before
        k = (sisj[touch] > 0).ravel()
        b1, b2 = np.repeat(a, 14)[k], touch.ravel()[k]
        parts.append((bi[b1], bj[b1], bi[b2], bj[b2], 0 * b1, 2.0 * bJ[b1] * 2.0 * bJ[b2]))
        # exchange b twice: back to s
        parts.append((bi[a], bj[a], bi[a], bj[a], 0 * m, m * m))
        # two disjoint exchanges, once each, weight 2 m m'
        p, q = np.triu_indices(n, 1)
        ip, jp, iq, jq = bi[a[p]], bj[a[p]], bi[a[q]], bj[a[q]]
        ok = (ip != iq) & (ip != jq) & (jp != iq) & (jp != jq)
        p, q = p[ok], q[ok]
        parts.append((ip[ok], jp[ok], iq[ok], jq[ok], 0 * m[p], 2.0 * m[p] * m[q]))
        for i1, j1, i2, j2, wx, wy in parts:
            out.append((np.full(len(i1), r, np.int32), i1, j1, i2, j2, wx, wy))
    rows, i1, j1, i2, j2, wx, wy = (np.concatenate(v) for v in zip(*out))
    return rows, i1, j1, i2, j2, np.stack([wx, wy], 1).astype(np.float64), diag


def evaluate(params, samples, perms, flips, c, chunk, block, verbose=True, done=None):
    """x, y, the diagonal and log psi for each sample, in blocks of samples (done(k, partial) is called after
    each block)."""
    import jax
    import jax.numpy as jnp
    from wsp import fast, vit

    @jax.jit
    def sums(params, s_dev, rows, i1, j1, i2, j2, w, perm, flip, ref):
        s = s_dev[rows]
        ar = jnp.arange(s.shape[0])
        for k in (i1, j1, i2, j2):
            s = s.at[ar, k].multiply(-1)
        lp = vit.log_psi({"params": params}, s[..., perm] * flip)
        val = w * jnp.exp(lp - ref[rows])[:, None]
        return jax.ops.segment_sum(val, rows, num_segments=ref.shape[0], indices_are_sorted=True)

    l = math.isqrt(samples.shape[1])
    bond = fast.bonds(l)
    inc = site_bonds(l, bond)
    n = len(samples)
    x, y = np.empty(n, np.complex128), np.empty(n, np.complex128)
    diag, ref_all = np.empty(n), np.empty(n, np.complex128)
    t0, configs = time.time(), 0
    for start in range(0, n, block):
        idx = np.arange(start, min(start + block, n))
        s = samples[idx]
        if len(idx) < block:  # fixed block shape: pad with the last sample, drop its rows below
            s = np.concatenate([s, np.repeat(s[-1:], block - len(idx), 0)])
        s_dev = jnp.asarray(s)
        lp = np.stack([np.asarray(fast._copy_logpsi(params, s_dev, p, f)) for p, f in zip(perms, flips)])
        mx = lp.real.max(0)
        ref = mx + np.log(np.exp(lp - mx).sum(0))
        rows, i1, j1, i2, j2, w, diag_b = configurations(s, c, bond, inc)
        size = min(chunk, -(-len(rows) // 4096) * 4096)
        chunks = []
        for start_c in range(0, len(rows), size):
            k = min(size, len(rows) - start_c)
            sel = np.arange(start_c, start_c + size)
            sel[k:] = start_c + k - 1
            wk = w[sel] * (np.arange(size) < k)[:, None]
            chunks.append(tuple(jnp.asarray(v) for v in (rows[sel], i1[sel], j1[sel], i2[sel], j2[sel], wk)))
        ref_dev = jnp.asarray(ref)
        acc = np.zeros((block, 2), np.complex128)
        for p, f in zip(perms, flips):
            for ch in chunks:
                acc += np.asarray(sums(params, s_dev, *ch, p, f, ref_dev))
        e0 = diag_b + acc[:, 0]
        xb = e0 - c
        yb = (diag_b - c) * xb + acc[:, 1]
        keep = slice(0, len(idx))
        x[idx], y[idx], diag[idx], ref_all[idx] = xb[keep], yb[keep], diag_b[keep], ref[keep]
        configs += len(rows) * len(idx) // block
        if verbose:
            el = time.time() - t0
            print(f"  {idx[-1] + 1}/{n} samples  {len(rows) / block:.0f} configs/sample  {el:.0f}s "
                  f"({el / (idx[-1] + 1):.2f} s/sample)", flush=True)
        if done is not None:
            done(idx[-1] + 1, dict(x=x[:idx[-1] + 1], y=y[:idx[-1] + 1], diag=diag[:idx[-1] + 1],
                                   logpsi=ref_all[:idx[-1] + 1], seconds=time.time() - t0, configs=configs))
    return x, y, diag, ref_all


# ---------------------------------------------------------------- analysis (numpy only)

def pencil(x, y, wt=None):
    """Krylov {psi, H' psi} matrices (H', overlap), estimated by reweighting from |psi|^2 (weights wt)."""
    wt = np.full(len(x), 1 / len(x)) if wt is None else wt / wt.sum()
    b, q = wt @ x.real, wt @ np.abs(x) ** 2
    a1 = wt @ y.real + q
    a2 = wt @ (np.conj(x) * y).real
    return np.array([[b, a1 / 2], [a1 / 2, a2]]), np.array([[1.0, b], [b, q]])


def best_alpha(x, y, wt=None):
    h, s = pencil(x, y, wt)
    vals, vecs = np.linalg.eig(np.linalg.solve(s, h))
    v = vecs[:, np.argmin(vals.real)].real
    return v[1] / v[0]


def at_alpha(x, y, alpha, wt=None):
    """E' and <H'^2> - E'^2 of (1 + alpha H') psi."""
    wt = np.full(len(x), 1 / len(x)) if wt is None else wt / wt.sum()
    den = wt @ np.abs(1 + alpha * x) ** 2
    e = wt @ (np.conj(1 + alpha * x) * (x + alpha * y)).real / den
    return e, wt @ np.abs(x + alpha * y) ** 2 / den - e ** 2


def moments(x, y):
    """Per-sample terms whose means fix the pencil and the variance of psi_alpha:
    Re x, |x|^2, Re y, Re conj(x) y, |y|^2."""
    return np.stack([x.real, np.abs(x) ** 2, y.real, (np.conj(x) * y).real, np.abs(y) ** 2], 1)


def at_moments(m, a):
    """E' and <H'^2> - E'^2 of (1 + a H') psi from mean moments m."""
    b, q, ry, xy, yy = m
    den = 1 + 2 * a * b + a * a * q
    e = (b + a * (ry + q) + a * a * xy) / den
    return e, (q + 2 * a * xy + a * a * yy) / den - e * e


def best_alpha_m(m):
    b, q, ry, xy, _ = m
    h = np.array([[b, (ry + q) / 2], [(ry + q) / 2, xy]])
    vals, vecs = np.linalg.eig(np.linalg.solve(np.array([[1.0, b], [b, q]]), h))
    v = vecs[:, np.argmin(vals.real)].real
    return v[1] / v[0]


def summary(x, y, c, n_sites, n_boot=2000, seed=0, pairs=None):
    """Per-site E0, E1 (in-sample alpha and two-fold cross-fitted alpha), variances, V-scores, bootstrap errors.
    pairs = (m32, m64): per-sample moments of a random subsample evaluated in both precisions; their mean
    difference corrects every pooled float32 moment (a control variate for the float32 bias of y)."""
    per = 1 / (4 * n_sites)
    rng = np.random.default_rng(seed)
    m_all = moments(x, y)
    diff = None if pairs is None else pairs[1] - pairs[0]

    def point(idx, pidx):
        m = m_all[idx].mean(0) + (0 if diff is None else diff[pidx].mean(0))
        a = best_alpha_m(m)
        e1, v1 = at_moments(m, a)
        e0, v0 = m[0], m[1] - m[0] ** 2
        w = np.abs(1 + a * x[idx]) ** 2
        return np.array([(e0 + c) * per, (e1 + c) * per, (e1 - e0) * per, v0, v1, a,
                         w.sum() ** 2 / (w ** 2).sum() / len(w), m[2] - m[1]])

    n, n_p = len(x), 0 if diff is None else len(diff)
    every, p_every = np.arange(n), np.arange(n_p)
    full = point(every, p_every)
    half = rng.permutation(n)
    cross = []
    for fit, ev in ((half[: n // 2], half[n // 2:]), (half[n // 2:], half[: n // 2])):
        corr = 0 if diff is None else diff.mean(0)
        m_fit, m_ev = m_all[fit].mean(0) + corr, m_all[ev].mean(0) + corr
        e1, _ = at_moments(m_ev, best_alpha_m(m_fit))
        cross.append(((e1 - m_ev[0]) * per, (e1 + c) * per))
    boot = np.array([point(rng.integers(0, n, n), rng.integers(0, n_p, n_p) if n_p else p_every)
                     for _ in range(n_boot)])
    se = boot.std(0, ddof=1)
    e0, e1, de, v0, v1, a, ess, h2gap = full
    vs = lambda v, e: v / (16 * n_sites * e ** 2)
    # two-point variance extrapolation to zero variance (assumes the linear regime)
    ext = e1 - v1 * (e0 - e1) / (v0 - v1) if v0 != v1 else float("nan")
    return {"n": n, "n_fp64_pairs": n_p, "E0": e0, "E0_se": se[0], "E1": e1, "E1_se": se[1], "dE": de,
            "dE_se": se[2], "dE_crossfit": float(np.mean([d for d, _ in cross])),
            "E1_crossfit": float(np.mean([e for _, e in cross])), "var0": v0, "var1": v1, "V0": vs(v0, e0),
            "V1": vs(v1, e1), "alpha": a, "alpha_se": se[5], "ess_fraction": ess, "E_var_extrapolated": ext,
            # mean y and mean |x|^2 both estimate <H'^2>: their difference must vanish within its error
            "H2_check": {"mean_y_minus_abs_x2": h2gap, "se": se[7]},
            "fp64_correction": None if diff is None else dict(zip(
                ["Re x", "|x|^2", "Re y", "Re conj(x) y", "|y|^2"], diff.mean(0).tolist()))}


# ---------------------------------------------------------------- exact 4x4 check

def check():
    """4x4 (12,870 configs at Sz = 0), against sparse H applied twice: (1) trxy's 64 copies with the sign rule
    (units 3, 6), x and y on 96 random configs; (2) one copy of the same base net, x and y on every config, and
    the Lanczos energy from reweighting with exact |psi|^2 weights against the exact Krylov pencil. No window
    head (it needs >= 10x10); the code under test does not depend on the network's internals."""
    import itertools
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from wsp import fast, native, vit
    from wsp.transfer import Settings, pooled_mean, transfer

    l, n_sites = 4, 16
    P = vit.load_params("symm_trxy_ising")
    zbar = pooled_mean(P, vit.load_samples("symm_trxy_ising")[:256])
    params = transfer(P, l, Settings(kappa=0.16, kappa_unit3=1.0, sign_rule=True, sign_units=(3, 6)), zbar)
    confs = np.full((12870, n_sites), -1, np.int8)
    for k, up in enumerate(itertools.combinations(range(n_sites), n_sites // 2)):
        confs[k, list(up)] = 1
    # sparse H from first-order exchanges only
    code = {c.tobytes(): k for k, c in enumerate(confs)}
    rows, bi, bj, mel, diag = fast.neighbours(confs)
    nb = confs[rows].copy()
    nb[np.arange(len(rows)), bi] *= -1
    nb[np.arange(len(rows)), bj] *= -1
    cols = np.array([code[c.tobytes()] for c in nb])
    hm = sp.csr_matrix((mel, (rows, cols)), shape=(len(confs),) * 2) + sp.diags(diag)
    e_gs = sla.eigsh(hm, k=1, which="SA")[0][0] / (4 * n_sites)
    c = -0.55 * 4 * n_sites

    def exact(perms, flips):
        lp = np.stack([np.asarray(fast._copy_logpsi(params, confs, p, f)) for p, f in zip(perms, flips)])
        mx = lp.real.max(0)
        logpsi = mx + np.log(np.exp(lp - mx).sum(0))
        psi = np.exp(logpsi - logpsi.real.max())
        hp = hm @ psi - c * psi
        return logpsi, psi, hp, hm @ hp - c * hp

    ok = True
    for name, (perms, flips), pick in (
            ("trxy, 64 copies", native.copies("symm_trxy_ising", l),
             np.random.default_rng(0).choice(len(confs), 96, replace=False)),
            ("one copy", native.copies("main", l), np.arange(len(confs)))):
        logpsi, psi, hp, h2p = exact(perms, flips)
        x, y, d, ref = evaluate(params, confs[pick], perms, flips, c, chunk=65536,
                                block=min(len(pick), 1287), verbose=False)
        ex, ey = hp[pick] / psi[pick], h2p[pick] / psi[pick]
        dx, dy = np.abs(x - ex).max() / np.abs(ex).max(), np.abs(y - ey).max() / np.abs(ey).max()
        print(f"{name}: logpsi max |diff| {np.abs(ref - logpsi[pick]).max():.2e}, x rel diff {dx:.2e}, "
              f"y rel diff {dy:.2e}")
        ok &= dx < 1e-10 and dy < 1e-10
        if len(pick) == len(confs):
            nrm = np.vdot(psi, psi).real
            m1, m2, m3 = (np.vdot(psi, hp).real / nrm, np.vdot(hp, hp).real / nrm,
                          np.vdot(hp, hm @ hp - c * hp).real / nrm)
            vals = np.linalg.eigvals(np.linalg.solve(np.array([[1, m1], [m1, m2]]),
                                                     np.array([[m1, m2], [m2, m3]])))
            e1_exact = (vals.real.min() + c) / (4 * n_sites)
            wt = np.abs(psi) ** 2
            e1 = (at_alpha(x, y, best_alpha(x, y, wt), wt)[0] + c) / (4 * n_sites)
            print(f"  E0 {(m1 + c) / (4 * n_sites):.8f}  E1 exact pencil {e1_exact:.8f}  E1 reweighted {e1:.8f}"
                  f"  ground state {e_gs:.8f}")
            ok &= abs(e1 - e1_exact) < 1e-10 and e1 >= e_gs - 1e-12
    assert ok, "check failed"
    print("check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", required=True)
    p.parse_args()
    check()


if __name__ == "__main__":
    main()
