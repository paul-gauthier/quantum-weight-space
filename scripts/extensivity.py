"""Is the transferred amplitude extensive? Per-site log|psi| penalty for disorder, and the sign rule, by size.

The global head ends in LayerNorms (norm2, norm3), so its log|psi| is bounded whatever the lattice size,
while the number of disordered configurations grows like exp(N). The window head (Settings.window) applies
the untouched 10x10 head to every 5x5-token window and sums log|psi| / 25, which is extensive.

For each L, on resized released samples ("ordered"), the same with a fraction f of sites scrambled by random
swaps, and random Sz = 0 configs: (mean log|psi| ordered - mean log|psi| disordered) / N. An extensive
amplitude keeps it roughly constant in L. Then the share of ordered samples whose phase is within 0.3 rad of
pi*M_A + pi*[b3 < pi/2] + c, for the global phase and for the window head's own phase (sum of window phases
/ 25, not used by the window head: window phases are not multiples of pi).
Settings: the frozen 12x12 transfer (uniform fill on nonlocal heads, cut 0.7, de-alias 0.1, unit 3 gain 1).

usage: PYTHONPATH=. python -u scripts/extensivity.py [N_CONFIGS]
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from wsp import vit
from wsp.transfer import Settings, pooled_mean, resize_samples, transfer

P, S = vit.load_params("main"), vit.load_samples("main")
zbar = pooled_mean(P, S[:1024])
rng = np.random.default_rng(0)
n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 256


def scramble(x, f):
    x = x.copy()
    for row in x:
        for _ in range(int(f * x.shape[1] / 2)):
            i, j = rng.integers(x.shape[1], size=2)
            row[i], row[j] = row[j], row[i]
    return x


def window_phase(x):
    """Im of the 10x10 head on every window, summed / 25 (a read only)."""
    h = P["output"]
    n, t, d = x.shape
    lp = int(np.sqrt(t))
    g = x.reshape(n, lp, lp, d)
    g = sum(jnp.roll(g, a, axis=1) for a in range(-2, 3))
    g = sum(jnp.roll(g, a, axis=2) for a in range(-2, 3))
    z = vit.layer_norm(g.reshape(n, t, d), h["out_layer_norm"])
    amp = vit.layer_norm(vit.dense(z, h["output_layer0"]), h["norm2"])
    sgn = vit.layer_norm(vit.dense(z, h["output_layer1"]), h["norm3"])
    return jnp.sum(vit.log_cosh(amp + 1j * sgn).imag, axis=(1, 2)) / 25


def agree(phi, target):
    r = np.angle(np.exp(1j * (phi - target)))
    c = np.angle(np.exp(1j * r).mean())
    return (np.abs(np.angle(np.exp(1j * (r - c)))) < 0.3).mean()


for L in (10, 12, 14, 16, 18, 20):
    lp, k = L // 2, (L // 2) ** 2 / 25
    s = Settings(kappa=k, fill="uniform", fill_heads="nonlocal", local=0.7, rest_fill="dealias", dealias=0.1,
                 rest_heads="all", kappa_unit3=1.0 if L > 10 else None)
    if L == 10:  # the transfer rules don't apply; the window head is the released head
        s = Settings()
    Q = transfer(P, L, s, zbar)
    Qw = transfer(P, L, Settings(**{**vars(s), "window": True}), zbar)
    order = resize_samples(S[:4 * n_cfg], L)[:n_cfg]
    rand = np.array([rng.permutation(np.repeat([1, -1], L * L // 2)) for _ in range(n_cfg)])
    sets = {"ordered": order, "f0.05": scramble(order, 0.05), "f0.2": scramble(order, 0.2), "random": rand}
    tok = jax.jit(lambda x: vit.tokens(Q, x))
    heads = {"global": jax.jit(lambda x: vit.head(Q, x)), "window": jax.jit(lambda x: vit.head(Qw, x))}
    lps = {name: {h: [] for h in heads} for name in sets}
    for name, cfg in sets.items():
        for i in range(0, len(cfg), 128):
            x = tok(jnp.asarray(cfg[i:i + 128]))
            for h, f in heads.items():
                lps[name][h].append(np.asarray(f(x)))
    lps = {name: {h: np.concatenate(v) for h, v in d.items()} for name, d in lps.items()}
    N = L * L
    line = [f"L={L:2d} n={len(order)}"]
    for h in heads:
        o = lps["ordered"][h].real
        pens = " ".join(f"{name} {(o.mean() - lps[name][h].real.mean()) / N:+.4f}" for name in ("f0.05", "f0.2", "random"))
        line.append(f"{h}: penalty/site {pens}")
    x = tok(jnp.asarray(order))
    hq = Q["output"]
    z = vit.layer_norm(x.sum(1), hq["out_layer_norm"])
    b3 = np.asarray(vit.layer_norm(vit.dense(z, hq["output_layer1"]), hq["norm3"])[:, 3])
    A = ((np.arange(L)[:, None] + np.arange(L)[None]) % 2 == 0).reshape(-1)
    target = np.pi * (order[:, A].astype(int).sum(1) // 2) + np.pi * (b3 < np.pi / 2)
    line.append(f"sign rule: global phase {agree(lps['ordered']['global'].imag, target):.0%}, "
                f"window phase {agree(np.asarray(window_phase(x)), target):.0%}")
    assert np.allclose(lps["ordered"]["global"].imag, lps["ordered"]["window"].imag)  # the window head keeps the phase
    print(" | ".join(line), flush=True)
