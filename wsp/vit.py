"""Pure-JAX ViT neural quantum state matching nqs-models/j1j2_square_10x10_05.

Re-implements the released Flax code (transformer.py / attentions.py, Apache-2.0,
Rende & Viteritti) as plain functions of a nested parameter dict, so weights can be
read and edited directly.

The lattice size is read from the input (l*l spins) and from J (lp*lp entries per head),
so the same functions run a network written for another l (see wsp/transfer.py).
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "hf"

L = 10  # lattice side
B = 2  # patch side
LP = L // B  # patches per side
N_LAYERS = 8
D = 72
H = 12
LN_EPS = 1e-6


def load_params(revision="main"):
    flat = load_file(str(CKPT / revision / "model.safetensors"))
    tree = {}
    for key, val in flat.items():
        parts = key.split(".")
        assert parts[0] == "params"
        node = tree
        for p in parts[1:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(val)
    return tree


def load_samples(revision="main"):
    from flax.serialization import msgpack_restore

    raw = (CKPT / revision / "spins").read_bytes()
    return np.asarray(msgpack_restore(raw), dtype=np.int8)


def layer_norm(x, p):
    mean = x.mean(-1, keepdims=True)
    var = jnp.maximum((x * x).mean(-1, keepdims=True) - mean * mean, 0.0)
    return (x - mean) * jax.lax.rsqrt(var + LN_EPS) * p["scale"] + p["bias"]


def dense(x, p):
    return x @ p["kernel"] + p["bias"]


def log_cosh(x):
    sgn = -2 * jnp.signbit(x.real) + 1
    x = x * sgn
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def extract_patches(x):
    """(batch, l*l) -> (batch, lp*lp, 4); patch index = lp*row + col, in-patch = 2*dr + dc."""
    b = x.shape[0]
    lp = math.isqrt(x.shape[-1]) // B
    x = x.reshape(b, lp, B, lp, B).transpose(0, 1, 3, 2, 4)
    return x.reshape(b, lp * lp, B * B)


def full_attention(J):
    """(h, lp*lp) relative-position weights -> (h, lp*lp, lp*lp) with A[p, q] = J[q - p]."""
    lp = math.isqrt(J.shape[-1])
    j = J.reshape(J.shape[0], lp, lp)
    rows = []
    for di in range(lp):
        for dj in range(lp):
            rows.append(jnp.roll(jnp.roll(j, di, axis=-2), dj, axis=-1).reshape(J.shape[0], -1))
    return jnp.stack(rows, axis=1)


def attention(x, p):
    v = dense(x, p["v"])  # (b, tokens, 72)
    b, n, _ = v.shape
    v = v.reshape(b, n, H, D // H).transpose(0, 2, 1, 3)  # (b, h, tokens, 6)
    x = jnp.einsum("hpq,bhqd->bhpd", full_attention(p["J"]), v)
    x = x.transpose(0, 2, 1, 3).reshape(b, n, D)
    return dense(x, p["W"])


def encoder_block(x, p):
    x = x + attention(layer_norm(x, p["layer_norm_1"]), p["attn"])
    h = layer_norm(x, p["layer_norm_2"])
    h = jax.nn.gelu(dense(h, p["ff"]["layers_0"]), approximate=True)
    return x + dense(h, p["ff"]["layers_2"])


def tokens(params, spins):
    """Embedded tokens after the full encoder, (batch, tokens, 72), in the params' dtype."""
    x = jnp.atleast_2d(spins).astype(params["patches_and_embed"]["embed"]["kernel"].dtype)
    x = dense(extract_patches(x), params["patches_and_embed"]["embed"])
    for i in range(N_LAYERS):
        x = encoder_block(x, params["encoder"][f"layers_{i}"])
    return x


def head(params, x, spins=None):
    p = params["output"]
    z = layer_norm(x.sum(axis=1), p["out_layer_norm"])
    amp = layer_norm(dense(z, p["output_layer0"]), p["norm2"])
    sign = layer_norm(dense(z, p["output_layer1"]), p["norm3"])
    terms = log_cosh(amp + 1j * sign)
    b = sign  # sign inputs the sign rule reads (window-averaged for windowed and "avg" units, below)
    if "window" not in p:
        out = jnp.sum(terms, axis=-1)
    else:
        # wsp/transfer.py's window head. log|psi| of the masked units: summed over every LP x LP window of
        # tokens and divided by LP^2 (each token is in LP^2 windows), so it stays extensive. The phase, and
        # log|psi| of the other units, stay global, except the "avg" units: their whole term is read from the
        # window-averaged inputs. On an LP x LP torus every window is the whole lattice: the released head.
        w = p["window"]
        amp_w, sign_w = window_inputs(w, x)
        out = (jnp.sum((1 - w["mask"]) * terms.real, axis=-1) + 1j * jnp.sum(terms.imag, axis=-1)
               + w.get("scale", 1.0) * jnp.sum(w["mask"] * log_cosh(amp_w + 1j * sign_w).real, axis=(1, 2)) / LP ** 2)
        if "avg" in w:
            out = out + jnp.sum(w["avg"] * (log_cosh(amp_w.mean(1) + 1j * sign_w.mean(1)) - terms), axis=-1)
        b = b + (w["mask"] + w.get("avg", 0.0)) * (sign_w.mean(1) - b)
    if "sign_rule" in p:
        # wsp/transfer.py's sign rule: on 10x10 the released phase is pi * (M_A + flips of the sign units) + const
        # to ~1 mrad, so write that rule instead of the head's phase, which drifts on larger lattices.
        assert spins is not None, "the sign rule reads the spins"
        out = out.real + 1j * sign_rule_phase(spins, b, p["sign_rule"])
    return out


def sign_rule_phase(spins, b, units):
    """pi * (M_A + sum over the sign units u of [cos b_u < 0]) mod 2 pi. M_A is half the A-sublattice spin sum (an
    integer at S^z = 0); units is a 0/1 mask of the units whose amplitude input is ~0, so that b_u crossing pi/2
    flips the sign (unit 3; also unit 6 in trxy's base net). Modulo 2 pi, pi M_A is unchanged by translations,
    point-group operations and spin flip, so copies of a symmetrized state keep the base net's relative signs."""
    l = math.isqrt(spins.shape[-1])
    on_a = ((np.arange(l)[:, None] + np.arange(l)) % 2 == 0).reshape(-1)
    m_a = jnp.round(jnp.sum(jnp.where(on_a, jnp.asarray(spins, b.dtype), 0), axis=-1) / 2)
    flips = jnp.sum(units * (jnp.cos(b) < 0), axis=-1)
    return jnp.pi * ((m_a + flips) % 2)


def window_inputs(w, x):
    """Inputs (amp, sign) of head w on the pooled sum of every LP x LP window of tokens, (batch, tokens, D)."""
    n, t, d = x.shape
    lp = math.isqrt(t)
    g = x.reshape(n, lp, lp, d)
    r = range(-(LP // 2), LP - LP // 2)
    g = sum(jnp.roll(g, a, axis=1) for a in r)
    g = sum(jnp.roll(g, a, axis=2) for a in r)
    z = layer_norm(g.reshape(n, t, d), w["out_layer_norm"])
    return (layer_norm(dense(z, w["output_layer0"]), w["norm2"]),
            layer_norm(dense(z, w["output_layer1"]), w["norm3"]))


def log_psi(variables, spins):
    """NetKet-compatible apply_fun: variables = {'params': tree}."""
    params = variables["params"]
    return head(params, tokens(params, spins), jnp.atleast_2d(spins))


def shift_indexes(l=L):
    """The 4 in-patch translations (0,0), (0,1), (1,0), (1,1), as gather indexes."""
    idx = np.arange(l * l).reshape(l, l)
    out = [np.roll(np.roll(idx, dr, axis=0), dc, axis=1).reshape(-1) for dr in (0, 1) for dc in (0, 1)]
    return np.stack(out)


SHIFTS = shift_indexes()


def log_psi_symm(variables, spins):
    """Translation-symmetrized wavefunction: log sum_t psi(T_t sigma), as in symm_t."""
    spins = jnp.atleast_2d(spins)
    outs = jnp.stack([log_psi(variables, spins[..., idx]) for idx in SHIFTS])
    ref = outs[0]
    return ref + jnp.log(jnp.sum(jnp.exp(outs - ref), axis=0))
