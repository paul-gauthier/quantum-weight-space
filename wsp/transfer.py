"""Write a network for an l x l lattice from the released 10x10 weights, with named settings.

The embedding, FFNs, LayerNorms and output head carry over unchanged. Two parts
need a rule:

- J. Each head mixes tokens with relative-position weights J[q - p] on a torus of 5 x 5
  patches. On an lp x lp torus, every old offset keeps its weight (offsets that land on the same
  entry of a smaller torus are folded together), and offsets the 10x10 net never had are filled.
- Pooling. z = LN(sum of tokens) = LN(mean), so a local move's effect on z shrinks like
  1/N_tokens. kappa scales the head's response to z - zbar to make up for it:
  W' = kappa W, b' = b + (1 - kappa) zbar W, for both output layers.

transfer(params, 10, Settings()) returns the released weights bit for bit.
"""

import json
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import vit


@dataclass(frozen=True)
class Settings:
    kappa: float = 1.0  # head gain on z - zbar; N_tokens,new / N_tokens,old offsets mean pooling
    kappa_amp: float | None = None  # separate gain for the amplitude layer (output_layer0); None = kappa
    kappa_unit3: float | None = None  # separate gain for unit 3's sign input (output_layer1 column 3)
    kappa_sign_units: tuple = ()  # more (unit, gain) pairs for sign inputs, e.g. random-unit controls
    kappa_phase: float | None = None  # gain for the sign layer (output_layer1) apart from overrides; None = kappa
    unit3_shift: float = 0.0  # moves unit 3's sign input (norm3 bias) relative to its node at pi/2
    rotor: float = 0.0  # phase rotor: extra phase (rad) per unit of the phase integer n = kappa r.(z - zbar)
    #                     (wsp/phase_direction.json), written through the sign inputs of rotor_pairs
    rotor_pairs: tuple = ((23, 10),)  # (unit with a >> 0, unit with a << 0): phase ~ +-b and |psi| hardly
    #                                   depends on b; +D on one and -D on the other add phases, keep LN3's mean
    spread: float = 1.0  # with fill="uniform": the fraction of each head's mean J spread over the new torus
    spread_layers: tuple = tuple(range(8))  # layers that use `spread` (the others spread all of it)
    spread_from: int = 0  # offsets nearer than this (Chebyshev) keep their 10x10 J exactly; only the far
    #                       offsets' mean is spread, over all far offsets of the new torus (sum kept)
    fill: str = "zero"  # J at new offsets: "zero"; "ring" = the mean of the head's outer ring;
    #                     "clamp" = the outer-ring entry in the same direction (averaged at ties);
    #                     "uniform" = spread the head's mean J over the whole new torus (sum kept),
    #                     pad or fold the rest. A head's mean part then still sees the global sum.
    fill_heads: str = "diffuse"  # heads that get the ring fill: "diffuse", "nonlocal" (+ mixed), "all"
    rest_fill: str = "zero"  # with fill="uniform": how the non-mean rest is extended ("zero", "clamp", "ring",
    #                          "dealias": move a fraction `dealias` of each ring entry to its distance-3 image)
    dealias: float = 0.3  # on the 5-patch torus offset +2 is offset -3; the share of a ring weight that is the image
    rest_heads: str = "nonlocal"  # ... for which heads
    rest_layers: tuple = tuple(range(8))  # ... in which layers
    keep_mass: bool = True  # rescale filled heads so that sum |J| is unchanged
    fold: str = "sum"  # smaller torus: offsets that coincide are "sum"med or "mean"ed
    force_uniform: tuple = ()  # (layer, head) pairs that get the fill rule whatever their class
    force_zero: tuple = ()  # (layer, head) pairs that are plain zero-padded whatever their class
    window: bool = False  # log|psi| from the untouched 10x10 head on every 5x5-token window (vit.window_amplitude),
    #                       summed / 25: extensive, where the global head's norm2 LayerNorm caps log|psi|.
    #                       The phase stays global (kappa, rotor), since window phases are not multiples of pi
    window_global_units: tuple = ()  # units whose log|psi| term stays global too, e.g. (3,): unit 3's node
    #                                  then stays where its global sign flip is
    window_avg_units: tuple = ()  # units whose whole term (amplitude and phase) is read from the window-averaged
    #                               inputs, e.g. (3,): unit 3's node then sees 10x10-range inputs, not a global
    #                               sign LayerNorm whose other 71 inputs are amplified by kappa
    window_scale: float = 1.0  # temperature on the windowed log|psi| (Go-style named setting; 1 on 10x10)
    sign_rule: bool = False  # phase = pi * (M_A + flips of sign_units) (vit.sign_rule_phase), the rule the released
    #                          phase follows on 10x10, instead of the head's phase; with the window head each unit's
    #                          input is window-averaged, and kappa, kappa_phase and the rotor then no longer matter
    sign_units: tuple = (3,)  # units whose input crossing pi/2 flips the sign: (3,) for main/symm_t, (3, 6) for trxy
    local: float = 0.8  # head class: share of |J| on the same or a neighbouring patch ...
    diffuse: float = 0.5  # ... >= local is "local", <= diffuse is "diffuse", between is "mixed"


def offsets(lp):
    """Signed (dr, dc) patch offset of each J entry on an lp x lp torus (entry lp*dr + dc mod lp)."""
    s = (np.arange(lp) + lp // 2) % lp - lp // 2
    dr, dc = np.meshgrid(s, s, indexing="ij")
    return dr.ravel(), dc.ravel()


def near_share(J):
    """Share of each head's |J| on the same or a neighbouring patch (Chebyshev distance <= 1)."""
    dr, dc = offsets(math.isqrt(J.shape[-1]))
    a = np.abs(np.asarray(J))
    return a[:, np.maximum(abs(dr), abs(dc)) <= 1].sum(-1) / a.sum(-1)


def classify(J, s=Settings()):
    near = near_share(J)
    return np.where(near >= s.local, "local", np.where(near <= s.diffuse, "diffuse", "mixed"))


def transfer_J(J, lp, s=Settings(), layer=None):
    """(h, lp0*lp0) -> (h, lp*lp) relative-position weights."""
    J = np.asarray(J)
    lp0 = math.isqrt(J.shape[-1])
    if s.fill == "uniform":
        f = s.spread if layer is None or layer in s.spread_layers else 1.0
        cheb0 = np.maximum(*map(np.abs, offsets(lp0)))
        cheb1 = np.maximum(*map(np.abs, offsets(lp)))
        far0, far1 = cheb0 >= s.spread_from, cheb1 >= s.spread_from
        if not far1.any() or lp < lp0:  # shrinking: spread from every offset, as before
            far0, far1 = np.ones_like(far0), np.ones_like(far1)
        mean = f * J[:, far0].mean(-1, keepdims=True)  # the far mean, spread over every far offset
        chosen = _chosen(J, s, layer)[:, None]
        base = np.where(chosen, J - mean * far0, J)
        rest = transfer_J(base, lp, Settings(**{**vars(s), "fill": "zero"}))
        rest_fill = s.rest_fill if layer is None or layer in s.rest_layers else "zero"
        if rest_fill != "zero":
            filled = transfer_J(base, lp, Settings(**{**vars(s), "fill": rest_fill, "fill_heads": "all",
                                                       "keep_mass": False}))
            filled -= np.where(chosen, filled.mean(-1, keepdims=True), 0.0)  # keep the mean part exact
            pick = _chosen(J, Settings(**{**vars(s), "fill_heads": s.rest_heads}))[:, None]  # on the full J
            rest = np.where(pick, filled, rest)
        return rest + np.where(chosen, mean * far0.sum() / far1.sum() * far1, 0.0)
    dr0, dc0 = offsets(lp0)
    target = (dr0 % lp) * lp + (dc0 % lp)
    new = np.zeros((J.shape[0], lp * lp), J.dtype)
    count = np.zeros(lp * lp)
    for k, t in enumerate(target):
        new[:, t] += J[:, k]
        count[t] += 1
    if s.fold == "mean":
        new[:, count > 1] /= count[count > 1]
    empty = count == 0
    if s.fill == "dealias" and lp > lp0:
        return _dealias(J, lp, s, _chosen(J, s))
    assert s.fill in ("zero", "ring", "clamp"), s.fill
    if s.fill != "zero" and empty.any():
        chosen = _chosen(J, s)
        ring = np.maximum(abs(dr0), abs(dc0)) == lp0 // 2
        dr, dc = offsets(lp)
        for h in np.flatnonzero(chosen):
            if s.fill == "ring":
                new[h, empty] = J[h, ring].mean()
            else:
                for t in np.flatnonzero(empty):
                    new[h, t] = np.mean([J[h, k] for k in _clamp_sources(dr[t], dc[t], lp, lp0)])
            if s.keep_mass:
                new[h] *= np.abs(J[h]).sum() / np.abs(new[h]).sum()
    return new


def _dealias(J, lp, s, chosen):
    """Growing only: per axis, a coordinate at +-2 on the old torus is split between +-2 (1 - w) and its
    image -+3 (w); corner entries split separably. Sums are kept. Unchosen heads are zero-padded."""
    lp0 = math.isqrt(J.shape[-1])
    half, w = lp0 // 2, s.dealias
    dr0, dc0 = offsets(lp0)
    new = np.zeros((J.shape[0], lp * lp), J.dtype)
    for k, (r, c) in enumerate(zip(dr0, dc0)):
        rs = [(r, 1.0)] if abs(r) < half else [(r, 1 - w), (r - lp0 * np.sign(r), w)]
        cs = [(c, 1.0)] if abs(c) < half else [(c, 1 - w), (c - lp0 * np.sign(c), w)]
        for rr, wr in rs:
            for cc, wc in cs:
                t = (rr % lp) * lp + (cc % lp)
                new[:, t] += np.where(chosen, wr * wc, 1.0 if (rr, cc) == (r, c) else 0.0) * J[:, k]
    return new


def _chosen(J, s, layer=None):
    """Heads that get the fill rule (the class rule, then per-head overrides for this layer)."""
    classes = classify(J, s)
    chosen = {"diffuse": classes == "diffuse", "nonlocal": classes != "local",
              "all": np.ones(len(J), bool)}[s.fill_heads].copy()
    for l_, h in s.force_uniform:
        if l_ == layer:
            chosen[h] = True
    for l_, h in s.force_zero:
        if l_ == layer:
            chosen[h] = False
    return chosen


def _clamp_sources(r, c, lp, lp0):
    """Old J entries nearest in direction to offset (r, c) of an lp torus: its minimal images,
    each clamped onto the old kernel's outer ring."""
    reps = [(r + i * lp, c + j * lp) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    dist = [max(abs(a), abs(b)) for a, b in reps]
    half = lp0 // 2
    clamped = {(int(np.clip(a, -half, half)), int(np.clip(b, -half, half)))
               for (a, b), d in zip(reps, dist) if d == min(dist)}
    return [(a % lp0) * lp0 + (b % lp0) for a, b in sorted(clamped)]


def resize_samples(samples, l):
    """l0 x l0 configs -> l x l, by duplicating (or deleting) adjacent row pairs, then column pairs,
    with zero magnetization: S^z stays 0 and the checkerboard stays consistent. Configs without such
    a pair are dropped. A starting point for chains, and a probe set; not samples of the new net."""
    return resize_with_index(samples, l)[0]


def resize_with_index(samples, l, pick=0):
    """resize_samples, plus for each kept config the source site of every new site, and which
    input configs were kept: resized[k] == samples[kept[k]][index[k]]. pick chooses which
    zero-magnetization pair is edited (0 = first, -1 = last), for a second version of each config."""
    l0 = math.isqrt(samples.shape[1])
    assert (l - l0) % 2 == 0, "rows and columns change in pairs"
    out, index, kept = [], [], []
    for n, s in enumerate(samples):
        g, ix = s.reshape(l0, l0), np.arange(l0 * l0).reshape(l0, l0)
        for axis in (0, 1):
            while g is not None and g.shape[axis] != l:
                g, ix = _pair_edit(g, ix, axis, grow=g.shape[axis] < l, pick=pick)
        if g is not None:
            out.append(g.reshape(-1)); index.append(ix.reshape(-1)); kept.append(n)
    return np.stack(out), np.stack(index), np.array(kept)


def _pair_edit(g, ix, axis, grow, pick=0):
    g, ix = np.moveaxis(g, axis, 0), np.moveaxis(ix, axis, 0)
    ok = [i for i in range(g.shape[0] - 1) if g[i:i + 2].sum() == 0]
    if not ok:
        return None, None
    i = ok[pick]
    edit = ((lambda a: np.concatenate([a[:i + 2], a[i:i + 2], a[i + 2:]])) if grow
            else (lambda a: np.concatenate([a[:i], a[i + 2:]])))
    return np.moveaxis(edit(g), 0, axis), np.moveaxis(edit(ix), 0, axis)


def pooled_mean(params, samples, batch=1024):
    """Mean of the pooled feature z = LN(sum of tokens) over samples."""
    f = jax.jit(lambda p, x: vit.layer_norm(vit.tokens(p, x).sum(axis=1), p["output"]["out_layer_norm"]))
    z = np.concatenate([np.asarray(f(params, jnp.asarray(samples[i:i + batch])))
                        for i in range(0, len(samples), batch)])
    return z.mean(0)


def transfer(params, l, s=Settings(), zbar=None):
    """Params for the l x l lattice. zbar (from pooled_mean on 10x10 samples) is needed if kappa != 1."""
    out = jax.tree.map(lambda x: x, params)  # new containers, shared leaves
    for name, layer in out["encoder"].items():
        J = layer["attn"]["J"]
        index = int(name.split("_")[-1])
        layer["attn"] = dict(layer["attn"], J=jnp.asarray(transfer_J(J, l // vit.B, s, index), J.dtype))
    d = vit.D
    gains = {"output_layer0": np.full(d, s.kappa if s.kappa_amp is None else s.kappa_amp),
             "output_layer1": np.full(d, s.kappa if s.kappa_phase is None else s.kappa_phase)}
    if s.kappa_unit3 is not None:
        gains["output_layer1"][3] = s.kappa_unit3
    for unit, gain in s.kappa_sign_units:
        gains["output_layer1"][unit] = gain
    head = out["output"]
    if s.unit3_shift:
        n3 = head["norm3"]
        head["norm3"] = dict(n3, bias=n3["bias"].at[3].add(s.unit3_shift))
    for k, kappa in gains.items():  # per output column
        if (kappa != 1.0).any():
            assert zbar is not None, "kappa != 1 needs zbar"
            W, b = np.asarray(head[k]["kernel"]), np.asarray(head[k]["bias"])
            head[k] = {"kernel": jnp.asarray(kappa * W, W.dtype),
                       "bias": jnp.asarray(b + (1 - kappa) * (np.asarray(zbar) @ W), b.dtype)}
    if s.rotor:
        assert zbar is not None, "the rotor needs zbar"
        r = np.asarray(phase_direction()["r"])
        head["output_layer1"] = add_rotor(head, s.rotor * s.kappa * r, zbar, s.rotor_pairs)
    if s.window:
        assert l // vit.B >= vit.LP, "the window head needs at least a 5x5-token lattice"
        p0 = params["output"]
        mask, avg = np.ones(d), np.zeros(d)
        mask[list(s.window_global_units) + list(s.window_avg_units)] = 0
        avg[list(s.window_avg_units)] = 1
        dtype = p0["norm2"]["scale"].dtype
        head["window"] = {k: p0[k] for k in ("out_layer_norm", "output_layer0", "norm2", "output_layer1", "norm3")}
        head["window"]["mask"] = jnp.asarray(mask, dtype)
        if s.window_avg_units:
            head["window"]["avg"] = jnp.asarray(avg, dtype)
        if s.window_scale != 1.0:
            head["window"]["scale"] = jnp.asarray(s.window_scale, dtype)
    if s.sign_rule:
        units = np.zeros(d)
        units[list(s.sign_units)] = 1
        head["sign_rule"] = jnp.asarray(units, params["output"]["norm3"]["scale"].dtype)
    return out


def phase_direction():
    return json.load(open(vit.ROOT / "wsp" / "phase_direction.json"))


def add_rotor(head, u, zref, pairs=((23, 10),)):
    """output_layer1 with phase u.(z - zref) (rad) added through the sign inputs of rotor unit pairs (p, q).

    For a unit with |a| >> 0, Im logcosh(a + ib) ~ sign(a) b and Re hardly depends on b. Adding +D to p's and
    -D to q's pre-LayerNorm input (a_p > 0 > a_q) adds (sens_p g3_p - sens_q g3_q) D / sigma3 to the phase
    and leaves LayerNorm 3's mean unchanged (its spread changes at second order)."""
    d = phase_direction()
    sens, g3 = np.asarray(d["phase_sens"]), np.asarray(head["norm3"]["scale"])
    k = head["output_layer1"]
    W, b = np.array(k["kernel"]), np.array(k["bias"])
    for p, q in pairs:
        v = np.asarray(u) / len(pairs) * d["sigma3"] / (sens[p] * g3[p] - sens[q] * g3[q])
        W[:, p] += v
        W[:, q] -= v
        b[p] -= v @ np.asarray(zref)
        b[q] += v @ np.asarray(zref)
    return {"kernel": jnp.asarray(W, k["kernel"].dtype), "bias": jnp.asarray(b, k["bias"].dtype)}


def from_flags(l, amp_scale=1.0, unit3_scale=1.0, fill="uniform", fill_heads="all", sets=()):
    """Settings from command-line style flags: gains are multiples of kappa = the token ratio,
    sets are "field=json" strings (kappa_phase also a multiple of kappa)."""
    kappa = (l // vit.B) ** 2 / vit.LP ** 2
    extra = {k: (tuple(map(tuple, v)) if k == "rotor_pairs" else tuple(v) if isinstance(v, list) else v)
             for k, v in ((a.split("=", 1)[0], json.loads(a.split("=", 1)[1])) for a in sets)}
    s = Settings(kappa=kappa, fill=fill, fill_heads=fill_heads,
                 kappa_amp=None if amp_scale == 1 else amp_scale * kappa,
                 kappa_unit3=None if unit3_scale == 1 else unit3_scale * kappa, **extra)
    if s.kappa_phase is not None:
        s = Settings(**{**vars(s), "kappa_phase": s.kappa_phase * kappa})
    return s
