"""Weight reads that set the lattice-transfer rules.

Per head, from the weights: where J sits by patch distance, J's asymmetry under the 8 point
operations and under inversion, and the head's gain-weighted OV norm (LN scale x V x W_out).
Per head, FFN unit and residual dim, on released samples (forward passes, no edits): how much
each one actually writes. Low-use FFN units and output units are candidates for a graft.

usage: PYTHONPATH=. python -u scripts/weight_reads.py [REVISION [N_SAMPLES]]
"""
import json, os, sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from wsp import vit
from wsp.transfer import Settings, classify, near_share, offsets

rev = sys.argv[1] if len(sys.argv) > 1 else "main"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
P = vit.load_params(rev)
S = jnp.asarray(vit.load_samples(rev)[:n])
H, DH, NL = vit.H, vit.D // vit.H, vit.N_LAYERS

dr, dc = offsets(vit.LP)
cheb, manh = np.maximum(abs(dr), abs(dc)), abs(dr) + abs(dc)
idx = lambda r, c: (r % vit.LP) * vit.LP + (c % vit.LP)
POINT = [lambda r, c: (r, c), lambda r, c: (-c, r), lambda r, c: (-r, -c), lambda r, c: (c, -r),
         lambda r, c: (c, r), lambda r, c: (-r, c), lambda r, c: (r, -c), lambda r, c: (-c, -r)]
OPS = [np.array([idx(*op(r, c)) for r, c in zip(dr, dc)]) for op in POINT]


def run(spins):
    """The encoder, with per-layer usage statistics of each head and FFN unit."""
    x = vit.dense(vit.extract_patches(spins.astype(jnp.float64)), P["patches_and_embed"]["embed"])
    keep = []
    for i in range(NL):
        p = P["encoder"][f"layers_{i}"]
        v = vit.dense(vit.layer_norm(x, p["layer_norm_1"]), p["attn"]["v"])
        b, t, _ = v.shape
        v = v.reshape(b, t, H, DH).transpose(0, 2, 1, 3)
        mixed = jnp.einsum("hpq,bhqd->bhpd", vit.full_attention(p["attn"]["J"]), v)
        heads = jnp.einsum("bhpd,hde->bhpe", mixed, p["attn"]["W"]["kernel"].reshape(H, DH, vit.D))
        attn = heads.sum(1)
        x = x + attn + p["attn"]["W"]["bias"]
        pre = vit.dense(vit.layer_norm(x, p["layer_norm_2"]), p["ff"]["layers_0"])
        g = jax.nn.gelu(pre, approximate=True)
        ff = vit.dense(g, p["ff"]["layers_2"])
        x = x + ff
        keep.append({"head_write": jnp.sqrt((heads ** 2).sum(-1).mean((0, 2))),  # rms |write| per token
                     "attn_rms": jnp.sqrt((attn ** 2).sum(-1).mean()),
                     "g2": (g ** 2).mean((0, 1)), "active": (pre > 0).mean((0, 1)),
                     "ffn_rms": jnp.sqrt(((ff - p["ff"]["layers_2"]["bias"]) ** 2).sum(-1).mean())})
    return x, keep


x, keep = jax.jit(run)(S)
assert np.allclose(np.asarray(x), np.asarray(vit.tokens(P, S)), atol=1e-10)
keep = [{k: np.asarray(v) for k, v in d.items()} for d in keep]

s = Settings()
out = {"revision": rev, "samples": n, "settings": vars(s) | {}, "layers": []}
print(f"== heads ({rev}, {n} samples)   near = share of |J| at Chebyshev patch distance <= 1")
print("layer head  class    self  near  far3+  ring_coh  D4_asym  inv_asym  sumJ    |OV|   write_rms")
for i in range(NL):
    p = P["encoder"][f"layers_{i}"]
    J = np.asarray(p["attn"]["J"])
    classes, near = classify(J, s), near_share(J)
    a = np.abs(J)
    gain = np.asarray(p["layer_norm_1"]["scale"])
    Wv = np.asarray(p["attn"]["v"]["kernel"]).reshape(vit.D, H, DH)
    Wo = np.asarray(p["attn"]["W"]["kernel"]).reshape(H, DH, vit.D)
    k = keep[i]
    write = k["head_write"]
    W2 = np.asarray(p["ff"]["layers_2"]["kernel"])
    unit_write = np.sqrt(k["g2"]) * np.linalg.norm(W2, axis=1)  # rms write per FFN unit
    active = k["active"]
    layer = {"heads": [], "ffn": {}}
    for h in range(H):
        sym = np.mean([J[h][op] for op in OPS], axis=0)
        ring = J[h][cheb == 2]
        ov = np.linalg.norm(gain[:, None] * Wv[:, h] @ Wo[h])
        row = {"class": str(classes[h]), "self": a[h, cheb == 0].sum() / a[h].sum(), "near": near[h],
               "far3": a[h, manh >= 3].sum() / a[h].sum(),
               "ring_coherence": abs(ring.mean()) / np.sqrt((ring ** 2).mean()),
               "d4_asym": np.linalg.norm(J[h] - sym) / np.linalg.norm(J[h]),
               "inv_asym": np.linalg.norm(J[h] - J[h][OPS[2]]) / (2 * np.linalg.norm(J[h])),
               "sum_J": J[h].sum(), "ov": ov, "write_rms": write[h]}
        layer["heads"].append({k: (float(v) if k != "class" else v) for k, v in row.items()})
        print(f"{i:5d} {h:4d}  {row['class']:7s} {row['self']:5.2f} {row['near']:5.2f} {row['far3']:6.2f}"
              f"  {row['ring_coherence']:8.2f}  {row['d4_asym']:7.2f}  {row['inv_asym']:8.2f}"
              f"  {row['sum_J']:6.2f}  {ov:6.2f}  {write[h]:9.3f}")
    med = np.median(unit_write)
    layer["ffn"] = {"units": len(unit_write), "write_median": float(med),
                    "low_1pct_of_median": int((unit_write < 0.01 * med).sum()),
                    "low_10pct_of_median": int((unit_write < 0.1 * med).sum()),
                    "active_under_1pct": int((active < 0.01).sum()),
                    "lowest": [int(u) for u in np.argsort(unit_write)[:8]],
                    "lowest_write": [float(w) for w in np.sort(unit_write)[:8]],
                    "attn_write_total": float(k["attn_rms"]), "ffn_write_total": float(k["ffn_rms"])}
    out["layers"].append(layer)

print("\n== FFN units per layer: write = rms(gelu) x |W_out row|")
print("layer  median   <1%med  <10%med  active<1%  attn_rms  ffn_rms   lowest units")
for i, L_ in enumerate(out["layers"]):
    f = L_["ffn"]
    print(f"{i:5d}  {f['write_median']:6.3f}  {f['low_1pct_of_median']:7d}  {f['low_10pct_of_median']:7d}"
          f"  {f['active_under_1pct']:9d}  {f['attn_write_total']:8.3f}  {f['ffn_write_total']:7.3f}   {f['lowest'][:6]}")

# Residual stream and head: how much each pooled dim and each output unit varies and is read.
o = P["output"]
z = np.asarray(vit.layer_norm(x.sum(1), o["out_layer_norm"]))
W0, W1 = np.asarray(o["output_layer0"]["kernel"]), np.asarray(o["output_layer1"]["kernel"])
read = z.std(0) * (np.linalg.norm(W0, axis=1) + np.linalg.norm(W1, axis=1))
amp = np.asarray(vit.layer_norm(vit.dense(jnp.asarray(z), o["output_layer0"]), o["norm2"]))
sgn = np.asarray(vit.layer_norm(vit.dense(jnp.asarray(z), o["output_layer1"]), o["norm3"]))
# a unit's log cosh(a + ib) term moves log psi by about |tanh a| std(a) + |tan b| std(b)
unit_use = np.abs(np.tanh(amp)).mean(0) * amp.std(0) + sgn.std(0) * np.abs(np.tan(sgn)).clip(max=50).mean(0)
out["pooled_dims"] = {"read": read.tolist(), "lowest": [int(k) for k in np.argsort(read)[:8]]}
out["output_units"] = {"a_mean": amp.mean(0).tolist(), "a_std": amp.std(0).tolist(),
                       "b_mean": sgn.mean(0).tolist(), "b_std": sgn.std(0).tolist(),
                       "use": unit_use.tolist(), "lowest": [int(k) for k in np.argsort(unit_use)[:10]]}
print("\n== pooled z dims: std(z) x |head rows|;  median {:.3f}".format(np.median(read)))
print("  lowest dims", [f"{k}:{read[k]:.3f}" for k in np.argsort(read)[:8]])
print("== output units: |tanh a| std(a) + |tan b| std(b);  median {:.3f}".format(np.median(unit_use)))
print("  lowest units", [f"{k}:{unit_use[k]:.3f}" for k in np.argsort(unit_use)[:10]])
print("  unit 3: a {:.4f} +- {:.4f}, b {:.4f} +- {:.4f}".format(amp[:, 3].mean(), amp[:, 3].std(), sgn[:, 3].mean(), sgn[:, 3].std()))

cls = np.array([[hd["class"] for hd in L_["heads"]] for L_ in out["layers"]])
print("\n== classes by layer (local / mixed / diffuse):")
for i in range(NL):
    print(f"  layer {i}: {(cls[i] == 'local').sum():2d} / {(cls[i] == 'mixed').sum():2d} / {(cls[i] == 'diffuse').sum():2d}")
path = f"runs/weight_reads_{rev}.json"
os.makedirs("runs", exist_ok=True)
json.dump(out, open(path, "w"), indent=1)
print("saved", path)
