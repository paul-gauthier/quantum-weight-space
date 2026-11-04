"""J kernels for panel a: one local and one diffuse attention head of the released 10x10 ViT (revision main),
on the trained 5x5 patch torus and after our transfer rule to the 18x18 lattice (9x9 patch torus).

The rule is the fine-tune "ours" start (scripts/finetune.py): fill="uniform", fill_heads="nonlocal", local=0.7.
Local heads are zero-padded; nonlocal heads have their mean J spread over the whole new torus (sum kept) and the
rest zero-padded.

usage (from the repo root): python figure/data/make_j.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wsp import vit  # noqa: E402
from wsp.transfer import Settings, classify, near_share, offsets, transfer_J  # noqa: E402

S = Settings(fill="uniform", fill_heads="nonlocal", local=0.7)
P = vit.load_params("main")
blocks = sorted(k for k in P["Encoder_0"] if k.startswith("EncoderBlock")) if "Encoder_0" in P else None


def find_J(tree, path=()):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from find_J(v, path + (k,))
        elif k == "J":
            yield path, np.asarray(v)


Js = sorted(find_J(P), key=lambda t: t[0])
print([p for p, _ in Js])


def grid(j):
    """(lp*lp,) entries indexed by (dr mod lp, dc mod lp) -> lp x lp array with offset (0, 0) at the centre."""
    lp = int(round(len(j) ** 0.5))
    dr, dc = offsets(lp)
    g = np.zeros((lp, lp))
    c = lp // 2
    for k, (r, q) in enumerate(zip(dr, dc)):
        g[r + c, q + c] = j[k]
    return g


out = {"rule": "fill=uniform, fill_heads=nonlocal, local=0.7 (scripts/finetune.py 'ours')", "heads": []}
for li, (path, J) in enumerate(Js):
    J = J.reshape(J.shape[0], -1) if J.ndim > 2 else J
    near = near_share(J)
    cls = classify(J, S)
    print(li, path, J.shape, np.round(near, 2), list(cls))

# pick: the most local head and the most diffuse head among layers with a clear example
pick = {}
for li, (path, J) in enumerate(Js):
    near = near_share(J)
    for h in range(len(J)):
        if near[h] >= 0.7 and ("local" not in pick or near[h] > pick["local"][2]):
            pick["local"] = (li, h, near[h])
        if near[h] <= 0.5 and ("diffuse" not in pick or near[h] < pick["diffuse"][2]):
            pick["diffuse"] = (li, h, near[h])
print(pick)

for kind, (li, h, nr) in pick.items():
    J = Js[li][1]
    J9 = transfer_J(J, 9, S, layer=li)
    out["heads"].append({"kind": kind, "layer": li, "head": int(h), "near_share": float(nr),
                         "J5": grid(J[h]).tolist(), "J9": grid(J9[h]).tolist()})

dst = Path(__file__).with_name("j_kernels.json")
dst.write_text(json.dumps(out))
print("wrote", dst)
