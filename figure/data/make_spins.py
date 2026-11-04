"""Spin configurations drawn on the lattices in panel a: one released 10x10 sample (revision main), and one 18x18
sample of the fine-tuned 18x18 state (guided Lanczos samples, results/lanczos/18/ft18cg_s57.npz;
the configuration whose importance weight is closest to the median).

usage (from the repo root): python figure/data/make_spins.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wsp import vit  # noqa: E402

S = np.asarray(vit.load_samples("main"))
s10 = S[7]
Z = np.load(ROOT / "results/lanczos/18/ft18cg_s57.npz")
w = Z["logw"]
s18 = Z["samples"][int(np.argmin(abs(w - np.median(w))))]
out = {"source": "10x10: checkpoints/hf/main spins, sample 7; 18x18: ft18cg_s57.npz, median-weight sample",
       "s10": s10.reshape(10, 10).astype(int).tolist(), "s18": s18.reshape(18, 18).astype(int).tolist()}
dst = Path(__file__).with_name("spins.json")
dst.write_text(json.dumps(out))
for k in ("s10", "s18"):
    a = np.array(out[k]); L = a.shape[0]
    stag = (a * ((-1) ** np.add.outer(np.arange(L), np.arange(L)))).mean()
    print(k, a.shape, "sum", a.sum(), "staggered mag %.2f" % stag)
