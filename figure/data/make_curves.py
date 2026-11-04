"""Training curves (panel c) and per-size energies (panels d) for the blog figure.

Curves: 14x14 energy per site on the training chains, 10-step running means, from the run reports
(results/finetune/, weights/ft14.json). Energies: the table in README.md (sources in results/); published values from
results/published_energies.json (best variational). 16x16 fine-tuned is the 200-sweep audit (results/sanity/audit.json).

usage (from the repo root): python figure/data/make_curves.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT
BENCH = json.loads((ROOT / "results/published_energies.json").read_text())
BEST = {L: v["E"] for L, v in BENCH["best_variational"].items()}


def trace(path, offset=0):
    t = json.loads((FT / path).read_text())["trace"]
    return [(x["step"] + offset, x["E"]) for x in t]


def running(tr, w=10):
    e = [E for _, E in tr]
    return [[s, sum(e[max(0, i - w + 1):i + 1]) / len(e[max(0, i - w + 1):i + 1])] for i, (s, _) in enumerate(tr)]


ours = trace("results/finetune/ft14_steps0-199.json") + trace("weights/ft14.json", 200)
curves = {
    "ours": running(ours),
    "naive": running(trace("results/finetune/ft14_naive_lr0.01.json")),
    "scratch": running(trace("results/finetune/ft14_scratch_lr0.003.json")),
    "ours_lanczos": {"step": 500, "E": -0.496753, "err": 0.000025},
    "published": BEST["14"],
}

sizes = {
    "12": {"edits": -0.49564, "ft": -0.496904, "ft_err": 0.000042, "lz": -0.497019, "lz_err": 0.000016},
    "14": {"edits": -0.49399, "ft": -0.496665, "ft_err": 0.000030, "lz": -0.496753, "lz_err": 0.000025},
    "16": {"edits": -0.49303, "ft": -0.496470, "ft_err": 0.000013, "lz": -0.496512, "lz_err": 0.000022},
    "18": {"edits": -0.49220, "ft": -0.496368, "ft_err": 0.000032, "lz": -0.496469, "lz_err": 0.000016},
}
for L, d in sizes.items():
    d["pub"] = BEST[L]

# panel b: per-site disorder penalty from scripts/extensivity.py (256 configurations per size)
penalty = {"L": [10, 12, 14, 16, 18, 20],
           "global": [0.266, 0.230, 0.192, 0.156, 0.131, 0.111],
           "window": [0.266, 0.260, 0.265, 0.271, 0.278, 0.281]}

dst = Path(__file__).with_name("plots.json")
dst.write_text(json.dumps({"curves": curves, "sizes": sizes, "penalty": penalty}))
for k in ("ours", "naive", "scratch"):
    c = curves[k]
    print(k, len(c), "first %.4f last %.4f" % (c[0][1], c[-1][1]))
print("wrote", dst)
