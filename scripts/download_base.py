"""Download the released 10x10 ViT NQS (Rende, Viteritti et al.) from Hugging Face into checkpoints/hf/REVISION.

nqs-models/j1j2_square_10x10_05 (Apache-2.0). Only the files this code reads are fetched, plus the model card.
main ("Plain") is the network every result here starts from. symm_trxy_ising is also read by
scripts/lanczos.py --check, and symm_t and symm_trxy_ising by scripts/sign_rule_check.py's default revisions.

usage: python scripts/download_base.py [REVISION ...]      (default: main)
"""
import sys
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/nqs-models/j1j2_square_10x10_05/resolve"
FILES = {"main": ("model.safetensors", "spins", "config.json", "README.md"),
         "symm_t": ("model.safetensors", "spins", "config.json", "README.md", "symm_indexes.txt"),
         "symm_trxy_ising": ("model.safetensors", "spins", "config.json", "README.md", "symm_indexes.txt")}
ROOT = Path(__file__).resolve().parent.parent

for revision in sys.argv[1:] or ["main"]:
    dest = ROOT / "checkpoints" / "hf" / revision
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES[revision]:
        if (dest / name).exists():
            continue
        print(f"{revision}/{name}", flush=True)
        tmp = dest / (name + ".part")
        urllib.request.urlretrieve(f"{REPO}/{revision}/{name}", tmp)
        tmp.replace(dest / name)
    print("saved", dest)
