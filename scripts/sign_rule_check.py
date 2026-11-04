"""Checks of the sign rule (Settings.sign_rule, vit.sign_rule_phase) before any MCMC.

10x10, each released revision: the base net's phase on every native copy of the released samples, against
   the rule (mod 2 pi, one constant removed); and the native state's energy with the rule, reweighted from the
   released samples (the samples come from the released state), against the released state itself.

usage: PYTHONPATH=. python -u scripts/sign_rule_check.py [--n 512] [--revisions main symm_t symm_trxy_ising]
       [--sign-units 3 6] [--output OUT.json]
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from wsp import fast, native, vit
from wsp.transfer import Settings, transfer

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n", type=int, default=512, help="released samples per revision for part 1")
p.add_argument("--revisions", nargs="*", default=["main", "symm_t", "symm_trxy_ising"])
p.add_argument("--sign-units", type=int, nargs="+", default=[3], help="Settings.sign_units")
p.add_argument("--chunk", type=int, default=4096)
p.add_argument("--output")
args = p.parse_args()
report = {"10x10": {}}


def centred(phase):
    res = np.angle(np.exp(1j * phase))
    return np.angle(np.exp(1j * (res - np.angle(np.mean(np.exp(1j * res))))))


for rev in args.revisions:
    P = vit.load_params(rev)
    R = transfer(P, vit.L, Settings(sign_rule=True, sign_units=tuple(args.sign_units)))
    S = vit.load_samples(rev)[:args.n]
    perms, flips = native.copies(rev, vit.L)
    own = fast.copy_terms(P, S, perms, flips, chunk=args.chunk)
    m = own["lp"].real.max(0)
    ref = m + np.log(np.exp(own["lp"] - m).sum(0))  # log psi of the released state, which drew the samples
    rule = fast.copy_terms(R, S, perms, flips, ref=ref, chunk=args.chunk)
    d = rule["lp"] - own["lp"]
    res = centred(d.imag.ravel())  # one constant for all copies: the base net's own constant
    e_own, e_rule = fast.energy(own), fast.energy(rule)
    out = {"copies": len(perms), "samples": len(S), "max_d_log_abs": float(np.abs(d.real).max()),
           "phase_within_1mrad": float((np.abs(res) < 1e-3).mean()), "phase_within_0.1rad": float((np.abs(res) < 0.1).mean()),
           "phase_rms": float(np.sqrt((res ** 2).mean())), "E_own": e_own["mean"], "E_rule": e_rule["mean"],
           "dE": e_rule["mean"] - e_own["mean"], "err_own": e_own["err"], "ess_rule": e_rule["ess"]}
    report["10x10"][rev] = out
    print(f"10x10 {rev:16s} copies {out['copies']:2d}  phase vs rule: {out['phase_within_1mrad']:.1%} within 1 mrad, "
          f"{out['phase_within_0.1rad']:.1%} within 0.1 rad, rms {out['phase_rms']:.4f}  |  E own {out['E_own']:.6f}"
          f"  rule {out['E_rule']:.6f}  dE {out['dE']:+.2e} (err {out['err_own']:.1e}, ESS {out['ess_rule']:.0f})",
          flush=True)

if args.output:
    json.dump(report, open(args.output, "w"), indent=1)
