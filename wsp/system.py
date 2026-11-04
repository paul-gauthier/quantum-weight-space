"""10x10 J1-J2 Heisenberg model at J2/J1 = 0.5, as in the nqs-models model card.

`make(l)` builds the same model on a periodic l x l lattice; the module-level names are l = 10.
"""

import functools
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import netket as nk

L = 10
J2 = 0.5


@functools.lru_cache
def make(l):
    """Lattice, Hilbert space and Hamiltonian of the periodic l x l model."""
    n = l * l
    lattice = nk.graph.Hypercube(length=l, n_dim=2, pbc=True, max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=n, total_sz=0)
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
    ).to_jax_operator()
    # NetKet uses Pauli matrices: divide energies by 4 * N for the S = sigma/2 convention.
    return SimpleNamespace(L=l, N=n, lattice=lattice, hilbert=hilbert, hamiltonian=hamiltonian,
                           PER_SITE=1.0 / (4 * n))


def for_samples(samples):
    """The model whose lattice matches samples of shape (..., l*l)."""
    return make(math.isqrt(samples.shape[-1]))


_default = make(L)
N = _default.N
lattice = _default.lattice
hilbert = _default.hilbert
hamiltonian = _default.hamiltonian
PER_SITE = _default.PER_SITE


def make_state(apply_fun, params, samples, chunk_size=1024, seed=0):
    """MCState whose chains start from the given samples (thermalized, or a cold start)."""
    system = for_samples(samples)
    n = samples.shape[0]
    sampler = nk.sampler.MetropolisExchange(
        hilbert=system.hilbert, graph=system.lattice, d_max=2, n_chains=n, sweep_size=system.N
    )
    vs = nk.vqs.MCState(
        sampler=sampler,
        apply_fun=apply_fun,
        sampler_seed=jax.random.PRNGKey(seed),
        n_samples=n,
        n_discard_per_chain=0,
        variables={"params": params},
        chunk_size=chunk_size,
    )
    vs.sampler_state = vs.sampler_state.replace(σ=jnp.asarray(samples, dtype=jnp.int8))
    return vs
