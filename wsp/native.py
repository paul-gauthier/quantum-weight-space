"""Native symmetrized released states on any even square lattice.

The encoder is invariant under translations by two sites. Four coset representatives
therefore suffice for full translation symmetry; adding the eight square point-group
operations and spin flip gives the released trxy state's 64 evaluations. At L=10,
the geometric representatives need not be in the release file's order, but define
the same state because their difference is an encoder-invariant patch translation.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import vit


@functools.lru_cache
def copies(revision, l):
    """Return (gather permutations, spin-flip signs) for the revision's native state."""
    if l < 4 or l % vit.B:
        raise ValueError("The lattice side must be even and at least four")
    grid = np.arange(l * l, dtype=np.int32).reshape(l, l)
    identity = grid.reshape(-1)
    if revision == "main":
        return identity[None], np.ones(1, np.int8)
    shifts = [np.roll(np.roll(grid, dr, axis=0), dc, axis=1).reshape(-1)
              for dr in (0, 1) for dc in (0, 1)]
    if revision == "symm_t":
        return np.stack(shifts), np.ones(4, np.int8)
    if revision != "symm_trxy_ising":
        raise ValueError(f"Unknown released revision: {revision}")
    points = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        points.extend([rotated.reshape(-1), rotated.T.reshape(-1)])
    spatial = np.stack([p[s] for s in shifts for p in points])
    return (np.concatenate([spatial, spatial]),
            np.repeat(np.array([1, -1], np.int8), len(spatial)))


def apply_fun(revision, l, copy_batch=8):
    """NetKet apply_fun evaluating the native sum, in bounded batches of copies.

    Each copy batch is flattened into the input batch so the base ViT sees a large
    GPU-friendly matrix multiplication. Different copy batches use lax.map to bound
    activation memory. The caller can also bound the sample batch with MCState's
    chunk_size. copy_batch must divide the copy count (values above it are clipped).
    This returns log(sum psi_g), with the same unnormalized sum as the release.
    """
    permutations, flips = copies(revision, l)
    g = len(flips)
    if copy_batch < 1:
        raise ValueError("copy_batch must be positive")
    copy_batch = min(int(copy_batch), g)
    if g % copy_batch:
        raise ValueError(f"copy_batch={copy_batch} does not divide {g} copies")
    permutations = jnp.asarray(permutations.reshape(-1, copy_batch, l * l))
    flips = jnp.asarray(flips.reshape(-1, copy_batch))

    def apply(variables, spins):
        spins = jnp.atleast_2d(spins)
        leading = spins.shape[:-1]
        flat = spins.reshape(-1, l * l)

        def block(pf):
            perm, flip = pf
            transformed = flat[:, perm] * flip[None, :, None]
            lp = vit.log_psi(variables, transformed.reshape(-1, l * l))
            return lp.reshape(len(flat), copy_batch).T

        lp = jax.lax.map(block, (permutations, flips)).reshape(g, len(flat))
        maximum = lp.real.max(axis=0)
        summed = maximum + jnp.log(jnp.exp(lp - maximum).sum(axis=0))
        return summed.reshape(leading)

    return apply
