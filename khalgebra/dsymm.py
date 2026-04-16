"""
khalgebra — Khalil Optimal Bilinear Algorithms
Author: Mahmood Khalil (2025)

DSYMM: optimal symmetric matrix-matrix product.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from khalgebra._types import Mat


@jax.jit
def khal_dsymm(A: Mat, B: Mat) -> Mat:
    """
    Optimal symmetric matrix-matrix product.

    Bilinear complexity: m·n(n+1)/2 multiplications (Khalil 2025).
    Standard BLAS DSYMM uses m·n² multiplications.

    Args:
        A: n×n symmetric matrix, shape (n, n), float64
        B: n×m general matrix, shape (n, m), float64

    Returns:
        C = A @ B, shape (n, m), float64
    """
    upper = jnp.triu(A, k=1)
    diag_adj = jnp.diag(A) - jnp.sum(upper, axis=1) - jnp.sum(upper, axis=0)

    def _col(v):
        v_sum = v[:, None] + v[None, :]
        M = upper * v_sum
        w_off = jnp.sum(M, axis=1) + jnp.sum(M, axis=0)
        return w_off + diag_adj * v

    return jax.vmap(_col, in_axes=1, out_axes=1)(B)


@jax.jit
def naive_dsymm(A: Mat, B: Mat) -> Mat:
    """
    Reference symmetric matrix-matrix product.

    Args:
        A: n×n symmetric matrix, shape (n, n), float64
        B: n×m general matrix, shape (n, m), float64

    Returns:
        C = A @ B, shape (n, m), float64
    """
    return jnp.dot(A, B)
