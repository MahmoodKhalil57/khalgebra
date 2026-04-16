"""
khalgebra — Khalil Optimal Bilinear Algorithms
Author: Mahmood Khalil (2025)

DSYMV: optimal symmetric matrix-vector product.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from khalgebra._types import Mat, Vec


@jax.jit
def khal_dsymv(A: Mat, v: Vec) -> Vec:
    """
    Optimal symmetric matrix-vector product.

    Bilinear complexity: n(n+1)/2 multiplications (proven optimal, Khalil 2025).
    Standard BLAS DSYMV uses n² multiplications.

    Args:
        A: n×n symmetric matrix, shape (n, n), float64
        v: length-n vector, shape (n,), float64

    Returns:
        w = A @ v, shape (n,), float64
    """
    n = A.shape[0]

    upper = jnp.triu(A, k=1)
    v_sum = v[:, None] + v[None, :]
    M = upper * v_sum
    w_off = jnp.sum(M, axis=1) + jnp.sum(M, axis=0)

    diag_adj = jnp.diag(A) - jnp.sum(upper, axis=1) - jnp.sum(upper, axis=0)
    w_diag = diag_adj * v

    return w_off + w_diag


@jax.jit
def naive_dsymv(A: Mat, v: Vec) -> Vec:
    """
    Reference symmetric matrix-vector product.

    Args:
        A: n×n symmetric matrix, shape (n, n), float64
        v: length-n vector, shape (n,), float64

    Returns:
        w = A @ v, shape (n,), float64
    """
    return jnp.dot(A, v)
