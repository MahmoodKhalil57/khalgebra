"""
khalgebra — Khalil Optimal Bilinear Algorithms
Author: Mahmood Khalil (2025)

Sym22: optimal 2×2 symmetric × 2×2 general matrix product.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def khal_sym22(
    a: float, b: float, c: float,
    p: float, q: float, r: float, s: float,
) -> tuple[float, float, float, float]:
    """
    Optimal 2×2 symmetric × 2×2 general matrix product.

    Bilinear complexity: 6 scalar multiplications (proven optimal, Khalil 2025).
    Naive requires 8.

    Args:
        a: A[0,0]
        b: A[0,1] = A[1,0]
        c: A[1,1]
        p: B[0,0]
        q: B[0,1]
        r: B[1,0]
        s: B[1,1]

    Returns:
        (C[0,0], C[0,1], C[1,0], C[1,1]) where C = A·B
    """
    ab = a - b
    cb = c - b
    M1 = b * (p + r)
    M2 = ab * p
    M3 = cb * r
    M4 = b * (q + s)
    M5 = ab * q
    M6 = cb * s
    return (M1 + M2, M4 + M5, M1 + M3, M4 + M6)


def naive_sym22(
    a: float, b: float, c: float,
    p: float, q: float, r: float, s: float,
) -> tuple[float, float, float, float]:
    """
    Reference 2×2 symmetric × 2×2 general matrix product.

    Args:
        a, b, c: upper triangle of symmetric A: [[a,b],[b,c]]
        p, q, r, s: entries of general B: [[p,q],[r,s]]

    Returns:
        (C[0,0], C[0,1], C[1,0], C[1,1]) where C = A·B
    """
    return (a * p + b * r, a * q + b * s, b * p + c * r, b * q + c * s)
