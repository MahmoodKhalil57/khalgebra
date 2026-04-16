"""
khalgebra — Khalil Optimal Bilinear Algorithms
Author: Mahmood Khalil (2025)

Shared type aliases, test data generators, and verification helpers.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

Vec     = Array
Mat     = Array
Tensor4 = Array

def _lcg_sequence(n: int, seed: int) -> list[float]:
    x = seed
    out: list[float] = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x / 0x7FFFFFFF * 2 - 1)
    return out

def make_sym_mat(n: int, seed: int = 42) -> Mat:
    flat = _lcg_sequence(n * n, seed)
    A = [[0.0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        A[i][i] = abs(flat[idx]) + 1
        idx += 1
        for j in range(i + 1, n):
            v = flat[idx]; idx += 1
            A[i][j] = v
            A[j][i] = v
    return jnp.array(A, dtype=jnp.float64)


def make_gen_mat(rows: int, cols: int, seed: int = 77) -> Mat:
    flat = _lcg_sequence(rows * cols, seed)
    return jnp.array(flat, dtype=jnp.float64).reshape(rows, cols)


def make_vec(n: int, seed: int = 1) -> Vec:
    flat = _lcg_sequence(n, seed)
    return jnp.array(flat, dtype=jnp.float64)


def make_riemann_tensor(n: int, seed: int = 42) -> Tensor4:
    comps = build_riemann_components(n)
    vals = _lcg_sequence(len(comps), seed)
    R = [[[[0.0] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for (a, b, c, d), v in zip(comps, vals):
        R[a][b][c][d] =  v
        R[b][a][c][d] = -v
        R[a][b][d][c] = -v
        R[b][a][d][c] =  v
        if a * n + b != c * n + d:
            R[c][d][a][b] =  v
            R[d][c][a][b] = -v
            R[c][d][b][a] = -v
            R[d][c][b][a] =  v
    return jnp.array(R, dtype=jnp.float64)

def max_abs_err(a: Array, b: Array) -> float:
    return float(jnp.max(jnp.abs(a - b)))


def build_riemann_components(n: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(n):
                for d in range(c + 1, n):
                    if a * n + b <= c * n + d:
                        out.append((a, b, c, d))
    return out
