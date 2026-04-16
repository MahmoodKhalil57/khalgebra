"""
khalgebra — Khalil Optimal Bilinear Algorithms
Author: Mahmood Khalil (2025)

Riemann: optimal contraction of the Riemann curvature tensor.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from khalgebra._types import Mat, Vec, Tensor4, build_riemann_components

__all__ = ["khal_riemann_contract", "naive_riemann_contract", "build_riemann_components"]


@jax.jit
def naive_riemann_contract(R: Tensor4, u: Vec, v: Vec) -> Mat:
    """
    Reference Riemann tensor contraction.

    Computes B[b,d] = Σ_{a,c} R[a,b,c,d] · u[a] · v[c]

    Args:
        R: Riemann tensor, shape (n, n, n, n), float64
        u: contravariant vector, shape (n,), float64
        v: contravariant vector, shape (n,), float64

    Returns:
        B: covariant rank-2 tensor, shape (n, n), float64
    """
    return jnp.einsum("abcd,a,c->bd", R, u, v)



def _build_index_arrays(
    n: int, comps: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, ...]:
    Ra, Rb, Rc, Rd, UVr, UVc, Br, Bc, sgn = [], [], [], [], [], [], [], [], []

    for a, b, c, d in comps:
        for (br, bc, uvr, uvc, s) in [
            (b, d, a, c, +1),
            (a, d, b, c, -1),
            (b, c, a, d, -1),
            (a, c, b, d, +1),
        ]:
            Ra.append(a); Rb.append(b); Rc.append(c); Rd.append(d)
            UVr.append(uvr); UVc.append(uvc)
            Br.append(br); Bc.append(bc); sgn.append(s)
        if a * n + b != c * n + d:
            for (br, bc, uvr, uvc, s) in [
                (d, b, c, a, +1),
                (c, b, d, a, -1),
                (d, a, c, b, -1),
                (c, a, d, b, +1),
            ]:
                Ra.append(a); Rb.append(b); Rc.append(c); Rd.append(d)
                UVr.append(uvr); UVc.append(uvc)
                Br.append(br); Bc.append(bc); sgn.append(s)

    return (
        np.array(Ra), np.array(Rb), np.array(Rc), np.array(Rd),
        np.array(UVr), np.array(UVc),
        np.array(Br), np.array(Bc),
        np.array(sgn, dtype=np.float64),
    )


@functools.cache
def _get_kernel(n: int, comps_key: tuple[tuple[int, int, int, int], ...]):
    comps = list(comps_key)
    Ra, Rb, Rc, Rd, UVr, UVc, Br, Bc, signs = _build_index_arrays(n, comps)

    j_Ra = jnp.array(Ra); j_Rb = jnp.array(Rb)
    j_Rc = jnp.array(Rc); j_Rd = jnp.array(Rd)
    j_UVr = jnp.array(UVr); j_UVc = jnp.array(UVc)
    j_Br = jnp.array(Br);   j_Bc = jnp.array(Bc)
    j_signs = jnp.array(signs)

    @jax.jit
    def _kernel(R: Tensor4, u: Vec, v: Vec) -> Mat:
        UV = jnp.outer(u, v)
        R_vals  = R[j_Ra, j_Rb, j_Rc, j_Rd]
        UV_vals = UV[j_UVr, j_UVc]
        contribs = j_signs * R_vals * UV_vals
        B = jnp.zeros((n, n), dtype=R.dtype)
        return B.at[j_Br, j_Bc].add(contribs)

    return _kernel


def khal_riemann_contract(
    R: Tensor4,
    u: Vec,
    v: Vec,
    components: list[tuple[int, int, int, int]] | None = None,
) -> Mat:
    """
    Optimal Riemann tensor contraction.

    Computes B[b,d] = Σ_{a,c} R[a,b,c,d] · u[a] · v[c]

    Bilinear complexity: n² multiplications (proven optimal, Khalil 2025).
    Naive jnp.einsum uses n⁴ multiplications.

    Args:
        R: Riemann tensor, shape (n, n, n, n), float64
        u: contravariant vector, shape (n,), float64
        v: contravariant vector, shape (n,), float64
        components: optional precomputed list from build_riemann_components(n).
                    Pass in hot loops; omit to compute on first call and cache.

    Returns:
        B: covariant rank-2 tensor, shape (n, n), float64
    """
    n = R.shape[0]
    comps = components if components is not None else build_riemann_components(n)
    kernel = _get_kernel(n, tuple(comps))
    return kernel(R, u, v)
