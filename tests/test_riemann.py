import pytest
import jax.numpy as jnp
import khalgebra as kh


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_riemann_correctness(n):
    R = kh.make_riemann_tensor(n)
    u = kh.make_vec(n, seed=1)
    v = kh.make_vec(n, seed=2)

    ref = kh.naive_riemann_contract(R, u, v)
    opt = kh.khal_riemann_contract(R, u, v)

    r_max = float(jnp.max(jnp.abs(R)))
    assert r_max > 0.01, f"n={n}: tensor is degenerate (|R|_max={r_max:.4f})"

    err = kh.max_abs_err(ref, opt)
    assert err < 1e-12, f"n={n}: err={err:.2e}"


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_riemann_precomputed_components(n):
    R = kh.make_riemann_tensor(n)
    u = kh.make_vec(n, seed=1)
    v = kh.make_vec(n, seed=2)
    comps = kh.build_riemann_components(n)

    with_comps    = kh.khal_riemann_contract(R, u, v, comps)
    without_comps = kh.khal_riemann_contract(R, u, v)

    assert kh.max_abs_err(with_comps, without_comps) < 1e-15


def test_riemann_n4_spot_check():
    """Spot-check three specific (b,d) entries for n=4."""
    n = 4
    R = kh.make_riemann_tensor(n)
    u = kh.make_vec(n, seed=1)
    v = kh.make_vec(n, seed=2)

    ref = kh.naive_riemann_contract(R, u, v)
    opt = kh.khal_riemann_contract(R, u, v)

    for b, d in [(0, 1), (2, 3), (1, 3)]:
        err = abs(float(opt[b, d]) - float(ref[b, d]))
        assert err < 1e-12, f"(b,d)=({b},{d}): err={err:.2e}"
