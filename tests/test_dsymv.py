import pytest
import jax.numpy as jnp
import khalgebra as kh


@pytest.mark.parametrize("n", [2, 3, 4, 8, 16, 32, 64, 100, 256])
def test_dsymv_correctness(n):
    A = kh.make_sym_mat(n)
    v = kh.make_vec(n)
    ref = kh.naive_dsymv(A, v)
    opt = kh.khal_dsymv(A, v)
    assert kh.max_abs_err(ref, opt) < 1e-9, f"n={n}: err={kh.max_abs_err(ref, opt):.2e}"


def test_dsymv_n3_hand_verified():
    A = jnp.array([[3.0, 1.5, -0.5],
                   [1.5, 2.0,  0.7],
                   [-0.5, 0.7, 4.0]])
    v = jnp.array([2.0, -1.0, 3.0])

    ref = kh.naive_dsymv(A, v)
    opt = kh.khal_dsymv(A, v)

    assert abs(float(ref[0]) - 3.0) < 1e-10
    assert abs(float(ref[1]) - 3.1) < 1e-10
    assert abs(float(ref[2]) - 10.3) < 1e-10
    assert kh.max_abs_err(ref, opt) < 1e-9
