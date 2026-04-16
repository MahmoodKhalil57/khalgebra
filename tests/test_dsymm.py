import pytest
import khalgebra as kh


@pytest.mark.parametrize("n,m", [
    (2, 1), (3, 3), (8, 8),
    (4, 4), (16, 1), (32, 8),
    (64, 8), (128, 8),
])
def test_dsymm_correctness(n, m):
    A = kh.make_sym_mat(n)
    B = kh.make_gen_mat(n, m)
    ref = kh.naive_dsymm(A, B)
    opt = kh.khal_dsymm(A, B)
    err = kh.max_abs_err(ref, opt)
    assert err < 1e-9, f"n={n} m={m}: err={err:.2e}"
