import khalgebra as kh
from khalgebra._types import _lcg_sequence


def test_sym22_fixed_input():
    """A=[[1,2],[2,3]], B=[[4,5],[6,7]] → C=[[16,19],[26,31]]"""
    c00, c01, c10, c11 = kh.khal_sym22(1, 2, 3, 4, 5, 6, 7)
    assert float(c00) == 16.0
    assert float(c01) == 19.0
    assert float(c10) == 26.0
    assert float(c11) == 31.0


def test_sym22_50_random_inputs():
    vals = _lcg_sequence(7 * 50, seed=99)
    for i in range(50):
        a, b, c, p, q, r, s = vals[i*7:(i+1)*7]
        ref = kh.naive_sym22(a, b, c, p, q, r, s)
        opt = kh.khal_sym22(a, b, c, p, q, r, s)
        for k in range(4):
            err = abs(float(opt[k]) - float(ref[k]))
            assert err < 1e-13, f"input {i} entry {k}: err={err:.2e}"
