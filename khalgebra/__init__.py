"""
khalgebra — Khalil Optimal Bilinear Algorithms
Author: Mahmood Khalil (2025)
"""

import jax
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
__author__ = "Mahmood Khalil"

from khalgebra.dsymv import khal_dsymv, naive_dsymv
from khalgebra.dsymm import khal_dsymm, naive_dsymm
from khalgebra.sym22 import khal_sym22, naive_sym22
from khalgebra.riemann import (
    khal_riemann_contract,
    naive_riemann_contract,
    build_riemann_components,
)

from khalgebra._types import (
    make_sym_mat,
    make_gen_mat,
    make_vec,
    make_riemann_tensor,
    max_abs_err,
)

__all__ = [
    "khal_dsymv", "naive_dsymv",
    "khal_dsymm", "naive_dsymm",
    "khal_sym22", "naive_sym22",
    "khal_riemann_contract", "naive_riemann_contract", "build_riemann_components",
    "make_sym_mat", "make_gen_mat", "make_vec", "make_riemann_tensor", "max_abs_err",
    "__version__", "__author__",
]
