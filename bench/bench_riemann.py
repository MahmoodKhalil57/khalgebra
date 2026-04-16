"""
Riemann contraction benchmark — khal_riemann_contract vs naive_riemann_contract
Run: python bench/bench_riemann.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import khalgebra as kh

REPS = 50_000

print(f"\n{'═'*72}")
print(f"  RIEMANN CONTRACTION BENCHMARK — khalgebra vs naive")
print(f"  n² mults vs n⁴ naive")
print(f"{'═'*72}\n")
print(f"  {'n':<4} {'naive_μs':>10} {'optimal_μs':>12} {'speedup':>9} {'mults_saved':>12}")
print(f"  {'─'*52}")

for n in [2, 3, 4, 5, 6, 10]:
    R = kh.make_riemann_tensor(n)
    u = kh.make_vec(n, seed=1)
    v = kh.make_vec(n, seed=2)
    comps = kh.build_riemann_components(n)  # precomputed outside hot loop

    # JIT warmup
    kh.naive_riemann_contract(R, u, v).block_until_ready()
    kh.khal_riemann_contract(R, u, v, comps)  # not jitted — Python loop

    t0 = time.perf_counter()
    for _ in range(REPS):
        kh.naive_riemann_contract(R, u, v).block_until_ready()
    t_naive = (time.perf_counter() - t0) / REPS * 1e6

    t0 = time.perf_counter()
    for _ in range(REPS):
        kh.khal_riemann_contract(R, u, v, comps)
    t_opt = (time.perf_counter() - t0) / REPS * 1e6

    reduction = (1 - n**2 / n**4) * 100
    print(f"  {n:<4} {t_naive:>10.3f} {t_opt:>12.3f} {t_naive/t_opt:>9.2f}x {reduction:>10.1f}% fewer mults")

print()
