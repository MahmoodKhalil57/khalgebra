"""
DSYMM benchmark — khal_dsymm vs naive_dsymm
Run: python bench/bench_dsymm.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import khalgebra as kh

print(f"\n{'═'*60}")
print(f"  DSYMM BENCHMARK — khalgebra vs naive (n×n × n×n)")
print(f"  m·n(n+1)/2 mults vs m·n² standard")
print(f"{'═'*60}\n")
print(f"  {'n':<6} {'naive_ms':>10} {'khalgebra_ms':>14} {'speedup':>9}")
print(f"  {'─'*42}")

for n in [16, 32, 64, 128]:
    A = kh.make_sym_mat(n)
    B = kh.make_gen_mat(n, n)
    REPS = 500 if n <= 32 else 100 if n <= 64 else 20

    kh.naive_dsymm(A, B).block_until_ready()
    kh.khal_dsymm(A, B).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(REPS):
        kh.naive_dsymm(A, B).block_until_ready()
    naive_ms = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        kh.khal_dsymm(A, B).block_until_ready()
    khal_ms = (time.perf_counter() - t0) / REPS * 1000

    print(f"  {n:<6} {naive_ms:>10.3f} {khal_ms:>14.3f} {naive_ms/khal_ms:>9.3f}x")

print()
