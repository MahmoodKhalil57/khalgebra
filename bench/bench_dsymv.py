"""
DSYMV benchmark — khal_dsymv vs naive_dsymv
Run: python bench/bench_dsymv.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import jax
import jax.numpy as jnp
import khalgebra as kh

REPS = 200
VS_COUNT = 50

print(f"\n{'═'*72}")
print(f"  DSYMV BENCHMARK — khalgebra vs naive")
print(f"  n(n+1)/2 mults vs n² standard")
print(f"{'═'*72}\n")
print(f"  {'n':<6} {'naive_ms':>10} {'khalgebra_ms':>14} {'speedup':>9} {'mults_saved':>12}")
print(f"  {'─'*56}")

for n in [32, 64, 128, 256, 512, 1024]:
    A = kh.make_sym_mat(n)
    VS = [kh.make_vec(n, seed=i+1) for i in range(VS_COUNT)]

    # JIT warmup
    for v in VS[:3]:
        kh.naive_dsymv(A, v).block_until_ready()
        kh.khal_dsymv(A, v).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(REPS):
        for v in VS:
            kh.naive_dsymv(A, v).block_until_ready()
    naive_ms = (time.perf_counter() - t0) / (REPS * VS_COUNT) * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        for v in VS:
            kh.khal_dsymv(A, v).block_until_ready()
    khal_ms = (time.perf_counter() - t0) / (REPS * VS_COUNT) * 1000

    speedup = naive_ms / khal_ms
    saved_pct = (1 - n*(n+1)/2 / (n*n)) * 100
    print(f"  {n:<6} {naive_ms:>10.4f} {khal_ms:>14.4f} {speedup:>9.3f}x {saved_pct:>10.0f}% fewer mults")

print()
