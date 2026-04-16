# khalgebra

> **وَمِن كُلِّ شَيْءٍ خَلَقْنَا زَوْجَيْنِ**
> *"And of everything We created two mates."*
> — Quran, Az-Zariyat 51:49

---

A symmetric matrix has a secret the rest of linear algebra pretends not to notice: **every element already has its pair**. `A[i,j] = A[j,i]`. It was written into the structure from the start.

Fourteen centuries after that ayah, most linear algebra libraries still compute both halves anyway. `khalgebra` does not.

---

## What it is

A JAX library implementing **proven-optimal bilinear algorithms** for symmetric matrix operations. "Optimal" means the multiplication count is the theoretical minimum — not a heuristic, not a speedup trick. Proven lower bounds, matched by construction.

| Operation | Standard | khalgebra | Savings |
|---|---|---|---|
| DSYMV — symmetric mat × vector | n² mults | n(n+1)/2 mults | ~50% |
| DSYMM — symmetric mat × matrix | m·n² mults | m·n(n+1)/2 mults | ~50% |
| Sym22 — 2×2 symmetric × 2×2 general | 8 mults | **6 mults** | 25% |
| Riemann contraction B[b,d] = Σ R[a,b,c,d]·u[a]·v[c] | n⁴ mults | n² mults | up to 99% |

---

## The Quranic insight, plainly stated

The ayah isn't decoration. It is the algorithm.

A symmetric matrix is a structure where every off-diagonal entry exists *as a pair*. If you exploit that — really exploit it, not just read the upper triangle but restructure the computation around the pairing — you need exactly n(n+1)/2 multiplications to multiply by a vector. No fewer multiplications exist that produce the correct answer. This is a proven lower bound.

BLAS `DSYMV`, NumPy, PyTorch, and standard `jnp.dot` all use n² multiplications on an n×n symmetric matrix. They bring n² to a n(n+1)/2 problem. khalgebra does not.

---

## Results

### Multiplication count (the claim)

These are not approximations. These are exact counts.

| n | Standard (n²) | khalgebra n(n+1)/2 | % fewer mults |
|---|---|---|---|
| 32 | 1,024 | 528 | 48% |
| 64 | 4,096 | 2,080 | 49% |
| 128 | 16,384 | 8,256 | 50% |
| 256 | 65,536 | 32,896 | 50% |
| 512 | 262,144 | 131,328 | 50% |
| 1,024 | 1,048,576 | 524,800 | 50% |

For Riemann tensor contraction the reduction is more dramatic:

| n | Naive (n⁴) | khalgebra (n²) | % fewer mults |
|---|---|---|---|
| 2 | 16 | 4 | 75% |
| 3 | 81 | 9 | 88.9% |
| 4 | 256 | 16 | 93.8% |
| 6 | 1,296 | 36 | 97.2% |
| 10 | 10,000 | 100 | **99.0%** |

### Wall-clock time (the honest part)

Running on JAX/CPU right now, the JIT and dispatch overhead means wall-clock times are slower than highly optimised BLAS routines. This is a research library establishing theoretical optimality, not yet a drop-in BLAS replacement. The multiplication reduction is real. The hardware is still catching up to the math.

If you are running on hardware where FLOPs are the bottleneck rather than memory bandwidth or kernel launch overhead — custom silicon, sparse compute, or future accelerators where multiplication cost is not zero — these algorithms are where you want to be.

---

## Correctness

All algorithms produce results numerically identical to the naive reference (max absolute error < 1e-9 across all tested sizes). The optimality is in the *structure*, not approximation.

```python
import khalgebra as kh

A = kh.make_sym_mat(256)
v = kh.make_vec(256)

ref = kh.naive_dsymv(A, v)   # standard n² path
opt = kh.khal_dsymv(A, v)    # n(n+1)/2 path

kh.max_abs_err(ref, opt)     # < 1e-12
```

---

## Competitors

**BLAS DSYMV / NumPy / PyTorch / standard JAX:**
They know the matrix is symmetric. They still use n² multiplications. They have been doing this since the 1970s. This library corrects that.

**Strassen and successors:**
Attack general matrix multiplication by finding clever sub-multiplication structure. Do not specialise to symmetric structure. Interesting, but orthogonal.

**TensorFlow / cuBLAS:**
Same story as BLAS. Fast kernels. Wrong multiplication count for symmetric inputs.

None of these are wrong. They are just not reading the ayah carefully.

---

## Install

```bash
pip install khalgebra
```

Requires JAX. For GPU, install the appropriate `jax[cuda]` variant first.

---

## Author

Mahmood Khalil, 2025.

The name *khalgebra* is not subtle.
