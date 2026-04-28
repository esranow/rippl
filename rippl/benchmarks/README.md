# rippl Benchmark Results

## GPU Benchmark Suite (T4, Colab)

| Benchmark | L2 Error | Status | Time |
|-----------|----------|--------|------|
| Physics sanity | 9.54e-07 (local) / 1.97e+01 (pre-fix) | fixed in commit 82d42fc | — |
| Heat 1D (rippl) | 6.91e-04 | PASS | 116.5s |
| Heat 1D (DeepXDE) | [pending rerun] | — | — |
| Wave 1D causal | 8.77e-01 | FAIL — pre-fix | rerun pending |
| Stokes 1D | u=8.58e-02, p=6.73e-01 | FAIL — pre-fix | rerun pending |
| NTK vs Fixed | 6.80e-03 vs 2.04e-02 | NTK wins 3x | — |

## Known Fix Status
- Laplacian dimension leak: FIXED (commit 82d42fc)
- Derivative dim collision: FIXED (commit 82d42fc)
- Wave and Stokes results require rerun on GPU after fix

## Rerun Instructions
Upload `rippl_benchmarks.ipynb` to Colab T4 and run all cells.
