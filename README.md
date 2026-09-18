# GEMM Test

GPU DGEMM / DSYRK benchmark on Frontier (AMD MI250X, gfx90a).

Uses **system rocBLAS** (ROCm 7.2.0) with a **custom-built hipBLASLt** from
`rocm-libraries` (branch `users/pkamd/ornl_ticket`). rocBLAS dispatches to
hipBLASLt for certain GEMM paths — the custom hipBLASLt has optimized Tensile
kernels for ORNL workload sizes.

## Dependencies

| Library     | Version | Source                                            |
|-------------|---------|---------------------------------------------------|
| HIP runtime | 7.2.0   | System (`/opt/rocm-7.2.0`)                        |
| rocBLAS     | system  | System (`/opt/rocm-7.2.0/lib/librocblas.so`)      |
| hipBLASLt   | 1.4.1   | Custom (`rocm-libraries/install-hipblaslt/`)       |

## Quick Start

```bash
# 1. Load the ROCm module
module load rocm/7.2.0

# 2. Build
cd /lustre/orion/ven114/proj-shared/kswirydo/gemm_test
make clean && make

# 3. Verify libraries
ldd gemm_test | grep -E "rocblas|hipblaslt"
# Expected:
#   librocblas.so.5   => /opt/rocm-7.2.0/lib/librocblas.so.5
#   libhipblaslt.so.1 => .../rocm-libraries/install-hipblaslt/lib/libhipblaslt.so.1

# 4. Run on a compute node
salloc -A VEN114 -t 00:30:00 -N 1 -p batch
ROCBLAS_USE_HIPBLASLT=1 ./gemm_test --sizes test_sizes.txt
```

`ROCBLAS_USE_HIPBLASLT=1` tells rocBLAS to dispatch GEMM calls through hipBLASLt,
which has the optimized Tensile kernels for these workload sizes.

## Build Details

Only `rocm/7.2.0` needs to be loaded. The Makefile uses:

- **System rocBLAS** — `-lrocblas` (no custom path, uses system Tensile kernels)
- **Custom hipBLASLt** — `-L$(HIPBLASLT_PATH)/lib -lhipblaslt`

```makefile
ROCM_PATH      ?= /opt/rocm-7.2.0
HIPBLASLT_PATH ?= .../rocm-libraries/install-hipblaslt
```

Override at build time:

```bash
make HIPBLASLT_PATH=/path/to/hipblaslt-install
```

## Running

### Interactive

```bash
module load rocm/7.2.0
salloc -A VEN114 -t 00:30:00 -N 1 -p batch
ROCBLAS_USE_HIPBLASLT=1 ./gemm_test --sizes test_sizes.txt
```

### Batch

```bash
#!/bin/bash
#SBATCH -A VEN114
#SBATCH -J gemm_test
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -p batch

module load rocm/7.2.0
cd /lustre/orion/ven114/proj-shared/kswirydo/gemm_test
ROCBLAS_USE_HIPBLASLT=1 srun -n 1 ./gemm_test --sizes test_sizes.txt
```

### Command-line Options

```
./gemm_test [options]
  --sizes FILE      Input file with test sizes (default: test_sizes.txt)
  --run-syrk 0|1    Run DSYRK tests (default: 1)
  --num-tests N     Number of timed iterations (default: 100)
  --help            Show help
```

## Test Sizes File Format

Each line: `N, K, N, P` (the third value is ignored, kept for compatibility).

```
1146175, 15, 1146175, 15
1146175, 3, 1146175, 3
```

Lines starting with `#` are comments.

## What It Computes

For each size (N, K, P):

- **GEMM**: `C = A^T * B` where A is N×K, B is N×P, C is K×P
- **DSYRK**: `C = A^T * A` where A is N×K, C is K×K (symmetric, lower triangle)

Reports per-operation time (ms) and GFLOPS.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ldd` shows custom rocblas instead of system | Rebuild — Makefile should NOT set `-L` for rocBLAS, only `-lrocblas` |
| Performance ~300x too low (5 GFLOPS vs 1.5 TFLOPS) | Using custom rocBLAS whose Tensile data isn't found. Switch to system rocBLAS |
| `libhipblaslt.so.1: cannot open` | Check `install-hipblaslt/lib/libhipblaslt.so.1` exists |
| Slow first GEMM call | Normal — Tensile lazy-loads kernel libraries on first use |

## NVIDIA Backend

```bash
make BACKEND=nvidia CUDA_PATH=/usr/local/cuda
```

## Build Log

See `build-config-rocm-libraries.md` for the full hipBLASLt build log including
all modules, cmake commands, and workarounds.
