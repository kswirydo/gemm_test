# GEMM Test

A GPU DGEMM (double-precision matrix multiplication) benchmark supporting AMD and NVIDIA backends.

Computes `C = A^T * B` where:
- A is NxK
- B is NxP  
- C is KxP

## Requirements

**AMD (ROCm):**
- ROCm with HIP and rocBLAS

**NVIDIA (CUDA):**
- CUDA Toolkit with cuBLAS

## Build

```bash
# AMD (default)
make

# NVIDIA
make BACKEND=nvidia

# Custom paths
make ROCM_PATH=/opt/rocm
make BACKEND=nvidia CUDA_PATH=/usr/local/cuda
```

## Run

```bash
# Using make
make run

# Direct execution
./gemm_test test_sizes.txt
./gemm_test /path/to/sizes.txt
```

## Test Sizes File Format

Each line specifies matrix dimensions as `N, K, N, P`:

```
1024, 512, 1024, 256
2048, 1024, 2048, 512
```

Lines starting with `#` are comments.

## Clean

```bash
make clean
```
