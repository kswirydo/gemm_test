# gemm_test: Build Configuration with rocm-libraries

## Overview

The `gemm_test` binary uses **system rocBLAS** (from ROCm 7.2.0) with a
**custom-built hipBLASLt** from the `rocm-libraries` project (branch
`users/pkamd/ornl_ticket`). rocBLAS dispatches to hipBLASLt internally for
certain GEMM paths — the custom hipBLASLt provides optimized Tensile kernels
for the ORNL workload sizes.

## Component Sources

| Component       | Source                                                            |
|-----------------|-------------------------------------------------------------------|
| **hipcc**       | `/opt/rocm-7.2.0/bin/hipcc` (system ROCm)                        |
| **HIP runtime** | `/opt/rocm-7.2.0/lib/` (system ROCm)                             |
| **rocBLAS**     | `/opt/rocm-7.2.0/lib/librocblas.so` (system ROCm)                |
| **hipBLASLt**   | `rocm-libraries/install-hipblaslt/` (custom build, v1.4.1)       |

**Why system rocBLAS?** The system rocBLAS already has properly installed Tensile
kernel data at `/opt/rocm-7.2.0/lib/rocblas/library/`. Using a custom rocBLAS build
caused Tensile data lookup failures (data installed under a `gfx90a/` subdirectory
but rocBLAS searches the parent directory), resulting in a silent fallback to a
reference implementation (~300x slower).

## Makefile Variables

```makefile
ROCM_PATH      ?= /opt/rocm-7.2.0
HIPBLASLT_PATH ?= /lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/install-hipblaslt
```

Both paths are overridable via environment or command-line variables.

## How to Build gemm_test

```bash
module load rocm/7.2.0

cd /lustre/orion/ven114/proj-shared/kswirydo/gemm_test
make clean
make
```

## How to Verify Linked Libraries

```bash
ldd gemm_test | grep -E "rocblas|hipblaslt"
```

Expected output:
```
librocblas.so.5   => /opt/rocm-7.2.0/lib/librocblas.so.5
libhipblaslt.so.1 => .../rocm-libraries/install-hipblaslt/lib/libhipblaslt.so.1
```

- `rocblas` must point to **system** `/opt/rocm-7.2.0/lib/`
- `hipblaslt` must point to **custom** `install-hipblaslt/lib/`

## How to Run

### Prerequisites

You must have `rocm/7.2.0` loaded (for the HIP runtime and system rocBLAS).

### Interactive (on a compute node)

```bash
module load rocm/7.2.0
salloc -A VEN114 -t 00:30:00 -N 1 -p batch

cd /lustre/orion/ven114/proj-shared/kswirydo/gemm_test
ROCBLAS_USE_HIPBLASLT=1 ./gemm_test --sizes test_sizes.txt
```

`ROCBLAS_USE_HIPBLASLT=1` tells rocBLAS to dispatch GEMM calls through hipBLASLt,
which has the optimized Tensile kernels for these workload sizes. Without it,
rocBLAS uses its own (slower) Tensile kernels.

### Batch job

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

### Override Paths (at build time)

```bash
make HIPBLASLT_PATH=/path/to/other/hipblaslt-install
```

---

## Building hipBLASLt from rocm-libraries

**Date:** 2026-09-18
**Build time:** ~25 minutes (including ~5 min Tensile kernel generation)

### Modules

```bash
module load rocm/7.2.0
module load cmake/3.31.11
module load git-lfs/3.7.1
module load cray-python/3.11.7
```

### Python packages (pip install --user)

```bash
pip install --user invoke simplejson ujson filelock rich
```

No modules available for `invoke` or `msgpack-cxx` on Frontier (`module spider`
returns no matches).

### Local dependency: msgpack-cxx (C++ header-only library)

No module available. Built from source:

```bash
cd /lustre/orion/ven114/proj-shared/kswirydo
git clone --depth 1 --branch cpp-6.1.1 https://github.com/msgpack/msgpack-c.git msgpack-cxx
cd msgpack-cxx && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF ..
cmake --build . --target install
```

Installed to: `/lustre/orion/ven114/proj-shared/kswirydo/msgpack-cxx/install/`

### Configure

```bash
export HIPBLASLT_SRC=/lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/projects/hipblaslt
export HIPBLASLT_BUILD=/lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/build-hipblaslt
export HIPBLASLT_INSTALL=/lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/install-hipblaslt
export MSGPACK_DIR=/lustre/orion/ven114/proj-shared/kswirydo/msgpack-cxx/install
export ROCM_PATH=/opt/rocm-7.2.0

mkdir -p $HIPBLASLT_BUILD && cd $HIPBLASLT_BUILD

cmake \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ \
  -DCMAKE_C_COMPILER=$ROCM_PATH/bin/amdclang \
  -DCMAKE_AR=$ROCM_PATH/llvm/bin/llvm-ar \
  -DCMAKE_RANLIB=$ROCM_PATH/llvm/bin/llvm-ranlib \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx90a \
  -DHIPBLASLT_ENABLE_CLIENT=OFF \
  -DHIPBLASLT_ENABLE_MARKER=OFF \
  -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
  -DHIPBLASLT_ENABLE_YAML=OFF \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH;$ROCM_PATH/hip;$MSGPACK_DIR/lib64/cmake" \
  -DCMAKE_MODULE_PATH="$ROCM_PATH/hip/cmake" \
  -DCMAKE_INSTALL_PREFIX=$HIPBLASLT_INSTALL \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DROCM_PATH=$ROCM_PATH \
  $HIPBLASLT_SRC
```

Key CMake options explained:
- `-DCMAKE_AR` / `-DCMAKE_RANLIB`: **Must** use `llvm-ar`/`llvm-ranlib` from ROCm.
  The Cray PE `ranlib` (`/opt/cray/pe/cce/18.0.1/binutils/...`) cannot read object
  files produced by `amdclang++`, causing `file format not recognized` errors.
- `-DHIPBLASLT_ENABLE_CLIENT=OFF`: Skip test/benchmark clients (avoids LAPACK/BLAS deps).
- `-DHIPBLASLT_ENABLE_MARKER=OFF`: Skip roctx marker support (avoids rocTracer dep).
- `-DHIPBLASLT_ENABLE_ROCROLLER=OFF`: Skip rocRoller (not needed for library-only build).
- `-DHIPBLASLT_ENABLE_YAML=OFF`: Use msgpack format (default), avoids LLVM dependency.
- `hipblas-common`: Found from system ROCm 7.2.0 (`/opt/rocm-7.2.0/lib/cmake/hipblas-common/`).
- `origami`: Built automatically from `rocm-libraries/shared/origami/` via `add_subdirectory`.
- `nanobind`: Fetched automatically via CMake FetchContent (git clone during configure).

### Build & install

```bash
cmake --build . --target install -j 32
```

### Post-install: Tensile data symlinks

hipBLASLt installs Tensile kernel files under `lib/hipblaslt/library/gfx90a/`, but
the library searches `lib/hipblaslt/library/` directly. Symlink them up:

```bash
cd /lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/install-hipblaslt/lib/hipblaslt/library/
for f in $(pwd)/gfx90a/*; do ln -sf "$f" .; done
```

### Install contents

```
install-hipblaslt/
├── include/hipblaslt/             # Public headers
├── lib/
│   ├── libhipblaslt.so.1.4       # Shared library (v1.4.1)
│   ├── libtensilelite-host.a     # TensileLite host static library
│   ├── cmake/                    # CMake package config
│   └── hipblaslt/library/
│       ├── gfx90a/               # 202 Tensile kernel files (.co, .dat.zlib)
│       └── *.co, *.dat.zlib      # Symlinks to gfx90a/ (required!)
└── share/
```

---

## Building rocBLAS from rocm-libraries (optional)

A custom rocBLAS was also built but is **not currently used** by gemm_test (system
rocBLAS is used instead). The custom build is at `rocm-libraries/install-rocblas/`.
See `build-rocblas-notes.md` in the parent directory for the full build log.

---

## Issues Encountered & Workarounds

### hipBLASLt build

1. **Cray PE ar/ranlib incompatible with amdclang objects**: The default
   `ar`/`ranlib` from Cray PE (`cce/18.0.1/binutils`) produces
   `file format not recognized` when archiving `.o` files compiled by `amdclang++`.
   Fix: `-DCMAKE_AR=$ROCM_PATH/llvm/bin/llvm-ar -DCMAKE_RANLIB=$ROCM_PATH/llvm/bin/llvm-ranlib`.
2. **No msgpack-cxx module on Frontier**: Need C++ header-only library. No module
   available (`module spider msgpack-c` returns nothing). Fix: build from
   `github.com/msgpack/msgpack-c` branch `cpp-6.1.1`.
3. **No `invoke` module**: Required by hipblaslt's `tasks.py`. Fix: `pip install --user invoke`.
4. **Slow cmake configure on Lustre**: FetchContent downloads `nanobind` via git clone
   during configure. This takes ~10 min on Lustre due to metadata overhead.
5. **Transient Tensile kernel assembly failure**: 1 of 7003 assembly `.o` files was
   missing after the first build attempt (Lustre race condition). Re-running succeeded.
6. **Tensile data under gfx90a/ subdirectory**: hipBLASLt installs kernel files under
   `lib/hipblaslt/library/gfx90a/` but searches `lib/hipblaslt/library/`. Fix: symlink
   files up one level (see post-install step above).

### General

7. **git-lfs required**: `module load git-lfs/3.7.1` before any git operations in
   the rocm-libraries repo.
8. **LFS skip-smudge**: Use `GIT_LFS_SKIP_SMUDGE=1` for fast checkout.
9. **System Python too old**: `/usr/bin/python3` is 3.6.15; need ≥3.9.
   Fix: `module load cray-python/3.11.7`.
10. **RPATH vs RUNPATH**: Use `-Wl,--disable-new-dtags` in the gemm_test Makefile to
    embed RPATH (searched before `LD_LIBRARY_PATH`) instead of RUNPATH.
11. **Custom rocBLAS Tensile data lookup failure**: The custom rocBLAS install puts
    Tensile data under `gfx90a/` subdirectory but the library searches the parent.
    This caused silent fallback to a reference implementation (~300x slower, e.g.
    5 GFLOPS instead of 1.5 TFLOPS). Fix: use system rocBLAS instead.

---

## Module Summary

| Module             | Version  | Purpose                                       |
|--------------------|----------|-----------------------------------------------|
| `rocm`             | 7.2.0    | HIP compiler, runtime, system rocBLAS         |
| `cmake`            | 3.31.11  | Build system (hipBLASLt build only)            |
| `git-lfs`          | 3.7.1    | Git LFS for rocm-libraries (build only)        |
| `cray-python`      | 3.11.7   | Python ≥3.9 for TensileLite (build only)       |

For running gemm_test, only `rocm/7.2.0` is needed.

### pip install --user (no module available)

| Package      | Purpose                                    |
|--------------|--------------------------------------------|
| `rich`       | Tensile dependency (progress display)      |
| `invoke`     | hipblaslt build system (`tasks.py`)         |
| `simplejson` | TensileLite dependency                     |
| `ujson`      | TensileLite dependency                     |
| `filelock`   | TensileLite dependency                     |

### Local build (no module available)

| Dependency   | Path                                                      |
|--------------|-----------------------------------------------------------|
| msgpack-cxx  | `/lustre/orion/ven114/proj-shared/kswirydo/msgpack-cxx/install/` |

---

## Quick Reproduction Script

```bash
#!/bin/bash
# Build hipBLASLt from rocm-libraries and compile gemm_test on Frontier

# --- Modules ---
module load rocm/7.2.0 cmake/3.31.11 git-lfs/3.7.1 cray-python/3.11.7
pip install --user rich invoke simplejson ujson filelock

REPO=/lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries
ROCM=/opt/rocm-7.2.0
MSGPACK=/lustre/orion/ven114/proj-shared/kswirydo/msgpack-cxx/install

# --- hipBLASLt ---
mkdir -p $REPO/build-hipblaslt && cd $REPO/build-hipblaslt
cmake \
  -DCMAKE_CXX_COMPILER=$ROCM/bin/amdclang++ \
  -DCMAKE_C_COMPILER=$ROCM/bin/amdclang \
  -DCMAKE_AR=$ROCM/llvm/bin/llvm-ar \
  -DCMAKE_RANLIB=$ROCM/llvm/bin/llvm-ranlib \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx90a \
  -DHIPBLASLT_ENABLE_CLIENT=OFF -DHIPBLASLT_ENABLE_MARKER=OFF \
  -DHIPBLASLT_ENABLE_ROCROLLER=OFF -DHIPBLASLT_ENABLE_YAML=OFF \
  -DCMAKE_PREFIX_PATH="$ROCM;$ROCM/hip;$MSGPACK/lib64/cmake" \
  -DCMAKE_MODULE_PATH="$ROCM/hip/cmake" \
  -DCMAKE_INSTALL_PREFIX=$REPO/install-hipblaslt \
  -DCMAKE_INSTALL_LIBDIR=lib -DROCM_PATH=$ROCM \
  $REPO/projects/hipblaslt
cmake --build . --target install -j 32

# --- Tensile data symlinks ---
cd $REPO/install-hipblaslt/lib/hipblaslt/library/
for f in $(pwd)/gfx90a/*; do ln -sf "$f" .; done

# --- gemm_test ---
cd /lustre/orion/ven114/proj-shared/kswirydo/gemm_test
make clean && make
ldd gemm_test | grep -E "rocblas|hipblaslt"
```
