# Building rocm-libraries (hipBLASLt) and gemm_test on Frontier

## Overview

This document covers building **hipBLASLt** from the `rocm-libraries` repo
(branch `users/pkamd/ornl_ticket`) on Frontier, then compiling and running
`gemm_test` against it.

rocBLAS dispatches GEMM calls through hipBLASLt when `ROCBLAS_USE_HIPBLASLT=1`
is set — the custom hipBLASLt provides optimized Tensile kernels for the ORNL
workload sizes.

| Component       | Source                                                            |
|-----------------|-------------------------------------------------------------------|
| **hipcc**       | `/opt/rocm-7.2.0/bin/hipcc` (system ROCm)                        |
| **HIP runtime** | `/opt/rocm-7.2.0/lib/` (system ROCm)                             |
| **rocBLAS**     | `/opt/rocm-7.2.0/lib/librocblas.so` (system ROCm)                |
| **hipBLASLt**   | `rocm-libraries/install-hipblaslt/` (custom build, v1.4.1)       |

---

## 1. Modules

```bash
module load rocm/7.2.0
module load cmake/3.31.11
module load git-lfs/3.7.1
module load cray-python/3.11.7
```

| Module             | Version  | Purpose                                       |
|--------------------|----------|-----------------------------------------------|
| `rocm`             | 7.2.0    | HIP compiler, runtime, system rocBLAS         |
| `cmake`            | 3.31.11  | Build system                                  |
| `git-lfs`          | 3.7.1    | Git LFS for rocm-libraries repo               |
| `cray-python`      | 3.11.7   | Python ≥3.9 for TensileLite scripts           |

## 2. Python packages (pip install --user)

No modules available for these on Frontier.

```bash
pip install --user invoke simplejson ujson filelock rich
```

| Package      | Purpose                                    |
|--------------|--------------------------------------------|
| `rich`       | Tensile dependency (progress display)      |
| `invoke`     | hipblaslt build system (`tasks.py`)         |
| `simplejson` | TensileLite dependency                     |
| `ujson`      | TensileLite dependency                     |
| `filelock`   | TensileLite dependency                     |

## 3. Local dependency: msgpack-cxx

C++ header-only library. No module available (`module spider msgpack-c` returns
nothing). Build from source:

```bash
cd /lustre/orion/ven114/proj-shared/kswirydo
git clone --depth 1 --branch cpp-6.1.1 https://github.com/msgpack/msgpack-c.git msgpack-cxx
cd msgpack-cxx && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF ..
cmake --build . --target install
```

Installed to: `/lustre/orion/ven114/proj-shared/kswirydo/msgpack-cxx/install/`

## 4. Switch rocm-libraries branch

```bash
cd /lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries
git fetch origin users/pkamd/ornl_ticket
GIT_LFS_SKIP_SMUDGE=1 git checkout -f users/pkamd/ornl_ticket
```

## 5. Configure hipBLASLt

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
- `hipblas-common`: Found from system ROCm 7.2.0.
- `origami`: Built automatically from `rocm-libraries/shared/origami/`.
- `nanobind`: Fetched automatically via CMake FetchContent (slow on Lustre, ~10 min).

## 6. Build & install hipBLASLt

```bash
cmake --build . --target install -j 32
```

Build time: ~25 minutes (including ~5 min Tensile kernel generation, 7003 kernels).

## 7. Post-install: Tensile data symlinks

hipBLASLt installs Tensile kernel files under `lib/hipblaslt/library/gfx90a/`, but
the library searches `lib/hipblaslt/library/` directly. Symlink them up:

```bash
cd $HIPBLASLT_INSTALL/lib/hipblaslt/library/
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

## 8. Build gemm_test

```bash
module load rocm/7.2.0

cd /lustre/orion/ven114/proj-shared/kswirydo/gemm_test
make clean
make
```

The Makefile uses:
- **System rocBLAS** — `-lrocblas` (no custom path)
- **Custom hipBLASLt** — `-L$(HIPBLASLT_PATH)/lib -lhipblaslt`

```makefile
ROCM_PATH      ?= /opt/rocm-7.2.0
HIPBLASLT_PATH ?= /lustre/orion/ven114/proj-shared/kswirydo/rocm-libraries/install-hipblaslt
```

### Verify linked libraries

```bash
ldd gemm_test | grep -E "rocblas|hipblaslt"
```

Expected:
```
librocblas.so.5   => /opt/rocm-7.2.0/lib/librocblas.so.5
libhipblaslt.so.1 => .../rocm-libraries/install-hipblaslt/lib/libhipblaslt.so.1
```

- `rocblas` must point to **system** `/opt/rocm-7.2.0/lib/`
- `hipblaslt` must point to **custom** `install-hipblaslt/lib/`

## 9. Run gemm_test

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

### Override paths (at build time)

```bash
make HIPBLASLT_PATH=/path/to/other/hipblaslt-install
```

---

## Issues Encountered & Workarounds

### hipBLASLt build

1. **Cray PE ar/ranlib incompatible with amdclang objects**: The default
   `ar`/`ranlib` from Cray PE produces `file format not recognized` when
   archiving `.o` files from `amdclang++`.
   Fix: `-DCMAKE_AR=$ROCM_PATH/llvm/bin/llvm-ar -DCMAKE_RANLIB=$ROCM_PATH/llvm/bin/llvm-ranlib`.
2. **No msgpack-cxx module on Frontier**: Fix: build from source (see step 3).
3. **No `invoke` module**: Fix: `pip install --user invoke`.
4. **Slow cmake configure on Lustre**: FetchContent git clones take ~10 min.
5. **Transient Tensile kernel assembly failure**: 1 of 7003 `.o` files missing
   (Lustre race condition). Re-running succeeded.
6. **Tensile data under gfx90a/ subdirectory**: Library searches parent dir.
   Fix: symlink files up one level (see step 7).

### General

7. **git-lfs required**: `module load git-lfs/3.7.1` before git operations.
8. **LFS skip-smudge**: `GIT_LFS_SKIP_SMUDGE=1` for fast checkout.
9. **System Python too old** (3.6.15): Fix: `module load cray-python/3.11.7`.
10. **RPATH vs RUNPATH**: `-Wl,--disable-new-dtags` embeds RPATH (searched
    before `LD_LIBRARY_PATH`).
11. **Custom rocBLAS Tensile data lookup failure**: Custom install puts data
    under `gfx90a/` but library searches parent → silent fallback to reference
    impl (~300x slower). Fix: use system rocBLAS instead.

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

# --- Run ---
# ROCBLAS_USE_HIPBLASLT=1 ./gemm_test --sizes test_sizes.txt
```

---

## rocBLAS build (optional, not used by gemm_test)

A custom rocBLAS was also built but is **not currently used** — system rocBLAS
is used instead. See `../build-rocblas-notes.md` for the full build log.
