#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,      \
                    hipGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define ROCBLAS_CHECK(call)                                                    \
    do {                                                                       \
        rocblas_status stat = (call);                                          \
        if (stat != rocblas_status_success) {                                  \
            fprintf(stderr, "rocBLAS error %s:%d: status %d\n", __FILE__,     \
                    __LINE__, (int)stat);                                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct TestSize {
    int N, K, P;
};

static std::vector<TestSize> read_sizes(const char *filename) {
    std::vector<TestSize> sizes;
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        int n1, k, n2, p;
        char c1, c2, c3;
        std::istringstream iss(line);
        if (iss >> n1 >> c1 >> k >> c2 >> n2 >> c3 >> p) {
            sizes.push_back({n1, k, p});
        }
    }
    return sizes;
}

static void fill_random(double *buf, size_t count) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < count; ++i)
        buf[i] = dist(rng);
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --sizes FILE      Input file with test sizes (default: test_sizes.txt)\n");
    fprintf(stderr, "  --run-syrk 0|1    Run SYRK tests (default: 1)\n");
    fprintf(stderr, "  --num-tests N     Number of timed iterations (default: 100)\n");
    fprintf(stderr, "  --help            Show this help\n");
}

int main(int argc, char **argv) {
    const char *sizefile = "test_sizes.txt";
    int run_syrk = 1;
    int timed_iters = 100;
    const int warmup_iters = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizefile = argv[++i];
        } else if (strcmp(argv[i], "--run-syrk") == 0 && i + 1 < argc) {
            run_syrk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num-tests") == 0 && i + 1 < argc) {
            timed_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    auto sizes = read_sizes(sizefile);
    if (sizes.empty()) {
        fprintf(stderr, "No valid test sizes found in %s\n", sizefile);
        return 1;
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    printf("%-12s %-12s %-12s %-8s %-15s %-15s\n",
           "N", "K", "P", "Op", "Time(ms)", "GFLOPS");
    printf("------------------------------------------------------------------------\n");

    for (const auto &sz : sizes) {
        const int N = sz.N;
        const int K = sz.K;
        const int P = sz.P;

        /*
         * GEMM: A is NxK, B is NxP, C = A^T * B => KxP
         * SYRK: A is NxK, C = A^T * A => KxK (symmetric)
         */
        const size_t sizeA = (size_t)N * K;
        const size_t sizeB = (size_t)N * P;
        const size_t sizeC_gemm = (size_t)K * P;
        const size_t sizeC_syrk = (size_t)K * K;

        double *hA = (double *)malloc(sizeA * sizeof(double));
        double *hB = (double *)malloc(sizeB * sizeof(double));
        fill_random(hA, sizeA);
        fill_random(hB, sizeB);

        double *dA, *dB, *dC_gemm, *dC_syrk;
        HIP_CHECK(hipMalloc(&dA, sizeA * sizeof(double)));
        HIP_CHECK(hipMalloc(&dB, sizeB * sizeof(double)));
        HIP_CHECK(hipMalloc(&dC_gemm, sizeC_gemm * sizeof(double)));
        HIP_CHECK(hipMalloc(&dC_syrk, sizeC_syrk * sizeof(double)));

        HIP_CHECK(hipMemcpy(dA, hA, sizeA * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, hB, sizeB * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dC_gemm, 0, sizeC_gemm * sizeof(double)));
        HIP_CHECK(hipMemset(dC_syrk, 0, sizeC_syrk * sizeof(double)));

        const double alpha = 1.0;
        const double beta  = 0.0;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // ============ GEMM: C = A^T * B ============
        for (int i = 0; i < warmup_iters; ++i) {
            ROCBLAS_CHECK(rocblas_dgemm(handle,
                                        rocblas_operation_transpose,
                                        rocblas_operation_none,
                                        K, P, N,
                                        &alpha,
                                        dA, N,
                                        dB, N,
                                        &beta,
                                        dC_gemm, K));
        }
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < timed_iters; ++i) {
            ROCBLAS_CHECK(rocblas_dgemm(handle,
                                        rocblas_operation_transpose,
                                        rocblas_operation_none,
                                        K, P, N,
                                        &alpha,
                                        dA, N,
                                        dB, N,
                                        &beta,
                                        dC_gemm, K));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float total_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        double per_gemm_ms = (double)total_ms / timed_iters;

        double flops_gemm = 2.0 * (double)K * (double)P * (double)N;
        double gflops_gemm = (flops_gemm / (per_gemm_ms * 1e-3)) * 1e-9;

        printf("%-12d %-12d %-12d %-8s %-15.4f %-15.2f\n",
               N, K, P, "GEMM", per_gemm_ms, gflops_gemm);

        // ============ SYRK: C = A^T * A ============
        if (run_syrk) {
            for (int i = 0; i < warmup_iters; ++i) {
                ROCBLAS_CHECK(rocblas_dsyrk(handle,
                                            rocblas_fill_lower,
                                            rocblas_operation_transpose,
                                            K, N,
                                            &alpha,
                                            dA, N,
                                            &beta,
                                            dC_syrk, K));
            }
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipEventRecord(start));
            for (int i = 0; i < timed_iters; ++i) {
                ROCBLAS_CHECK(rocblas_dsyrk(handle,
                                            rocblas_fill_lower,
                                            rocblas_operation_transpose,
                                            K, N,
                                            &alpha,
                                            dA, N,
                                            &beta,
                                            dC_syrk, K));
            }
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));

            HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
            double per_syrk_ms = (double)total_ms / timed_iters;

            double flops_syrk = 2.0 * (double)K * (double)K * (double)N;
            double gflops_syrk = (flops_syrk / (per_syrk_ms * 1e-3)) * 1e-9;

            printf("%-12d %-12d %-12d %-8s %-15.4f %-15.2f\n",
                   N, K, P, "SYRK", per_syrk_ms, gflops_syrk);
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dC_gemm));
        HIP_CHECK(hipFree(dC_syrk));
        free(hA);
        free(hB);
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    return 0;
}
