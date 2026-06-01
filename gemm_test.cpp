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

int main(int argc, char **argv) {
    const char *sizefile = (argc > 1) ? argv[1] : "test_sizes.txt";
    const int warmup_iters = 10;
    const int timed_iters  = 100;

    auto sizes = read_sizes(sizefile);
    if (sizes.empty()) {
        fprintf(stderr, "No valid test sizes found in %s\n", sizefile);
        return 1;
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    printf("%-12s %-12s %-12s %-15s %-15s\n",
           "N", "K", "P", "Time(ms)", "GFLOPS");
    printf("--------------------------------------------------------------\n");

    for (const auto &sz : sizes) {
        const int N = sz.N;
        const int K = sz.K;
        const int P = sz.P;

        /*
         * A is NxK (col-major, lda=N)
         * B is NxP (col-major, ldb=N)
         * C = A^T * B  =>  KxP (col-major, ldc=K)
         *
         * rocblas_dgemm params (col-major convention):
         *   transA = transpose,  transB = none
         *   m = K,  n = P,  k = N
         */
        const size_t sizeA = (size_t)N * K;
        const size_t sizeB = (size_t)N * P;
        const size_t sizeC = (size_t)K * P;

        double *hA = (double *)malloc(sizeA * sizeof(double));
        double *hB = (double *)malloc(sizeB * sizeof(double));
        fill_random(hA, sizeA);
        fill_random(hB, sizeB);

        double *dA, *dB, *dC;
        HIP_CHECK(hipMalloc(&dA, sizeA * sizeof(double)));
        HIP_CHECK(hipMalloc(&dB, sizeB * sizeof(double)));
        HIP_CHECK(hipMalloc(&dC, sizeC * sizeof(double)));

        HIP_CHECK(hipMemcpy(dA, hA, sizeA * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, hB, sizeB * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dC, 0, sizeC * sizeof(double)));

        const double alpha = 1.0;
        const double beta  = 0.0;

        for (int i = 0; i < warmup_iters; ++i) {
            ROCBLAS_CHECK(rocblas_dgemm(handle,
                                        rocblas_operation_transpose,
                                        rocblas_operation_none,
                                        K, P, N,
                                        &alpha,
                                        dA, N,
                                        dB, N,
                                        &beta,
                                        dC, K));
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

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
                                        dC, K));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float total_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        double per_gemm_ms = (double)total_ms / timed_iters;

        double flops = 2.0 * (double)K * (double)P * (double)N;
        double gflops = (flops / (per_gemm_ms * 1e-3)) * 1e-9;

        printf("%-12d %-12d %-12d %-15.4f %-15.2f\n",
               N, K, P, per_gemm_ms, gflops);

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dC));
        free(hA);
        free(hB);
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    return 0;
}
