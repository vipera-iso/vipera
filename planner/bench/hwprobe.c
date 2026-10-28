/* hwprobe — measure the machine constants the scheduler assumes.
 *
 * `hw/*.toml` carries `ram_bw_gbs` (streaming bandwidth) and `cpu_gflops` (the
 * compute floor). Both were *estimates*. This measures them:
 *
 *   read   streaming read bandwidth over a buffer much larger than cache
 *   copy   streaming copy bandwidth (read + write)
 *   gemv   GEMV throughput on a matrix small enough to stay in L3, so it is
 *          ALU-bound rather than DRAM-bound
 *
 * gemv is what maps onto `cpu_gflops`: vipera charges
 * `2 * active_params / cpu_gflops` seconds per token for arithmetic, so the
 * constant must be the ALU-limited GEMV rate.
 *
 * Both kernels use several independent accumulators and OpenMP. A single
 * accumulator makes the loop latency-bound (one add per dependency chain), which
 * understates a wide out-of-order core by an order of magnitude — an earlier run
 * of this file reported 4 GB/s where the machine does ~35.
 *
 * Emits JSON on stdout. Build:
 *     gcc -O3 -march=native -fopenmp -o hwprobe bench/hwprobe.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define LANES 8

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static volatile double sink;

/* Streaming read: `LANES` independent accumulators per thread so the FP
 * dependency chain is not the bottleneck. */
static double bench_read(const float * a, size_t n, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double total = 0.0;
        const double t0 = now_s();
        /* Whole number of LANES-wide blocks, so the loop has canonical OpenMP
         * form; the < LANES leftover elements are not timed. */
        const size_t lim = n - (n % LANES);
#ifdef _OPENMP
#pragma omp parallel reduction(+:total)
#endif
        {
            double s[LANES];
            for (int k = 0; k < LANES; ++k) s[k] = 0.0;
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (size_t i = 0; i < lim; i += LANES) {
                for (int k = 0; k < LANES; ++k) {
                    s[k] += a[i + k];
                }
            }
            for (int k = 0; k < LANES; ++k) total += s[k];
        }
        const double dt = now_s() - t0;
        sink = total;
        if (dt < best) best = dt;
    }
    return (double) n * sizeof(float) / best / 1e9;
}

/* Streaming copy: reads n floats and writes n floats; bytes counts both sides. */
static double bench_copy(float * dst, const float * src, size_t n, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        memcpy(dst, src, n * sizeof(float));
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return 2.0 * (double) n * sizeof(float) / best / 1e9;
}

/* y = A x, A (m x n) row-major: 2*m*n flops. LANES accumulators per row. */
static double bench_gemv(const float * A, const float * x, float * y,
                         size_t m, size_t n, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < m; ++i) {
            const float * row = A + i * n;
            float s[LANES];
            for (int k = 0; k < LANES; ++k) s[k] = 0.0f;
            size_t j = 0;
            for (; j + LANES <= n; j += LANES) {
                for (int k = 0; k < LANES; ++k) {
                    s[k] += row[j + k] * x[j + k];
                }
            }
            float acc = 0.0f;
            for (int k = 0; k < LANES; ++k) acc += s[k];
            for (; j < n; ++j) acc += row[j] * x[j];
            y[i] = acc;
        }
        const double dt = now_s() - t0;
        sink = y[0];
        if (dt < best) best = dt;
    }
    return 2.0 * (double) m * (double) n / best / 1e9;
}

static float * alloc_floats(size_t n, double fill) {
    float * p = (float *) malloc(n * sizeof(float));
    if (!p) {
        fprintf(stderr, "hwprobe: out of memory for %zu floats\n", n);
        exit(1);
    }
    for (size_t i = 0; i < n; ++i) p[i] = (float) fill;
    return p;
}

int main(void) {
    /* 256 MiB streaming buffer: far larger than the 8 MiB L3, so this is DRAM. */
    const size_t stream_n = 64u * 1024 * 1024;
    /* 2 MiB matrix (512x1024): stays in L3, so GEMV is ALU-bound. */
    const size_t gm = 512, gn = 1024;

    float * a = alloc_floats(stream_n, 1.0);
    float * b = alloc_floats(stream_n, 0.0);
    float * A = alloc_floats(gm * gn, 0.5);
    float * x = alloc_floats(gn, 0.25);
    float * y = alloc_floats(gm, 0.0);

    int all = 1;
#ifdef _OPENMP
    all = omp_get_max_threads();
#endif

    const double read_bw = bench_read(a, stream_n, 5);
    const double copy_bw = bench_copy(b, a, stream_n, 5);

#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const double gemv_1 = bench_gemv(A, x, y, gm, gn, 100);
#ifdef _OPENMP
    omp_set_num_threads(all);
#endif
    const double gemv_all = bench_gemv(A, x, y, gm, gn, 400);

    printf("{\n");
    printf("  \"threads\": %d,\n", all);
    printf("  \"stream_buffer_mib\": %d,\n", (int) (stream_n * sizeof(float) / (1024 * 1024)));
    printf("  \"gemv_matrix\": [%d, %d],\n", (int) gm, (int) gn);
    printf("  \"read_gbs\": %.2f,\n", read_bw);
    printf("  \"copy_gbs\": %.2f,\n", copy_bw);
    printf("  \"gemv_gflops_single_thread\": %.2f,\n", gemv_1);
    printf("  \"gemv_gflops_all_threads\": %.2f,\n", gemv_all);
    printf("  \"note\": \"read/copy are DRAM streaming (copy counts read+write); gemv is L3-resident and ALU-bound\"\n");
    printf("}\n");

    free(a);
    free(b);
    free(A);
    free(x);
    free(y);
    return 0;
}
