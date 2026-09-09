/* bench_main.c — bin/bench_mpi:MPI 计时 harness(冻结文件,不得修改)
 *
 * 用法:
 *   bin/bench_mpi --seed <N> --tier <id:B,H,S,D> [--tier ...] --out-dir <dir>
 *                 [--ref-dir <dir>] [--runs R] [--warmup W] [--tol T]
 *
 * 计时协议(冻结):
 *   每档:datagen(不计时)→ warmup W 次 → R 次计时
 *   (每次:Barrier → MPI_Wtime 夹住 attention_forward_mpi →
 *    Allreduce MAX 取最慢 rank)→ 取 R 次中位数。
 *   并行度硬校验:1 ≤ np ≤ 96 且 np × omp_threads ≤ 192。
 *
 * 输出(rank 0):
 *   <out>/<id>/output.bin  fp32 输出(ATTBIN01 头)
 *   <out>/<id>.json        逐档计时 + 自检结果(results.h 格式)
 * 退出码:0 = 全部通过(或未启用自检);1 = 任一档失败;2 = 参数/协议错。
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "attention.h"
#include "data.h"
#include "harness_args.h"
#include "io_util.h"
#include "results.h"
#include "sha256.h"
#include "verify.h"

#define MAX_RANKS 96
#define MAX_RANKS_TIMES_THREADS 192

static double median_of(double *v, int n)
{
    for (int i = 1; i < n; i++) { /* 插入排序,n ≤ 32 */
        const double x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char **argv)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        return 2;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- 并行度硬校验(冻结边界,越界即协议错误) ---- */
    const int nthreads = omp_get_max_threads();
    int bad = (size < 1 || size > MAX_RANKS ||
               (long)size * nthreads > MAX_RANKS_TIMES_THREADS);
    if (bad) {
        if (rank == 0)
            fprintf(stderr,
                    "bench_mpi: parallelism limit violated: np=%d threads=%d "
                    "(require 1<=np<=%d, np*threads<=%d)\n",
                    size, nthreads, MAX_RANKS, MAX_RANKS_TIMES_THREADS);
        MPI_Finalize();
        return 2;
    }

    /* ---- CLI(各 rank 参数一致,由同一 mpirun 传入) ---- */
    unsigned long long seed = 0;
    const char *out_dir = NULL, *ref_dir = NULL;
    int runs = 5, warmup = 1;
    double tol = 0.02;
    tier_spec_t tiers[16];
    int n_tiers = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--tier") && i + 1 < argc) {
            if (n_tiers >= 16 ||
                parse_tier_spec(argv[++i], &tiers[n_tiers]) != 0) {
                if (rank == 0)
                    fprintf(stderr, "bench_mpi: bad --tier spec\n");
                MPI_Finalize();
                return 2;
            }
            n_tiers++;
        } else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (!strcmp(argv[i], "--ref-dir") && i + 1 < argc) {
            ref_dir = argv[++i];
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--tol") && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else {
            if (rank == 0)
                fprintf(stderr, "bench_mpi: unknown arg %s\n", argv[i]);
            MPI_Finalize();
            return 2;
        }
    }
    if (!out_dir || n_tiers == 0 || seed == 0) {
        if (rank == 0)
            fprintf(stderr, "usage: bench_mpi --seed N --tier id:B,H,S,D "
                            "[--tier ...] --out-dir DIR [--ref-dir D] "
                            "[--runs R] [--warmup W] [--tol T]\n");
        MPI_Finalize();
        return 2;
    }
    if (runs < 1) runs = 1;
    if (runs > 32) runs = 32;
    if (warmup < 0) warmup = 0;
    if (rank == 0)
        mkdir(out_dir, 0755);

    int any_fail = 0;

    for (int t = 0; t < n_tiers; t++) {
        const AttDims *d = &tiers[t].dims;
        const long n = att_numel(d);
        float *Q = malloc((size_t)n * 4), *K = malloc((size_t)n * 4),
              *V = malloc((size_t)n * 4), *O = malloc((size_t)n * 4);
        if (!Q || !K || !V || !O) {
            fprintf(stderr, "bench_mpi: alloc failed (rank %d)\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* 输入生成:每 rank 独立生成相同内容(不计时) */
        if (data_generate(seed, tiers[t].id, d, Q, K, V) != ATT_OK) {
            if (rank == 0)
                fprintf(stderr, "bench_mpi: datagen failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* warmup(吸收一次性成本;跨调用 scratch 合法) */
        for (int w = 0; w < warmup; w++) {
            AttStatus st = attention_forward_mpi(Q, K, V, d, MPI_COMM_WORLD, O);
            if (st != ATT_OK) {
                if (rank == 0)
                    fprintf(stderr, "bench_mpi: warmup failed: %s\n",
                            att_status_str(st));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        /* 计时:R 次,每次取最慢 rank,再取中位 */
        double times[32];
        for (int r = 0; r < runs; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            AttStatus st = attention_forward_mpi(Q, K, V, d, MPI_COMM_WORLD, O);
            const double t1 = MPI_Wtime();
            if (st != ATT_OK) {
                if (rank == 0)
                    fprintf(stderr, "bench_mpi: kernel failed: %s\n",
                            att_status_str(st));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            double local = t1 - t0, tmax = 0.0;
            MPI_Allreduce(&local, &tmax, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            times[r] = tmax;
        }
        const double med = median_of(times, runs);
        const double gf = (double)att_flops(d) / med / 1e9;

        /* rank 0 落盘 + 自检 */
        double rel_err = 0.0,mdiag = 0.0;
        int has_check = 0, pass = 0;
        char hex[65] = "n/a";
        char oname[256];
        snprintf(oname, sizeof(oname), "%.31s/output.bin", tiers[t].id);

        if (rank == 0) {
            char tdir[4096], opath[4096];
            snprintf(tdir, sizeof(tdir), "%.2040s/%.31s", out_dir, tiers[t].id);
            mkdir(tdir, 0755);
            snprintf(opath, sizeof(opath), "%.2040s/%.2040s", tdir, "output.bin");
            if (io_write_tensor(opath, d, ATT_DTYPE_F32, O) != 0) {
                fprintf(stderr, "bench_mpi: write failed: %s\n", opath);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            sha256_file(opath, hex);

            if (ref_dir) {
                char rpath[4096];
                snprintf(rpath, sizeof(rpath), "%.2040s/%.31s_output.bin",
                         ref_dir, tiers[t].id);
                att_bin_hdr_t rh;
                double *R = NULL;
                if (io_read_tensor(rpath, &rh, (void **)&R) != 0) {
                    fprintf(stderr, "bench_mpi: cannot read reference %s\n",
                            rpath);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (rh.B != d->B || rh.H != d->H || rh.S != d->S ||
                    rh.D != d->D || rh.dtype != ATT_DTYPE_F64) {
                    fprintf(stderr, "bench_mpi: reference shape mismatch "
                                    "for %s\n", tiers[t].id);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                has_check = 1;
                if (verify_rel_err(O, R, n, &rel_err, &mdiag) != ATT_OK)
                    MPI_Abort(MPI_COMM_WORLD, 1);
                free(R);
                pass = rel_err <= tol;
                if (!pass)
                    any_fail = 1;
            }

            tier_result_t res;
            memset(&res, 0, sizeof(res));
            res.tier_id = tiers[t].id;
            res.dims = *d;
            res.flops = att_flops(d);
            res.np = size;
            res.omp_threads = nthreads;
            res.warmup = warmup;
            res.repeats = runs;
            res.times_s = times;
            res.n_times = runs;
            res.time_s_median = med;
            res.throughput_gflops = gf;
            snprintf(res.output_file, sizeof(res.output_file), "%s", oname);
            snprintf(res.output_sha256, sizeof(res.output_sha256), "%s", hex);
            if (has_check) {
                res.has_check = 1;
                res.self_check_rel_err = rel_err;
                res.max_elem_diag = mdiag;
                res.tol = tol;
                res.pass = pass;
            }
            if (results_write_tier(out_dir, &res) != 0) {
                fprintf(stderr, "bench_mpi: results write failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            /* 控制台:误差以百分比显示(原始值在 tier JSON);tol 即预算 */
            char tag[64];
            if (has_check)
                snprintf(tag, sizeof(tag), " err=%.6f%%/%g%% %s",
                         rel_err * 100.0, tol * 100.0,
                         pass ? "[pass]" : "[FAIL]");
            else
                tag[0] = '\0';
            printf("mpi %-6s B%-2d H%-2d S%-5d D%-3d np=%-3d med=%.4fs "
                   "%.1f GF/s%s\n",
                   tiers[t].id, d->B, d->H, d->S, d->D, size, med, gf, tag);
            fflush(stdout);
        }

        free(Q); free(K); free(V); free(O);
    }

    MPI_Finalize();
    return any_fail ? 1 : 0;
}
