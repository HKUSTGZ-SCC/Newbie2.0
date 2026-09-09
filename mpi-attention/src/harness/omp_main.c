/* omp_main.c — bin/bench_omp:OpenMP 参照 harness(冻结文件,不得修改,不计分)
 *
 * 用法:
 *   bin/bench_omp --seed N --tier id:B,H,S,D [--tier ...] --out-dir DIR
 *                 [--ref-dir DIR] [--threads N] [--runs R] [--warmup W] [--tol T]
 *
 * 协议与 bench_main 相同:warmup W 次 + R 次计时取中位(omp_get_wtime)。
 * 输出:<out-dir>/<id>_omp.json(逐档)+ <out-dir>/omp_results.json(汇总)
 *       以及 <out-dir>/omp_<id>_output.bin(fp32 输出,供核对)。
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "attention_variants.h"
#include "data.h"
#include "harness_args.h"
#include "io_util.h"
#include "results.h"
#include "sha256.h"
#include "verify.h"

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
    unsigned long long seed = 0;
    const char *out_dir = NULL, *ref_dir = NULL;
    int threads = 0, runs = 5, warmup = 1;
    double tol = 0.02;
    tier_spec_t tiers[16];
    int n_tiers = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--tier") && i + 1 < argc) {
            if (n_tiers >= 16 || parse_tier_spec(argv[++i], &tiers[n_tiers]) != 0) {
                fprintf(stderr, "bench_omp: bad --tier spec\n");
                return 2;
            }
            n_tiers++;
        } else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (!strcmp(argv[i], "--ref-dir") && i + 1 < argc) {
            ref_dir = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--tol") && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else {
            fprintf(stderr, "bench_omp: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (!out_dir || n_tiers == 0 || seed == 0) {
        fprintf(stderr, "usage: bench_omp --seed N --tier id:B,H,S,D [--tier ...] "
                        "--out-dir DIR [--ref-dir D] [--threads N] [--runs R] "
                        "[--warmup W] [--tol T]\n");
        return 2;
    }
    if (runs < 1) runs = 1;
    mkdir(out_dir, 0755);
    if (threads <= 0)
        threads = omp_get_max_threads();

    FILE *summary = NULL;
    char spath[4096];
    snprintf(spath, sizeof(spath), "%s/omp_results.json", out_dir);
    summary = fopen(spath, "w");
    if (summary)
        fprintf(summary,
                "{\"schema_version\":1,\"bench\":\"mpi-attention-omp\","
                "\"threads\":%d,\"tiers\":[",
                threads);

    for (int t = 0; t < n_tiers; t++) {
        const AttDims *d = &tiers[t].dims;
        const long n = att_numel(d);
        float *Q = malloc((size_t)n * 4), *K = malloc((size_t)n * 4),
              *V = malloc((size_t)n * 4), *O = malloc((size_t)n * 4);
        double *R = NULL;
        if (!Q || !K || !V || !O)
            return 1;

        if (data_generate(seed, tiers[t].id, d, Q, K, V) != ATT_OK)
            return 1;

        /* 参考加载(可选) */
        char rpath[4096];
        if (ref_dir) {
            snprintf(rpath, sizeof(rpath), "%s/%s_output.bin", ref_dir, tiers[t].id);
            att_bin_hdr_t rh;
            if (io_read_tensor(rpath, &rh, (void **)&R) != 0) {
                fprintf(stderr, "bench_omp: cannot read reference %s\n", rpath);
                return 1;
            }
            if (rh.B != d->B || rh.H != d->H || rh.S != d->S || rh.D != d->D ||
                rh.dtype != ATT_DTYPE_F64) {
                fprintf(stderr, "bench_omp: reference shape mismatch for %s\n",
                        tiers[t].id);
                return 1;
            }
        }

        for (int w = 0; w < warmup; w++)
            attention_forward_omp(Q, K, V, d, threads, O);

        double times[32];
        if (runs > 32) runs = 32;
        for (int r = 0; r < runs; r++) {
            const double t0 = omp_get_wtime();
            if (attention_forward_omp(Q, K, V, d, threads, O) != ATT_OK)
                return 1;
            times[r] = omp_get_wtime() - t0;
        }
        const double med = median_of(times, runs);
        const double gf = (double)att_flops(d) / med / 1e9;

        /* 写 fp32 输出与逐档 JSON */
        char opath[4096], oname[256];
        snprintf(oname, sizeof(oname), "omp_%.31s_output.bin", tiers[t].id);
        snprintf(opath, sizeof(opath), "%s/%s", out_dir, oname);
        if (io_write_tensor(opath, d, ATT_DTYPE_F32, O) != 0)
            return 1;
        char hex[65];
        sha256_file(opath, hex);

        tier_result_t res;
        memset(&res, 0, sizeof(res));
        char idbuf[64];
        snprintf(idbuf, sizeof(idbuf), "%.31s_omp", tiers[t].id);
        res.tier_id = idbuf;
        res.dims = *d;
        res.flops = att_flops(d);
        res.np = 1;
        res.omp_threads = threads;
        res.warmup = warmup;
        res.repeats = runs;
        res.times_s = times;
        res.n_times = runs;
        res.time_s_median = med;
        res.throughput_gflops = gf;
        snprintf(res.output_file, sizeof(res.output_file), "%s", oname);
        snprintf(res.output_sha256, sizeof(res.output_sha256), "%s", hex);
        if (R) {
            res.has_check = 1;
            res.tol = tol;
            if (verify_rel_err(O, R, n, &res.self_check_rel_err,
                               &res.max_elem_diag) != ATT_OK)
                return 1;
            res.pass = res.self_check_rel_err <= tol;
        }
        /* results_write_tier 以 tier_id(<id>_omp)命名逐档文件 */
        {
            tier_result_t tmp = res;
            results_write_tier(out_dir, &tmp);
        }
        /* 控制台:误差以百分比显示(原始值在 tier JSON) */
        char tag[64];
        if (R)
            snprintf(tag, sizeof(tag), " err=%.6f%%/%g%% %s",
                     res.self_check_rel_err * 100.0, tol * 100.0,
                     res.pass ? "[pass]" : "[FAIL]");
        else
            tag[0] = '\0';
        printf("omp %-6s S=%-5d median=%.4fs %.1f GF/s%s\n", tiers[t].id,
               d->S, med, gf, tag);

        if (summary) {
            fprintf(summary,
                    "%s{\"tier_id\":\"%s\",\"np\":1,\"omp_threads\":%d,"
                    "\"time_s_median\":%.6f,\"throughput_gflops\":%.2f",
                    t ? "," : "", tiers[t].id, threads, med, gf);
            if (R)
                fprintf(summary,
                        ",\"self_check_rel_err\":%.6e,\"pass\":%s",
                        res.self_check_rel_err, res.pass ? "true" : "false");
            fprintf(summary, "}");
        }

        free(Q); free(K); free(V); free(O); free(R);
    }

    if (summary) {
        fprintf(summary, "]}\n");
        fclose(summary);
    }
    return 0;
}
