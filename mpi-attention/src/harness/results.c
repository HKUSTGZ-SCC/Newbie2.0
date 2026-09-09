/* results.c — 逐档结果文件写出实现(冻结文件,不得修改) */
#include "results.h"

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

int results_write_tier(const char *out_dir, const tier_result_t *r)
{
    if (!out_dir || !r || !r->tier_id)
        return -1;

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s.json", out_dir, r->tier_id);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;

    fprintf(f, "{\"tier_id\":\"%s\",\"dims\":{\"B\":%d,\"H\":%d,\"S\":%d,\"D\":%d}",
            r->tier_id, r->dims.B, r->dims.H, r->dims.S, r->dims.D);
    fprintf(f, ",\"flops\":%lld", r->flops);
    fprintf(f, ",\"np\":%d,\"omp_threads\":%d", r->np, r->omp_threads);
    fprintf(f, ",\"warmup\":%d,\"repeats\":%d", r->warmup, r->repeats);
    fprintf(f, ",\"times_s\":[");
    for (int i = 0; i < r->n_times; i++)
        fprintf(f, "%s%.6f", i ? "," : "", r->times_s[i]);
    fprintf(f, "],\"time_s_median\":%.6f", r->time_s_median);
    fprintf(f, ",\"throughput_gflops\":%.2f", r->throughput_gflops);
    fprintf(f, ",\"output_file\":\"%s\",\"output_sha256\":\"%s\"",
            r->output_file, r->output_sha256);
    if (r->has_check) {
        fprintf(f, ",\"self_check_rel_err\":%.6e,\"max_elem_diag\":%.6e",
                r->self_check_rel_err, r->max_elem_diag);
        fprintf(f, ",\"tol\":%.6e,\"pass\":%s", r->tol,
                r->pass ? "true" : "false");
    }
    fprintf(f, "}\n");

    return fclose(f) == 0 ? 0 : -1;
}
