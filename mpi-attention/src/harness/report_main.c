/* report_main.c — bin/report:聚合 results.json(冻结文件,不得修改)
 *
 * 用法:
 *   bin/report --out-dir <dir> --bench <name> --tier <id> [--tier ...]
 *
 * 读取 <dir>/<id>.json(bench_main 写出的单行 JSON 对象),原样嵌入
 * "tiers" 数组,计算吞吐几何平均与总内核时间,写 <dir>/results.json。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "harness_args.h"

#define MAX_LINE (1 << 20)

static int extract_double(const char *line, const char *key, double *out)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *p = strstr(line, pat);
    if (!p)
        return -1;
    if (sscanf(p + strlen(pat), "%lf", out) != 1)
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    const char *out_dir = NULL, *bench = "mpi-attention";
    const char *ids[32];
    int n_ids = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (!strcmp(argv[i], "--bench") && i + 1 < argc) {
            bench = argv[++i];
        } else if (!strcmp(argv[i], "--tier") && i + 1 < argc) {
            if (n_ids >= 32) {
                fprintf(stderr, "report: too many tiers\n");
                return 2;
            }
            ids[n_ids++] = argv[++i];
        } else {
            fprintf(stderr, "report: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (!out_dir || n_ids == 0) {
        fprintf(stderr, "usage: report --out-dir DIR --bench NAME "
                        "--tier id [--tier ...]\n");
        return 2;
    }

    char *line = malloc(MAX_LINE);
    if (!line)
        return 1;

    char rpath[4096];
    snprintf(rpath, sizeof(rpath), "%.2040s/results.json", out_dir);
    FILE *out = fopen(rpath, "w");
    if (!out) {
        fprintf(stderr, "report: cannot write %s\n", rpath);
        return 1;
    }

    fprintf(out, "{\"schema_version\":1,\"bench\":\"%s\",\"tiers\":[", bench);

    double log_sum = 0.0, total_time = 0.0;
    int ok_tiers = 0, failures = 0;

    for (int t = 0; t < n_ids; t++) {
        char path[4096];
        snprintf(path, sizeof(path), "%.2040s/%.31s.json", out_dir, ids[t]);
        FILE *f = fopen(path, "r");
        if (!f) {
            fprintf(stderr, "report: missing tier result %s\n", path);
            fclose(out);
            return 1;
        }
        if (!fgets(line, MAX_LINE, f)) {
            fprintf(stderr, "report: empty tier result %s\n", path);
            fclose(f); fclose(out);
            return 1;
        }
        fclose(f);

        double gf = 0.0, tmed = 0.0;
        if (extract_double(line, "throughput_gflops", &gf) != 0 ||
            extract_double(line, "time_s_median", &tmed) != 0) {
            fprintf(stderr, "report: malformed tier result %s\n", path);
            fclose(out);
            return 1;
        }
        /* pass 字段仅诊断;缺参考时视为 pass 以便汇总 */
        double passv = 1.0;
        const char *pp = strstr(line, "\"pass\":");
        if (pp && sscanf(pp + 7, "%lf", &passv) == 1 && passv < 0.5)
            failures++;

        fprintf(out, "%s\n  %s", t ? "," : "", line);
        if (gf > 0) {
            log_sum += log(gf);
            ok_tiers++;
        }
        total_time += tmed;
    }

    const double geo = ok_tiers ? exp(log_sum / ok_tiers) : 0.0;
    fprintf(out, "],\n \"geo_mean_gflops\":%.2f,\n \"total_kernel_time_s\":%.6f,"
                 "\n \"all_pass\":%s}\n",
            geo, total_time, failures ? "false" : "true");
    fclose(out);

    printf("report: %d tiers, geo-mean %.2f GF/s, total kernel %.3fs%s\n",
           n_ids, geo, total_time, failures ? " [HAS FAILURES]" : "");
    return failures ? 1 : 0;
}
