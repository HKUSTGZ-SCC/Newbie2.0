/* ref_main.c — bin/gen_ref:串行 fp64 参考生成入口(冻结文件,不得修改)
 *
 * 用法:
 *   bin/gen_ref --seed <N> --tier <id:B,H,S,D> [--tier ...] --out-dir <dir>
 *
 * 每档输出 <out-dir>/<id>_output.bin(fp64,ATTBIN01 格式)并打印 sha256。
 * 同一二进制在同一机器上重复生成必须位级一致(自测项)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "attention_variants.h"
#include "data.h"
#include "harness_args.h"
#include "io_util.h"
#include "sha256.h"

int main(int argc, char **argv)
{
    unsigned long long seed = 0;
    const char *out_dir = NULL;
    tier_spec_t tiers[16];
    int n_tiers = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--tier") && i + 1 < argc) {
            if (n_tiers >= 16 || parse_tier_spec(argv[++i], &tiers[n_tiers]) != 0) {
                fprintf(stderr, "gen_ref: bad --tier spec\n");
                return 2;
            }
            n_tiers++;
        } else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) {
            out_dir = argv[++i];
        } else {
            fprintf(stderr, "gen_ref: unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (!out_dir || n_tiers == 0) {
        fprintf(stderr,
                "usage: gen_ref --seed N --tier id:B,H,S,D [--tier ...] --out-dir DIR\n");
        return 2;
    }
    mkdir(out_dir, 0755); /* 已存在则忽略错误 */

    for (int t = 0; t < n_tiers; t++) {
        const AttDims *d = &tiers[t].dims;
        const long n = att_numel(d);
        float *Q = malloc((size_t)n * 4), *K = malloc((size_t)n * 4),
              *V = malloc((size_t)n * 4);
        double *O = malloc((size_t)n * 8);
        if (!Q || !K || !V || !O) {
            fprintf(stderr, "gen_ref: alloc failed for tier %s\n", tiers[t].id);
            return 1;
        }

        AttStatus st = data_generate(seed, tiers[t].id, d, Q, K, V);
        if (st != ATT_OK)
            return 1;
        st = attention_forward_ref(Q, K, V, d, O);
        if (st != ATT_OK) {
            fprintf(stderr, "gen_ref: ref failed (%s)\n", att_status_str(st));
            return 1;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s_output.bin", out_dir, tiers[t].id);
        if (io_write_tensor(path, d, ATT_DTYPE_F64, O) != 0) {
            fprintf(stderr, "gen_ref: write failed: %s\n", path);
            return 1;
        }
        char hex[65];
        sha256_file(path, hex);
        printf("%s %s %lld flops\n", tiers[t].id, hex, att_flops(d));

        free(Q); free(K); free(V); free(O);
    }
    return 0;
}
