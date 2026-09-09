/* results.h — 逐档结果文件写出(冻结文件,不得修改)
 *
 * 每档一个单行 JSON 对象,写到 <out_dir>/<tier_id>.json。
 * 该格式是 bench_main → report_main 的内部契约,也是选手自查产物。
 */
#ifndef RESULTS_H
#define RESULTS_H

#include "att_types.h"

typedef struct {
    const char *tier_id;
    AttDims dims;
    long long flops;
    int np;             /* MPI rank 数(OpenMP 参照为 1) */
    int omp_threads;    /* 每 rank OpenMP 线程数 */
    int warmup;
    int repeats;
    double *times_s;    /* repeats 个计时值 */
    int n_times;
    double time_s_median;
    double throughput_gflops;
    char output_file[256];   /* 相对 out_dir 的输出文件名 */
    char output_sha256[65];
    int has_check;           /* 0=未提供 fp64 参考,省略校验字段 */
    double self_check_rel_err;
    double max_elem_diag;
    double tol;
    int pass;
} tier_result_t;

/* 返回 0 成功 */
int results_write_tier(const char *out_dir, const tier_result_t *r);

#endif /* RESULTS_H */
