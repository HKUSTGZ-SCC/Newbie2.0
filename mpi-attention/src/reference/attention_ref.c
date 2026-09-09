/* attention_ref.c — 串行 fp64 参考实现(冻结文件,不得修改)
 *
 * 【角色】真值生成器:O 的 fp64 定义,gen_ref 由此产出 data/reference
 * 与裁判私有参考(sha256 冻结在 manifest)。正确性判据(相对误差
 * ≤ tol)以本实现为基准。
 * 【相对 serial 基准的差异轴】精度:全程 f64(exp 用 exp()),
 * 其余结构与 attention_serial.c 逐行对齐 —— diff 只见 float→double。
 * 【对照】差异矩阵见本目录 README.md。
 *
 * 两遍 softmax(与 serial 同构):
 *   分数 → 行 max → exp+sum → P·V → 归一化。
 * 不物化 S×S 矩阵(scratch 一行 S 个 double);串行、无 OpenMP/MPI,
 * 位级确定(同一二进制 + 同一机器)。
 */
#include "attention_variants.h"

#include <math.h>
#include <stdlib.h>

AttStatus attention_forward_ref(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                double *restrict O)
{
    if (!Q || !K || !V || !dims || !O)
        return ATT_ERR_PARAM;
    if (dims->B <= 0 || dims->H <= 0 || dims->S <= 0 || dims->D <= 0)
        return ATT_ERR_PARAM;

    const int S = dims->S, D = dims->D;
    const long block = (long)S * D;
    const long nunits = (long)dims->B * dims->H;
    const double scale = 1.0 / sqrt((double)D);

    double *scr = malloc(sizeof(double) * (size_t)S);
    if (!scr)
        return ATT_ERR_ALLOC;

    for (long u = 0; u < nunits; u++) {
        const float *q = Q + u * block;
        const float *k = K + u * block;
        const float *v = V + u * block;
        double *o = O + u * block;

        for (int s = 0; s < S; s++) {
            const float *qs = q + (long)s * D;

            /* 1) 分数 */
            for (int j = 0; j < S; j++) {
                const float *kj = k + (long)j * D;
                double dot = 0.0;
                for (int d = 0; d < D; d++)
                    dot += (double)qs[d] * (double)kj[d];
                scr[j] = dot * scale;
            }

            /* 2) max */
            double m = scr[0];
            for (int j = 1; j < S; j++)
                if (scr[j] > m)
                    m = scr[j];

            /* 3) exp + sum */
            double l = 0.0;
            for (int j = 0; j < S; j++) {
                scr[j] = exp(scr[j] - m);
                l += scr[j];
            }

            /* 4) 加权求和 V */
            double *os = o + (long)s * D;
            for (int d = 0; d < D; d++)
                os[d] = 0.0;
            for (int j = 0; j < S; j++) {
                const double p = scr[j];
                const float *vj = v + (long)j * D;
                for (int d = 0; d < D; d++)
                    os[d] += p * (double)vj[d];
            }

            /* 5) 归一化 */
            const double inv_l = 1.0 / l;
            for (int d = 0; d < D; d++)
                os[d] *= inv_l;
        }
    }

    free(scr);
    return ATT_OK;
}
