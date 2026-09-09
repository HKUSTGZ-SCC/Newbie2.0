/* attention_serial.c — 朴素串行实现(冻结文件,不得修改)
 *
 * 【角色】参照阶梯的最底层 + 调试工具:单核、无 MPI/OpenMP。
 * 【相对基准的差异轴】无 —— 本文件就是共享骨架的基准形态。
 * 【对照】omp = 本骨架 + 线程(单元维切分);ref = 本骨架 + fp64;
 *         baseline(本目录)= 本骨架 + MPI 进程(单元维切分)。
 *         三者同为两遍 softmax,输出位级一致;差异矩阵见本目录 README.md。
 *
 * 教科书式两遍 softmax,对每个 (b,h) 单元:
 *   1. 对每个 Q 行算全部分数 scr[j] = <q_s, k_j> / sqrt(D)
 *   2. 行内 max(数值稳定)
 *   3. exp(scr - m) 与求和得 l
 *   4. 加权求和 V
 *   5. 除以 l
 * 无任何并行指示与优化技巧;scratch 仅一行分数(S 个 float)。
 * 两遍策略的局限(先见全部分数才能求 max)见 README"取舍"一节。
 */
#include "attention_variants.h"

#include <math.h>
#include <stdlib.h>

AttStatus attention_forward_serial(const float *restrict Q,
                                   const float *restrict K,
                                   const float *restrict V,
                                   const AttDims *dims,
                                   float *restrict O)
{
    if (!Q || !K || !V || !dims || !O)
        return ATT_ERR_PARAM;
    if (dims->B <= 0 || dims->H <= 0 || dims->S <= 0 || dims->D <= 0)
        return ATT_ERR_PARAM;

    const int S = dims->S, D = dims->D;
    const long block = (long)S * D;
    const long nunits = (long)dims->B * dims->H;
    const float scale = 1.0f / sqrtf((float)D);

    float *scr = malloc(sizeof(float) * (size_t)S);
    if (!scr)
        return ATT_ERR_ALLOC;

    for (long u = 0; u < nunits; u++) {
        const float *q = Q + u * block;
        const float *k = K + u * block;
        const float *v = V + u * block;
        float *o = O + u * block;

        for (int s = 0; s < S; s++) {
            const float *qs = q + (long)s * D;

            /* 1) 分数 */
            for (int j = 0; j < S; j++) {
                const float *kj = k + (long)j * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += qs[d] * kj[d];
                scr[j] = dot * scale;
            }

            /* 2) max —— 两遍策略的特征:必须先见整行分数才能 exp。
             *    参照实现均为两遍;若选手做序列维切分(K/V 分散到多 rank),
             *    需要可结合的 online softmax 状态 (m,l,acc) 或等价归并
             *    方案 —— 见 README"取舍"。 */
            float m = scr[0];
            for (int j = 1; j < S; j++)
                if (scr[j] > m)
                    m = scr[j];

            /* 3) exp + sum */
            float l = 0.0f;
            for (int j = 0; j < S; j++) {
                scr[j] = expf(scr[j] - m);
                l += scr[j];
            }

            /* 4) 加权求和 V */
            float *os = o + (long)s * D;
            for (int d = 0; d < D; d++)
                os[d] = 0.0f;
            for (int j = 0; j < S; j++) {
                const float p = scr[j];
                const float *vj = v + (long)j * D;
                for (int d = 0; d < D; d++)
                    os[d] += p * vj[d];
            }

            /* 5) 归一化 */
            const float inv_l = 1.0f / l;
            for (int d = 0; d < D; d++)
                os[d] *= inv_l;
        }
    }

    free(scr);
    return ATT_OK;
}
