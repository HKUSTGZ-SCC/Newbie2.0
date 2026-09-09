/* attention_omp.c — OpenMP 参照实现(冻结文件,不得修改,不计分)
 *
 * 【角色】"单进程共享内存"的性能参照系(题面与 manifest 公示其成绩,
 * 本版参照配置 = 16 线程)。
 * 【相对 serial 基准的差异轴】仅并行化:一条 #pragma omp for
 * 沿 (b,h) 单元维切分,per-thread scratch(S 个 float)。
 * 无 simd pragma、无块化 —— 循环体与 attention_serial.c 逐行一致,
 * 同循环同顺序,输出与 serial 位级一致。
 * 【对照】diff -u attention_serial.c attention_omp.c 可见全部增量;
 * 差异矩阵见本目录 README.md。
 *
 * 两遍 softmax(与 serial 同构):分数 → 行 max → exp/Σ → P·V → 归一化。
 * long 档只有 B×H 个单元,线程大量闲置是刻意保留的朴素性 ——
 * 序列维并行是留给选手的挑战(见 README"取舍")。
 * 选手可读不可改;定位是"干净、正确、有代表性"的朴素并行实现。
 */
#include "attention_variants.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>

AttStatus attention_forward_omp(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                int nthreads,
                                float *restrict O)
{
    if (!Q || !K || !V || !dims || !O)
        return ATT_ERR_PARAM;
    if (dims->B <= 0 || dims->H <= 0 || dims->S <= 0 || dims->D <= 0)
        return ATT_ERR_PARAM;
    if (nthreads <= 0)
        nthreads = omp_get_max_threads();

    const int S = dims->S, D = dims->D;
    const long block = (long)S * D;
    const long nunits = (long)dims->B * dims->H;
    const float scale = 1.0f / sqrtf((float)D);

#pragma omp parallel num_threads(nthreads)
    {
        /* per-thread 分数 scratch,跨单元复用 */
        float *scr = malloc(sizeof(float) * (size_t)S);
        if (scr) {
#pragma omp for schedule(static)
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

                    /* 2) max */
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
        }
        free(scr);
    }

    return ATT_OK;
}
