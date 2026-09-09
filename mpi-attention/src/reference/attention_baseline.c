/* attention_baseline.c — MPI baseline 实现(选手的起点,可整文件重写)
 *
 * 【角色】四级参照阶梯的第四级(与 serial/omp/ref 同目录,差异对照与
 * 自由度地图见本目录 README.md)。相对 OpenMP 参照的差异轴 = MPI 进程化。
 * 计算循环与 attention_serial.c 逐行同构 —— 三者输出位级一致。
 * 【注意】本文件是本目录唯一预置的可改文件(其余全部冻结,见 manifest)。
 *
 * ============ 三节导读(改什么动哪里)============
 *
 *   1. 切分维度 —— 本实现:B×H 单元轮转(u ≡ rank mod np)。
 *      这是选手的主要改动点:
 *        - Q 行切分:rank 只算部分 Q 行 → 改第 1 节 + 第 3 节(按行 gather)
 *        - K/V 行片切分:softmax 需 online 可结合状态 (m,l,acc) +
 *          跨进程归并 → 第 1、2、3 节都要动(见 README"取舍")
 *        - 2D 混合切分:同理
 *   2. 计算 —— 两遍 softmax 循环体(与 serial 相同)。选手可换精度、
 *      加 SIMD、分块复用 K/V、融合线程(hybrid)……自由。
 *   3. 汇总 —— Gatherv 到 rank 0(契约:返回后 rank 0 持完整 O)。
 *
 * ============ 已知弱点(刻意保留,是选手的优化空间)============
 *   long 档只有 B×H=4 个单元,np>4 即大量 rank 闲置;每 Q 行整扫 K/V
 *   (内存带宽受限);无 SIMD、无精度优化、无通信重叠。
 *   实测数字见 README"参照阶梯"。
 *
 * 完整自由度清单见本目录 README.md;合法优化方向见
 * competition/README.md"可自由优化"。
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attention.h"

AttStatus attention_forward_mpi(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                MPI_Comm comm,
                                float *restrict O)
{
    /* ===== 0. 参数与并行信息 ===== */
    if (!Q || !K || !V || !dims || !O)
        return ATT_ERR_PARAM;
    if (dims->B <= 0 || dims->H <= 0 || dims->S <= 0 || dims->D <= 0)
        return ATT_ERR_PARAM;

    int rank, size;
    if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS ||
        MPI_Comm_size(comm, &size) != MPI_SUCCESS)
        return ATT_ERR_MPI;

    const int S = dims->S, D = dims->D;
    const long block = (long)S * D;
    const long nunits = (long)dims->B * dims->H;
    const float scale = 1.0f / sqrtf((float)D);

    /* ===== 1. 切分维度 =====
     * 本实现:单元维(B×H)轮转 —— rank r 整算单元 u ≡ r (mod size),
     * 计算阶段零通信。nunits < size 时多余 rank 空闲(my_cnt=0)。
     * (换切分维度的导引见文件头"三节导读") */
    const long my_cnt = (nunits - rank + size - 1) / size;

    /* ===== 2. 计算:两遍 softmax(与 attention_serial.c 逐行同构)===== */
    float *scr = malloc(sizeof(float) * (size_t)S); /* 一行分数,跨单元复用 */
    if (!scr) {
        /* 集合调用前失败无法各 rank 对齐 → 统一终止,避免死锁 */
        fprintf(stderr, "baseline: scratch alloc failed at rank %d\n", rank);
        fflush(stderr);
        MPI_Abort(comm, 2);
    }

    for (long k = 0; k < my_cnt; k++) {
        const long u = rank + k * size; /* 本 rank 的第 k 个单元 */
        const float *q = Q + u * block;
        const float *kmat = K + u * block;
        const float *v = V + u * block;
        float *o = O + u * block;

        for (int s = 0; s < S; s++) {
            const float *qs = q + (long)s * D;

            /* 1) 分数 */
            for (int j = 0; j < S; j++) {
                const float *kj = kmat + (long)j * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += qs[d] * kj[d];
                scr[j] = dot * scale;
            }

            /* 2) 行 max(数值稳定) */
            float m = scr[0];
            for (int j = 1; j < S; j++)
                if (scr[j] > m)
                    m = scr[j];

            /* 3) 原地 exp(减 max)+ Σ */
            float l = 0.0f;
            for (int j = 0; j < S; j++) {
                scr[j] = expf(scr[j] - m);
                l += scr[j];
            }

            /* 4) P·V 加权求和 */
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

    /* ===== 3. 汇总到 rank 0(契约:返回后 rank 0 持完整 O)=====
     * 各 rank 把自己算好的单元块(已在 O 正确偏移)打包 → Gatherv →
     * rank 0 按 (rank, k) → u = rank + k*size 解包。 */
    int *counts = malloc(sizeof(int) * (size_t)size);
    int *displs = malloc(sizeof(int) * (size_t)size);
    float *sendbuf = malloc(sizeof(float) *
                            (size_t)(my_cnt > 0 ? my_cnt * block : 1));
    float *recvbuf = NULL;
    if (rank == 0)
        recvbuf = malloc(sizeof(float) * (size_t)(nunits * block));
    if (!counts || !displs || !sendbuf || (rank == 0 && !recvbuf)) {
        fprintf(stderr, "baseline: gather alloc failed at rank %d\n", rank);
        fflush(stderr);
        MPI_Abort(comm, 2);
    }

    for (long k = 0; k < my_cnt; k++) {
        const long u = rank + k * size;
        memcpy(sendbuf + k * block, O + u * block,
               sizeof(float) * (size_t)block);
    }
    for (int r = 0; r < size; r++)
        counts[r] = (int)(((nunits - r + size - 1) / size) * block);
    displs[0] = 0;
    for (int r = 1; r < size; r++)
        displs[r] = displs[r - 1] + counts[r - 1];

    if (MPI_Gatherv(sendbuf, (int)(my_cnt * block), MPI_FLOAT, recvbuf,
                    counts, displs, MPI_FLOAT, 0, comm) != MPI_SUCCESS)
        return ATT_ERR_MPI;

    if (rank == 0) {
        long off = 0;
        for (int r = 0; r < size; r++) {
            const long cnt = (nunits - r + size - 1) / size;
            for (long k = 0; k < cnt; k++) {
                const long u = r + k * size;
                memcpy(O + u * block, recvbuf + off,
                       sizeof(float) * (size_t)block);
                off += block;
            }
        }
    }

    free(counts);
    free(displs);
    free(sendbuf);
    free(recvbuf);
    return ATT_OK;
}
