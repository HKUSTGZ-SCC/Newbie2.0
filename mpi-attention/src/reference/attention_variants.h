/* attention_variants.h — 参照实现统一入口(冻结文件,不得修改)
 *
 * 本目录是同一定理 O = softmax(Q·K^T/√D)·V 的四种写法:
 * 三个只读参照(serial/omp/ref)+ 选手起点 baseline(可改)。
 * 四个实现全部为教科书式两遍 softmax、纯标量循环;差异各占一个轴。
 * 差异细节与阅读路径见本目录 README.md(先读)。速览:
 *
 * | 实现                  | 精度 | softmax | 并行(均单元维)   | 角色          |
 * |-----------------------|------|---------|--------------------|---------------|
 * | attention_serial.c    | fp32 | 两遍    | 无                 | 朴素地板/调试 |
 * | attention_omp.c       | fp32 | 两遍    | OpenMP 线程        | 共享内存参照  |
 * | attention_ref.c       | fp64 | 两遍    | 无(串行)         | 真值生成器    |
 * | attention_baseline.c  | fp32 | 两遍    | MPI 进程 + Gatherv | 选手起点(可改)|
 *
 * 三个 fp32 实现循环同构、输出位级一致;阶梯实测(本版默认 16 核):
 * serial 4.15 → omp 35.87 → baseline 33.31 GF/s(geo)。
 * 序列维并行是留给选手的挑战(见 README"取舍")。
 */
#ifndef ATTENTION_VARIANTS_H
#define ATTENTION_VARIANTS_H

#include "att_types.h"

/* 朴素串行(基准形态):教科书两遍 softmax,标量求和,fp32。
 * O 与输入同布局([B][H][S][D] 行主序)。 */
AttStatus attention_forward_serial(const float *restrict Q,
                                   const float *restrict K,
                                   const float *restrict V,
                                   const AttDims *dims,
                                   float *restrict O);

/* OpenMP 参照:相对 serial 的差异轴 = 仅并行化(单元维一条
 * #pragma omp for,per-thread scratch)。nthreads ≤ 0 取环境默认。
 * 输出与 serial 位级一致。 */
AttStatus attention_forward_omp(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                int nthreads,
                                float *restrict O);

/* fp64 真值生成器:相对 serial 的差异轴 = 仅精度(f64 全程)。
 * O 为 fp64 输出;串行、位级确定(同一二进制 + 同一机器)。 */
AttStatus attention_forward_ref(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                double *restrict O);

#endif /* ATTENTION_VARIANTS_H */
