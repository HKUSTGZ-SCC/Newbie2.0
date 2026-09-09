/* attention.h — 选手实现的接口契约(冻结文件,签名与语义不得修改)
 *
 * 阅读路径建议:先读本目录 README.md(四实现差异对照 + 自由度地图),
 * 再读本契约;选手起点 attention_baseline.c(本目录,可改)。
 *
 * 语义:
 *  - Q,K,V 为行主序 [B][H][S][D] 连续 fp32,调用前在 comm 的每个 rank 上
 *    位级一致(harness 保证:每 rank 独立生成,内容相同)。
 *  - attention_forward_mpi 为集合调用:comm 内所有 rank 以相同参数共同进入。
 *    返回 ATT_OK 后,rank 0 的 O 必须持有完整的 [B][H][S][D] fp32 结果
 *    (内部 gather/归并计入耗时,不得把数据留给调用方收)。
 *    其余 rank 的 O 允许为未定义内容。
 *  - 结果正确性:与 fp64 参考的 Frobenius 相对误差 ≤ manifest.tolerance。
 *
 * 合法(题面"可自由优化"面):
 *  - 任意内部并行策略(切分方式、通信模式、通信计算重叠)
 *  - 任意低于 fp64 的计算精度(fp32/bf16/fp16/混合……误差阈值内即可)
 *  - 跨调用保留的私有 scratch/缓存(harness 有 warmup 吸收一次性成本)
 *  - 内部使用 OpenMP 线程(编译已带 -fopenmp;np*threads ≤ 192)
 *
 * 非法(题面"非法操作"):
 *  - 外部数学库(BLAS/MKL/OpenBLAS/FFTW/Eigen 等,libc/libm 除外)
 *  - 修改输入 Q/K/V;伪造/回放输出;削减语义等价的计算量
 */
#ifndef ATTENTION_H
#define ATTENTION_H

#include <mpi.h>

#include "att_types.h"

/* 选手实现目标:MPI 并行 scaled dot-product attention。
 * O = softmax(Q·K^T / sqrt(D)) · V
 * 返回 ATT_OK 或 ATT_ERR_*;不得调用 MPI_Abort 处理可恢复错误。 */
AttStatus attention_forward_mpi(const float *restrict Q,
                                const float *restrict K,
                                const float *restrict V,
                                const AttDims *dims,
                                MPI_Comm comm,
                                float *restrict O);

#endif /* ATTENTION_H */
