/* att_types.h — mpi-attention 公共类型(冻结文件,不得修改)
 *
 * 布局约定:Q/K/O 均为行主序 [B][H][S][D] 连续张量,fp32。
 * FLOPs 折算口径:4*B*H*S*S*D(QK^T 与 P·V 两个 matmul,softmax 不计)。
 */
#ifndef ATT_TYPES_H
#define ATT_TYPES_H

#include <stdint.h>

typedef struct {
    int B; /* batch  */
    int H; /* heads  */
    int S; /* seq len */
    int D; /* head dim */
} AttDims;

typedef enum {
    ATT_OK = 0,
    ATT_ERR_PARAM,   /* 非法维度/空指针 */
    ATT_ERR_ALLOC,   /* 内存分配失败 */
    ATT_ERR_MPI,     /* MPI 调用失败 */
    ATT_ERR_LIMIT,   /* 并行度越界(np 或 np*threads) */
    ATT_ERR_INTERNAL /* 内部不变量破坏 */
} AttStatus;

/* 行主序 [B][H][S][D] 寻址。dims 为含 B/H/S/D 的 AttDims*。 */
#define ATT_IDX(b, h, s, d, dims) \
    ((((long)(b) * (long)(dims)->H + (long)(h)) * (long)(dims)->S + (long)(s)) \
         * (long)(dims)->D + (long)(d))

/* 元素总数 */
static inline long att_numel(const AttDims *dims)
{
    return (long)dims->B * dims->H * dims->S * dims->D;
}

/* 折算 FLOPs(2 matmul × 2 flop/MAC) */
static inline long long att_flops(const AttDims *dims)
{
    return 4LL * dims->B * dims->H * (long long)dims->S * dims->S * dims->D;
}

#endif /* ATT_TYPES_H */
