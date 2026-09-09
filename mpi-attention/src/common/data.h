/* data.h — 确定性输入生成(冻结文件,不得修改)
 *
 * Q/K/V 由 (manifest 种子, 档位, 张量种类, b, h) 唯一决定,与 rank 数、
 * 线程数、调用次序无关。每个 rank 独立生成相同内容(harness 保证,
 * 无分发阶段),使任意切分策略都能拿到全部输入。
 */
#ifndef DATA_H
#define DATA_H

#include <stdint.h>

#include "att_types.h"

/* 生成整个 [B][H][S][D] 三元组到调用方缓冲区 Q/K/V(各 numel 元素)。
 * 返回 ATT_OK / ATT_ERR_PARAM。 */
AttStatus data_generate(uint64_t seed, const char *tier_id,
                        const AttDims *dims,
                        float *restrict Q, float *restrict K,
                        float *restrict V);

#endif /* DATA_H */
