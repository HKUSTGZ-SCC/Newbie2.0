/* io_util.h — 张量二进制读写(冻结文件,不得修改)
 *
 * 文件格式(小端,32 字节头):
 *   [0..7]   magic  "ATTBIN01"
 *   [8..23]  int32 B, H, S, D
 *   [24..27] int32 dtype(4=float32, 8=float64)
 *   [28..31] int32 保留(0)
 *   [32..]   行主序 [B][H][S][D] 数据
 */
#ifndef IO_UTIL_H
#define IO_UTIL_H

#include <stdint.h>

#include "att_types.h"

#define ATT_DTYPE_F32 4
#define ATT_DTYPE_F64 8

typedef struct {
    int B, H, S, D;
    int dtype;
} att_bin_hdr_t;

/* 写整个张量文件。返回 0 成功。 */
int io_write_tensor(const char *path, const AttDims *dims, int dtype,
                    const void *data);

/* 读文件:头部出 hdr_out(可 NULL),数据 malloc 到 *data_out(调用方 free)。
 * 返回 0 成功;-1 IO/格式错误;-2 维度与 dtype 非法。 */
int io_read_tensor(const char *path, att_bin_hdr_t *hdr_out, void **data_out);

#endif /* IO_UTIL_H */
