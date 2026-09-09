/* io_util.c — 张量二进制读写实现(冻结文件,不得修改) */
#include "io_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char MAGIC[8] = "ATTBIN01";

int io_write_tensor(const char *path, const AttDims *dims, int dtype,
                    const void *data)
{
    if (!path || !dims || !data)
        return -1;
    if (dtype != ATT_DTYPE_F32 && dtype != ATT_DTYPE_F64)
        return -1;

    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    int32_t hdr[8];
    memcpy(hdr, MAGIC, 8);
    hdr[2] = dims->B; hdr[3] = dims->H; hdr[4] = dims->S; hdr[5] = dims->D;
    hdr[6] = dtype;   hdr[7] = 0;

    if (fwrite(hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    const size_t elem = (size_t)dtype;
    const size_t n = (size_t)att_numel(dims);
    /* 分块写,避免 2GB 单次 fwrite 截断 */
    const char *p = (const char *)data;
    const size_t CHUNK = (size_t)1 << 30;
    for (size_t off = 0; off < n * elem; off += CHUNK) {
        const size_t take = (n * elem - off) < CHUNK ? (n * elem - off) : CHUNK;
        if (fwrite(p + off, 1, take, f) != take) {
            fclose(f);
            return -1;
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

int io_read_tensor(const char *path, att_bin_hdr_t *hdr_out, void **data_out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;

    int32_t hdr[8];
    if (fread(hdr, sizeof(hdr), 1, f) != 1 || memcmp(hdr, MAGIC, 8) != 0) {
        fclose(f);
        return -1;
    }

    att_bin_hdr_t h;
    h.B = hdr[2]; h.H = hdr[3]; h.S = hdr[4]; h.D = hdr[5];
    h.dtype = hdr[6];

    if (h.B <= 0 || h.H <= 0 || h.S <= 0 || h.D <= 0 ||
        (h.dtype != ATT_DTYPE_F32 && h.dtype != ATT_DTYPE_F64)) {
        fclose(f);
        return -2;
    }

    const long long numel = (long long)h.B * h.H * h.S * h.D;
    const size_t bytes = (size_t)numel * (size_t)h.dtype;
    void *buf = malloc(bytes ? bytes : 1);
    if (!buf) {
        fclose(f);
        return -1;
    }

    const char *p = (const char *)buf;
    const size_t CHUNK = (size_t)1 << 30;
    for (size_t off = 0; off < bytes; off += CHUNK) {
        const size_t take = (bytes - off) < CHUNK ? (bytes - off) : CHUNK;
        if (fread((char *)p + off, 1, take, f) != take) {
            free(buf);
            fclose(f);
            return -1;
        }
    }
    fclose(f);

    if (hdr_out)
        *hdr_out = h;
    if (data_out)
        *data_out = buf;
    else
        free(buf);
    return 0;
}
