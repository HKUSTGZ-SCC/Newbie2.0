/* sha256.h — 自带 SHA-256(冻结文件,不得修改)
 * FIPS 180-4。用于冻结哈希清单与输出文件校验,避免依赖 openssl。
 */
#ifndef SHA256_H
#define SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t total_len;
    uint8_t buf[64];
    size_t buf_len;
} sha256_ctx_t;

void sha256_init(sha256_ctx_t *ctx);
void sha256_update(sha256_ctx_t *ctx, const void *data, size_t len);
void sha256_final(sha256_ctx_t *ctx, uint8_t out[32]);

/* 32 字节 → 64 字符小写 hex + '\0' */
void sha256_to_hex(const uint8_t in[32], char out[65]);

/* 便捷接口:成功返回 0,hex 为 64 字符摘要 */
int sha256_buf(const void *buf, size_t len, char hex[65]);
int sha256_file(const char *path, char hex[65]);

#endif /* SHA256_H */
