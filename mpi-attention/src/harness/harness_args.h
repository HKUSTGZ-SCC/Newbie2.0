/* harness_args.h — 冻结 harness 共用的 CLI 解析(冻结文件,不得修改) */
#ifndef HARNESS_ARGS_H
#define HARNESS_ARGS_H

#include "att_types.h"

typedef struct {
    char id[32];
    AttDims dims;
} tier_spec_t;

/* 解析 "tier_id:B,H,S,D";成功返回 0。 */
int parse_tier_spec(const char *s, tier_spec_t *t);

const char *att_status_str(AttStatus st);

#endif /* HARNESS_ARGS_H */
