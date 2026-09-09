/* harness_args.c — 冻结 harness 共用的 CLI 解析(冻结文件,不得修改) */
#include "harness_args.h"

#include <stdio.h>
#include <string.h>

int parse_tier_spec(const char *s, tier_spec_t *t)
{
    if (!s || !t)
        return -1;
    const char *colon = strchr(s, ':');
    if (!colon || colon == s || (size_t)(colon - s) >= sizeof(t->id))
        return -1;
    memcpy(t->id, s, (size_t)(colon - s));
    t->id[colon - s] = '\0';

    int B, H, S, D;
    if (sscanf(colon + 1, "%d,%d,%d,%d", &B, &H, &S, &D) != 4)
        return -1;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0)
        return -1;
    t->dims.B = B; t->dims.H = H; t->dims.S = S; t->dims.D = D;
    return 0;
}

const char *att_status_str(AttStatus st)
{
    switch (st) {
    case ATT_OK:           return "ok";
    case ATT_ERR_PARAM:    return "invalid parameter";
    case ATT_ERR_ALLOC:    return "allocation failure";
    case ATT_ERR_MPI:      return "mpi error";
    case ATT_ERR_LIMIT:    return "parallelism limit exceeded";
    case ATT_ERR_INTERNAL: return "internal error";
    }
    return "unknown";
}
