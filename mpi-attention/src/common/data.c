/* data.c — 确定性输入生成实现(冻结文件,不得修改) */
#include "data.h"

#include <stddef.h>

#include "prng.h"

AttStatus data_generate(uint64_t seed, const char *tier_id,
                        const AttDims *dims,
                        float *restrict Q, float *restrict K,
                        float *restrict V)
{
    if (!tier_id || !dims || !Q || !K || !V)
        return ATT_ERR_PARAM;
    if (dims->B <= 0 || dims->H <= 0 || dims->S <= 0 || dims->D <= 0)
        return ATT_ERR_PARAM;

    const uint64_t seed_tier = prng_seed_for_tier(seed, tier_id);
    const long block = (long)dims->S * dims->D;
    const long nunits = (long)dims->B * dims->H;
    float *tensors[3] = { Q, K, V };

    for (int tag = 0; tag < 3; tag++) {
        float *dst = tensors[tag];
        for (long u = 0; u < nunits; u++) {
            const int b = (int)(u / dims->H);
            const int h = (int)(u % dims->H);
            xoshiro256pp_t st;
            xoshiro256pp_init(&st, prng_seed_stream(seed_tier, tag, b, h));
            float *blk = dst + u * block;
            for (long i = 0; i < block; i++)
                blk[i] = prng_uniform_pm1(&st);
        }
    }
    return ATT_OK;
}
