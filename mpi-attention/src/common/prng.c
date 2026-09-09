/* prng.c — 自带确定性 PRNG 实现(冻结文件,不得修改) */
#include "prng.h"

#include <stddef.h>

uint64_t splitmix64_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void xoshiro256pp_init(xoshiro256pp_t *st, uint64_t seed)
{
    uint64_t sm = seed;
    for (int i = 0; i < 4; i++)
        st->s[i] = splitmix64_next(&sm);
}

uint64_t xoshiro256pp_next(xoshiro256pp_t *st)
{
    uint64_t *s = st->s;
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

float prng_uniform_pm1(xoshiro256pp_t *st)
{
    /* 取高 23 位作尾数 → [1,2) → 减 3 得 [-1,1)。
     * 2^n 尺度内的减法全部精确,位级确定。 */
    uint32_t bits = (uint32_t)(xoshiro256pp_next(st) >> 41) | 0x40000000u;
    union { uint32_t u; float f; } cvt;
    cvt.u = bits;
    return cvt.f - 3.0f;
}

static uint64_t fnv1a64_str(const char *s)
{
    uint64_t h = 0xCBF29CE484222325ULL;
    while (*s) {
        h ^= (uint64_t)(unsigned char)*s++;
        h *= 0x100000001B3ULL;
    }
    return h;
}

uint64_t prng_seed_for_tier(uint64_t seed, const char *tier_id)
{
    uint64_t st = seed ^ fnv1a64_str(tier_id);
    return splitmix64_next(&st);
}

uint64_t prng_seed_stream(uint64_t seed_tier, int tag, int b, int h)
{
    uint64_t v = seed_tier;
    v = splitmix64_next(&v) ^ (0xC2B2AE3D27D4EB4FULL * (uint64_t)(tag + 1));
    v = splitmix64_next(&v) ^ (uint64_t)(b + 1) * 0x9E3779B97F4A7C15ULL;
    v = splitmix64_next(&v) ^ (uint64_t)(h + 1) * 0xBF58476D1CE4E5B9ULL;
    return splitmix64_next(&v);
}
