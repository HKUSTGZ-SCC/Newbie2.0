/* prng.h — 自带确定性 PRNG(冻结文件,不得修改)
 *
 * 为什么不用 rand()/drand48():平台相关、流不可控、跨机器不可复现。
 * 本模块为纯位运算实现,任何机器/优化级别下位级一致,且不依赖 libm。
 */
#ifndef PRNG_H
#define PRNG_H

#include <stdint.h>

typedef struct {
    uint64_t s[4];
} xoshiro256pp_t;

/* splitmix64:种子展开器(也用于子流派生) */
uint64_t splitmix64_next(uint64_t *state);

/* xoshiro256++ 主流;种子经 splitmix64 链展开,避免劣质种子 */
void xoshiro256pp_init(xoshiro256pp_t *st, uint64_t seed);
uint64_t xoshiro256pp_next(xoshiro256pp_t *st);

/* 均匀 [-1, 1) fp32:纯位运算 + 精确浮点运算,无 libm、无舍入歧义 */
float prng_uniform_pm1(xoshiro256pp_t *st);

/* 子流派生:tier 字符串(fnv1a 混入全局种子)→ (tag, b, h) 唯一流种子。
 * tag: 0=Q, 1=K, 2=V。保证任意 (seed, tier, tag, b, h) 组合流独立。 */
uint64_t prng_seed_for_tier(uint64_t seed, const char *tier_id);
uint64_t prng_seed_stream(uint64_t seed_tier, int tag, int b, int h);

#endif /* PRNG_H */
