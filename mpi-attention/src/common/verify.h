/* verify.h — 正确性误差口径(冻结文件,不得修改)
 *
 * 通过判据(唯一):E = ||O − R||_F / ||R||_F ≤ tol
 *   O:选手 fp32 输出;R:fp64 参考;范数与差均 fp64 累加。
 * 逐元素诊断值(不 gating):max|O_i − R_i| / max|R_i|。
 */
#ifndef VERIFY_H
#define VERIFY_H

#include "att_types.h"

/* n 为元素数。rel_err/max_elem_diag 可为 NULL(不计算)。 */
AttStatus verify_rel_err(const float *O, const double *R, long n,
                         double *rel_err, double *max_elem_diag);

#endif /* VERIFY_H */
