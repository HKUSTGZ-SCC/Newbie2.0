/* verify.c — 正确性误差口径实现(冻结文件,不得修改) */
#include "verify.h"

#include <stddef.h>

AttStatus verify_rel_err(const float *O, const double *R, long n,
                         double *rel_err, double *max_elem_diag)
{
    if (!O || !R || n <= 0)
        return ATT_ERR_PARAM;

    double num = 0.0, den = 0.0;
    double max_abs_diff = 0.0, max_abs_ref = 0.0;
    for (long i = 0; i < n; i++) {
        const double o = (double)O[i];
        const double r = R[i];
        const double diff = o - r;
        num += diff * diff;
        den += r * r;
        if (diff < 0 ? -diff > max_abs_diff : diff > max_abs_diff)
            max_abs_diff = diff < 0 ? -diff : diff;
        if (r < 0 ? -r > max_abs_ref : r > max_abs_ref)
            max_abs_ref = r < 0 ? -r : r;
    }

    if (den == 0.0)
        return ATT_ERR_PARAM; /* 参考全零:manifest 种子错误 */

    if (rel_err)
        *rel_err = __builtin_sqrt(num) / __builtin_sqrt(den);
    if (max_elem_diag)
        *max_elem_diag = max_abs_diff / max_abs_ref;
    return ATT_OK;
}
