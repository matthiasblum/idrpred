# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/matthiasblum/idrpred/blob/main/LICENSE

"""
Savitzky-Golay filter, reimplemented from TISEAN 2.1 (sav_gol.c).
"""

from fractions import Fraction
from functools import lru_cache

from cpython.array cimport array, clone

_POWER = 2
_template = array("d")


@lru_cache(maxsize=None)
def _coefficients(int nb, int nf, int deriv, int power=_POWER) -> array:
    """
    Return the filter coefficients for the given derivative, i.e. row
    `deriv` of the pseudo-inverse of the Vandermonde matrix, scaled by
    1/deriv!, as in TISEAN's make_coeff() and make_norm().
    Exact rational arithmetic is used to invert the (small) matrix.
    """
    n = power + 1
    mat = [[Fraction(sum(k ** (i + j) for k in range(-nb, nf + 1)))
             for j in range(n)] for i in range(n)]
    inv = _invert(mat)

    norm = Fraction(1)
    for i in range(2, deriv + 1):
        norm /= i

    return array("d", [
        float(norm * sum(inv[deriv][k] * Fraction(j - nb) ** k
                         for k in range(n)))
        for j in range(nb + nf + 1)
    ])


def _invert(mat: list) -> list:
    # Gauss-Jordan elimination
    n = len(mat)
    aug = [row[:] + [Fraction(int(i == j)) for j in range(n)]
           for i, row in enumerate(mat)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


cdef void _filter(const double[:] values, const double[:] coeff,
                  double[:] out, int nb, int nf, bint keep_edges) noexcept nogil:
    cdef Py_ssize_t i, j, n = values.shape[0]
    cdef double x

    for i in range(n):
        if i < nb or i >= n - nf:
            out[i] = values[i] if keep_edges else 0.0
        else:
            x = 0.0
            for j in range(-nb, nf + 1):
                x += coeff[j + nb] * values[i + j]
            out[i] = x


def smooth(values, int derivative, int smooth_frame):
    """
    Savitzky-Golay filter (polynomial of order 2) over a symmetric window
    of `smooth_frame` points on each side.
    Like TISEAN, the first/last `smooth_frame` values are left unfiltered
    (derivative = 0) or set to zero (derivative > 0).
    Return None if the series is too short.
    """
    cdef Py_ssize_t n = len(values)
    if n < 2 * smooth_frame:
        smooth_frame = n // 2
    elif smooth_frame == 0:
        smooth_frame = 1

    if _POWER >= 2 * smooth_frame + 1 or derivative > _POWER:
        # System is underdetermined
        return None

    cdef array src = array("d", values)
    cdef array out = clone(_template, n, zero=False)
    cdef array coeff = _coefficients(smooth_frame, smooth_frame, derivative)
    cdef double[:] src_v = src, out_v = out, coeff_v = coeff

    with nogil:
        _filter(src_v, coeff_v, out_v, smooth_frame, smooth_frame,
                derivative == 0)

    return out.tolist()
