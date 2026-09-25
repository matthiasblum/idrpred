# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/matthiasblum/idrpred/blob/main/LICENSE

"""
DisEMBL 1.3, reimplemented from disembl.c. 
Network weights are included verbatim.
"""

from libc.stdlib cimport malloc, free

from .smoothing import smooth

cdef extern from "_disembl/weights.h":
    const float sigmoid[256]
    const float r19_1[], r19_2[], r19_3[], r19_4[], r19_5[]
    const float b41_1[], b41_2[], b41_3[], b41_4[], b41_5[]
    const float m9_1[], m9_2[], m9_3[], m9_4[], m9_5[]
    const float m21_1[], m21_2[], m21_3[], m21_4[], m21_5[]

cdef enum:
    # Size of the alphabet (20 amino acids + padding)
    NA = 21
    PAD = 20
    # Maximum window size and number of hidden neurons
    MW = 41
    MH = 30

ALPHABET = "FIVWMLCHYAGNRTPDEQSK"
cdef int[256] _CODES
for _i in range(256):
    _CODES[_i] = -1
for _i, _aa in enumerate(ALPHABET):
    _CODES[ord(_aa)] = _i


# Cython promotes numeric literals to double: intermediate results are
# stored in float variables to round them like single-precision C code.

cdef inline float _activate(float x) noexcept nogil:
    cdef float y
    if x <= -16:
        return 0
    elif x >= 16:
        return 1
    else:
        y = x * 8
        y = y + 128
        return sigmoid[<int>y]


cdef float _feed_forward(const int *s, const float *w,
                         int nw, int nh) noexcept nogil:
    cdef float h[MH]
    cdef float o[2]
    cdef float x
    cdef int i, j

    # Shift input window to match network window size
    s += (MW - nw) // 2

    # Input to hidden layer (sparse encoding)
    for i in range(nh):
        x = w[(NA * nw + 1) * (i + 1) - 1]
        for j in range(nw):
            x += w[(NA * nw + 1) * i + NA * j + s[j]]
        h[i] = _activate(x)

    # Hidden to output layer
    for i in range(2):
        x = w[(NA * nw + 1) * nh + (nh + 1) * (i + 1) - 1]
        for j in range(nh):
            x += w[(NA * nw + 1) * nh + (nh + 1) * i + j] * h[j]
        o[i] = _activate(x)

    # Combine the scores from the two output neurons
    x = o[0] + 1
    x = x - o[1]
    return x / 2


cdef void _predict(const int *s, float *hot_loops,
                   float *remark_465) noexcept nogil:
    cdef float sr, sb, sm

    sr = 0
    sr += _feed_forward(s, r19_1, 19, 30)
    sr += _feed_forward(s, r19_2, 19, 30)
    sr += _feed_forward(s, r19_3, 19, 30)
    sr += _feed_forward(s, r19_4, 19, 30)
    sr += _feed_forward(s, r19_5, 19, 30)
    sr /= 5
    sr = 0.07387214 + 0.8020778 * sr

    sb = 0
    sb += _feed_forward(s, b41_1, 41, 5)
    sb += _feed_forward(s, b41_2, 41, 5)
    sb += _feed_forward(s, b41_3, 41, 5)
    sb += _feed_forward(s, b41_4, 41, 5)
    sb += _feed_forward(s, b41_5, 41, 5)
    sb /= 5
    sb = 0.08016882 + 0.6282424 * sb

    sm = 0
    sm += _feed_forward(s, m9_1, 9, 30)
    sm += _feed_forward(s, m9_2, 9, 30)
    sm += _feed_forward(s, m9_3, 9, 30)
    sm += _feed_forward(s, m9_4, 9, 30)
    sm += _feed_forward(s, m9_5, 9, 30)
    sm += _feed_forward(s, m21_1, 21, 30)
    sm += _feed_forward(s, m21_2, 21, 30)
    sm += _feed_forward(s, m21_3, 21, 30)
    sm += _feed_forward(s, m21_4, 21, 30)
    sm += _feed_forward(s, m21_5, 21, 30)
    sm /= 10

    # Coils (sr) are not used
    hot_loops[0] = sr * sb
    remark_465[0] = sm


cdef void _run(const int *codes, int n, float *hot_loops,
               float *remark_465) noexcept nogil:
    # Like disembl.c: the window is reversed (s[0] is the latest residue)
    cdef int s[MW]
    cdef int c, k, p

    for c in range(n):
        for k in range(MW):
            p = c + (MW - 1) // 2 - k
            s[k] = codes[p] if 0 <= p < n else PAD
        _predict(s, &hot_loops[c], &remark_465[c])


def raw_scores(str sequence):
    """
    Return the raw (unsmoothed) hot loops and REMARK-465 scores
    of the residues of the DisEMBL alphabet (other residues are skipped).
    """
    cdef bytes seq = sequence.encode("ascii", "replace")
    cdef int n = 0, i, code
    cdef int *codes = <int *> malloc(len(seq) * sizeof(int) + 1)
    cdef float *hot_loops = <float *> malloc(len(seq) * sizeof(float) + 1)
    cdef float *remark_465 = <float *> malloc(len(seq) * sizeof(float) + 1)

    if codes == NULL or hot_loops == NULL or remark_465 == NULL:
        free(codes)
        free(hot_loops)
        free(remark_465)
        raise MemoryError()

    try:
        for i in range(len(seq)):
            code = _CODES[seq[i]]
            if code >= 0:
                codes[n] = code
                n += 1

        with nogil:
            _run(codes, n, hot_loops, remark_465)

        return ([hot_loops[i] for i in range(n)],
                [remark_465[i] for i in range(n)])
    finally:
        free(codes)
        free(hot_loops)
        free(remark_465)


def predict(str sequence):
    """
    Return the smoothed hot loops and REMARK-465 scores of each residue.
    Residues not in the DisEMBL alphabet have a raw score of zero.
    """
    raw_hl, raw_rem465 = raw_scores(sequence)
    hot_loops = [0.0] * len(sequence)
    remark_465 = [0.0] * len(sequence)

    j = 0
    for i, aa in enumerate(sequence):
        if aa in ALPHABET:
            hot_loops[i] = raw_hl[j]
            remark_465[i] = raw_rem465[j]
            j += 1

    return _clip(smooth(hot_loops, 0, 8)), _clip(smooth(remark_465, 0, 8))


def _clip(scores):
    if scores is None:
        return None
    return [max(v, 0) for v in scores]
