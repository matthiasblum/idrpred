# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/matthiasblum/idrpred/blob/main/LICENSE
# Copyright (c) Matthias Blum <mblum@ebi.ac.uk>

"""
SEG, reimplemented from seg.c, with default parameters
(window: 12, trigger complexity: 2.2, extension complexity: 2.5)
and the `-x` option (masking of low-complexity segments).
"""

from libc.math cimport log, lgamma
from libc.stdlib cimport malloc, realloc, free

cdef extern from "_seg/lnfac.h":
    # ln(n!) for n in [0, 10000], rounded to six decimals
    const double lnfac[]

cdef enum:
    LNFAC_SIZE = 10001
    WINDOW = 12
    DOWNSET = (WINDOW + 1) // 2 - 1
    UPSET = WINDOW - DOWNSET
    MAXTRIM = 100
    # Residue codes: 0-19 amino acids, then residues kept in the sequence
    # but ignored in compositions, then punctuation ('-')
    NAA = 20
    OTHER = 20
    DASH = 21

cdef double LOCUT = 2.2
cdef double HICUT = 2.5
cdef double LN2 = 0.69314718055994530941723212145818
cdef double LN20 = 2.9957322735539909

cdef double[WINDOW + 1] _ENTRAY
cdef int _k
for _k in range(1, WINDOW + 1):
    _x = _k / <double> WINDOW
    _ENTRAY[_k] = -_x * log(_x) / LN2

cdef int[256] _CODES
for _k in range(256):
    # Residues stripped when reading the sequence
    _CODES[_k] = -1
for _k, _aa in enumerate("ACDEFGHIKLMNPQRSTVWY"):
    _CODES[ord(_aa)] = _k
    _CODES[ord(_aa.lower())] = _k
for _aa in "BUXZ*":
    _CODES[ord(_aa)] = OTHER
    _CODES[ord(_aa.lower())] = OTHER
_CODES[ord("-")] = DASH


ctypedef struct Segments:
    int *begin
    int *end
    int n
    int size


cdef int _append(Segments *segs, int begin, int end) noexcept nogil:
    cdef int size
    cdef int *p
    if segs.n == segs.size:
        size = segs.size * 2 if segs.size else 16
        p = <int *> realloc(segs.begin, size * sizeof(int))
        if p == NULL:
            return -1
        segs.begin = p
        p = <int *> realloc(segs.end, size * sizeof(int))
        if p == NULL:
            return -1
        segs.end = p
        segs.size = size

    segs.begin[segs.n] = begin
    segs.end[segs.n] = end
    segs.n += 1
    return 0


cdef inline double _lnfac(int n) noexcept nogil:
    if n < LNFAC_SIZE:
        return lnfac[n]
    return lgamma(n + 1.0)


cdef void _state(const int *comp, int *sv) noexcept nogil:
    # State vector: non-zero counts sorted in descending order, zero-padded
    cdef int i, j, c, n = 0
    for i in range(NAA):
        c = comp[i]
        if c == 0:
            continue
        j = n
        while j > 0 and sv[j - 1] < c:
            sv[j] = sv[j - 1]
            j -= 1
        sv[j] = c
        n += 1
    for i in range(n, NAA + 1):
        sv[i] = 0


cdef double _entropy(const int *comp) noexcept nogil:
    cdef int sv[NAA + 1]
    cdef int i, total = 0
    cdef double ent = 0, xtotrecip, xsv

    _state(comp, sv)
    i = 0
    while sv[i] != 0:
        total += sv[i]
        i += 1

    if total == WINDOW:
        i = 0
        while sv[i] != 0:
            ent += _ENTRAY[sv[i]]
            i += 1
        return ent
    elif total == 0:
        return 0

    xtotrecip = 1. / <double> total
    i = 0
    while sv[i] != 0:
        xsv = sv[i]
        ent += xsv * log(xsv * xtotrecip)
        i += 1
    return -ent * xtotrecip / LN2


cdef double _lnperm(const int *sv, int tot) noexcept nogil:
    cdef double ans = _lnfac(tot)
    cdef int i = 0
    while sv[i] != 0:
        ans -= _lnfac(sv[i])
        i += 1
    return ans


cdef double _lnass(const int *sv) noexcept nogil:
    cdef double ans = _lnfac(20)
    cdef int svi, svim1, klass, total, i, p

    if sv[0] == 0:
        return ans

    total = 20
    klass = 1
    svim1 = sv[0]
    i = 0
    p = 0
    while True:
        i += 1
        if i == 20:
            ans -= _lnfac(klass)
            break

        p += 1
        svi = sv[p]
        if svi == svim1:
            klass += 1
        else:
            total -= klass
            ans -= _lnfac(klass)
            if svi == 0:
                ans -= _lnfac(total)
                break
            klass = 1
        svim1 = svi
    return ans


cdef double _getprob(const int *comp, int total) noexcept nogil:
    cdef int sv[NAA + 1]
    _state(comp, sv)
    return _lnass(sv) + _lnperm(sv, total) - (<double> total) * LN20


cdef inline void _count(const int *codes, int start, int length,
                        int *comp) noexcept nogil:
    cdef int i, c
    for i in range(NAA):
        comp[i] = 0
    for i in range(start, start + length):
        c = codes[i]
        if c < NAA:
            comp[c] += 1


cdef inline void _shift(const int *codes, int start, int length,
                        int *comp) noexcept nogil:
    # Slide the window [start, start+length) by one residue
    cdef int c = codes[start]
    if c < NAA:
        comp[c] -= 1
    c = codes[start + length]
    if c < NAA:
        comp[c] += 1


cdef double *_seqent(const int *codes, int length,
                     bint punctuation) noexcept nogil:
    cdef double *H
    cdef int comp[NAA]
    cdef int i, j, first, last, start
    cdef bint dash

    if WINDOW > length:
        return NULL

    H = <double *> malloc(length * sizeof(double))
    if H == NULL:
        return NULL

    for i in range(length):
        H[i] = -1.

    first = DOWNSET
    last = length - UPSET
    _count(codes, 0, WINDOW, comp)

    for i in range(first, last + 1):
        start = i - first
        if punctuation:
            dash = False
            for j in range(start, start + WINDOW):
                if codes[j] == DASH:
                    dash = True
                    break
            if dash:
                H[i] = -1
                if start + WINDOW < length:
                    _shift(codes, start, WINDOW, comp)
                continue

        H[i] = _entropy(comp)
        if start + WINDOW < length:
            _shift(codes, start, WINDOW, comp)

    return H


cdef int _findlo(int i, int limit, const double *H) noexcept nogil:
    cdef int j = i
    while j >= limit:
        if H[j] == -1 or H[j] > HICUT:
            break
        j -= 1
    return j + 1


cdef int _findhi(int i, int limit, const double *H) noexcept nogil:
    cdef int j = i
    while j <= limit:
        if H[j] == -1 or H[j] > HICUT:
            break
        j += 1
    return j - 1


cdef void _trim(const int *codes, int length, int *leftend,
                int *rightend) noexcept nogil:
    # codes: segment [leftend, rightend] of `length` residues
    cdef int comp[NAA]
    cdef double prob, minprob = 1.
    cdef int lend = 0, rend = length - 1, minlen = 1
    cdef int n, i

    if length - MAXTRIM > minlen:
        minlen = length - MAXTRIM

    n = length
    while n > minlen:
        _count(codes, 0, n, comp)
        i = 0
        while True:
            prob = _getprob(comp, n)
            if prob < minprob:
                minprob = prob
                lend = i
                rend = n + i - 1
            if i + n >= length:
                break
            _shift(codes, i, n, comp)
            i += 1
        n -= 1

    leftend[0] = leftend[0] + lend
    rightend[0] = rightend[0] - (length - rend - 1)


cdef int _segseq(const int *codes, int length, bint punctuation,
                 int offset, Segments *segs) noexcept nogil:
    cdef double *H
    cdef int first, last, lowlim, loi, hii, i
    cdef int leftend, rightend, lend, rend

    H = _seqent(codes, length, punctuation)
    if H == NULL:
        return 0

    first = DOWNSET
    last = length - UPSET
    lowlim = first

    i = first
    while i <= last:
        if H[i] <= LOCUT and H[i] != -1:
            loi = _findlo(i, lowlim, H)
            hii = _findhi(i, last, H)

            leftend = loi - DOWNSET
            rightend = hii + UPSET - 1

            _trim(codes + leftend, rightend - leftend + 1,
                  &leftend, &rightend)

            if i + UPSET - 1 < leftend:
                # Check for trigger window in left trim
                lend = loi - DOWNSET
                rend = leftend - 1
                if _segseq(codes + lend, rend - lend + 1, False,
                           offset + lend, segs) != 0:
                    free(H)
                    return -1

            if _append(segs, leftend + offset, rightend + offset) != 0:
                free(H)
                return -1

            i = min(hii, rightend + DOWNSET)
            lowlim = i + 1
        i += 1

    free(H)
    return 0


cdef void _mergesegs(Segments *segs) noexcept nogil:
    # Merge overlapping segments (in place)
    cdef int k = 0, j
    if segs.n == 0:
        return

    for j in range(1, segs.n):
        if segs.end[k] >= segs.begin[j]:
            segs.end[k] = segs.end[j]
        else:
            k += 1
            segs.begin[k] = segs.begin[j]
            segs.end[k] = segs.end[j]
    segs.n = k + 1


def _mask(const int[:] codes, bint punctuation):
    cdef Segments segs
    cdef int i, j, rc
    cdef int length = codes.shape[0]
    segs.begin = segs.end = NULL
    segs.n = segs.size = 0

    try:
        with nogil:
            rc = _segseq(&codes[0], length, punctuation, 0, &segs) \
                if length else 0
            _mergesegs(&segs)
        if rc != 0:
            raise MemoryError()

        mask = [0] * length
        for i in range(segs.n):
            for j in range(segs.begin[i], segs.end[i] + 1):
                mask[j] = 1
        return mask
    finally:
        free(segs.begin)
        free(segs.end)


def predict(str sequence):
    """
    Return 1 for residues in low-complexity segments, 0 otherwise.
    Like SEG, only the 20 amino acids, B, U, X, Z, '*', and '-' are
    considered; other characters are ignored, and are masked if the
    residues surrounding them are.
    """
    from array import array

    cdef bytes seq = sequence.encode("ascii", "replace")
    cdef int i, c
    kept = []
    codes = array("i")
    for i in range(len(seq)):
        c = _CODES[seq[i]]
        if c >= 0:
            kept.append(i)
            codes.append(c)

    # Windows with a dash are ignored
    kept_mask = _mask(codes, DASH in codes)
    mask = [0] * len(seq)
    for i, c in zip(kept, kept_mask):
        mask[i] = c

    if len(kept) < len(seq):
        # Mask stripped residues within masked segments
        for j in range(1, len(kept)):
            if kept_mask[j - 1] and kept_mask[j]:
                for i in range(kept[j - 1] + 1, kept[j]):
                    mask[i] = 1

    return mask
