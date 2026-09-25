# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/matthiasblum/idrpred/blob/main/LICENSE

"""
IUPred (long and short disorder), reimplemented from iupred_string.c.
"""

import os

from cpython.array cimport array, clone
from libc.stdlib cimport malloc, free

_DATADIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                        "iupred")
_AA = "GAVLIFPSTCMWYNQDEKRH"

cdef enum:
    AAN = 20
    # Minimum/maximum sequence separation for the energy estimation
    LC = 1
    # Half-width of the smoothing window
    WS = 10

cdef int[256] _CODES
for _i in range(256):
    _CODES[_i] = -1
for _i, _aa in enumerate(_AA):
    _CODES[ord(_aa)] = _i
    _CODES[ord(_aa.lower())] = _i


cdef class _Parameters:
    cdef double[AAN * AAN] energy
    cdef array distro
    cdef double min, max, step
    cdef int nb
    cdef int uc
    cdef bint pad_ends
    cdef double end_energy

    def __init__(self, str matrix, str histogram, int uc, bint pad_ends,
                 double end_energy=0):
        self.uc = uc
        self.pad_ends = pad_ends
        self.end_energy = end_energy
        self._read_matrix(os.path.join(_DATADIR, matrix))
        self._read_histogram(os.path.join(_DATADIR, histogram))

    def _read_matrix(self, str path):
        cdef int i
        for i in range(AAN * AAN):
            self.energy[i] = 0

        with open(path, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                # Fixed-width columns: row index, column index, value
                p1 = int(line[:8].split()[0])
                p2 = int(line[8:17].split()[0])
                self.energy[p1 * AAN + p2] = float(line[17:].split()[0])

        if self.energy[9 * AAN + 9] < 0:
            for i in range(AAN * AAN):
                self.energy[i] *= -1

    def _read_histogram(self, str path):
        with open(path, "rt") as fh:
            _, v_min, v_max, nb = fh.readline().split()[:4]
            self.min = float(v_min)
            self.max = float(v_max)
            self.nb = int(nb)
            self.distro = array("d", [0.0] * self.nb)

            # Like the original code, comment lines count as histogram bins
            for i in range(self.nb):
                line = fh.readline()
                if not line:
                    break
                elif line[0] == "#":
                    continue
                self.distro[i] = float(line.split()[4])

        self.step = (self.max - self.min) / self.nb


cdef void _iupred(const int *codes, int naa, _Parameters params,
                  double *eprof, double *smp, double *en) noexcept nogil:
    cdef int i, j, a1, a2, p, d
    cdef double n2
    cdef const double *cc = params.energy
    cdef const double *distro = <const double *> params.distro.data.as_doubles
    cdef double v_min = params.min, v_max = params.max, step = params.step
    cdef int uc = params.uc

    for i in range(naa):
        eprof[i] = 0
        smp[i] = 0
        en[i] = 0

    # Estimated pairwise interaction energy of each residue
    for i in range(naa):
        a1 = codes[i]
        if a1 < 0:
            continue

        n2 = 0
        for j in range(max(0, i - uc + 1), min(naa, i + uc)):
            d = i - j if i > j else j - i
            if d > LC and d < uc:
                a2 = codes[j]
                if a2 < 0:
                    continue
                eprof[i] += cc[a1 * AAN + a2]
                n2 += 1

        eprof[i] /= n2

    # Smoothing
    if not params.pad_ends:
        for i in range(naa):
            n2 = 0
            # The original code reads one element past the end of the array
            # (index naa), which we consider to be zero
            for j in range(max(0, i - WS), min(naa, i + WS + 1) + 1):
                if j < naa:
                    smp[i] += eprof[j]
                n2 += 1
            smp[i] /= n2
    else:
        for i in range(naa):
            n2 = 0
            for j in range(i - WS, i + WS):
                if j < 0 or j >= naa:
                    smp[i] += params.end_energy
                else:
                    smp[i] += eprof[j]
                n2 += 1
            smp[i] /= n2

    # Energy to disorder probability
    for i in range(naa):
        if smp[i] <= v_min + 2 * step:
            en[i] = 1
        if smp[i] >= v_max - 2 * step:
            en[i] = 0
        if v_min + 2 * step < smp[i] < v_max - 2 * step:
            p = <int> ((smp[i] - v_min) * (1.0 / step))
            en[i] = distro[p]


cdef _Parameters _LONG = _Parameters("ss", "histo", uc=100, pad_ends=False)
cdef _Parameters _SHORT = _Parameters("ss_casp", "histo_casp", uc=25,
                                      pad_ends=True, end_energy=-1.26)


def _run(str sequence, _Parameters params):
    cdef int naa = len(sequence), i
    if naa == 0:
        return None

    cdef bytes seq = sequence.encode("ascii", "replace")
    cdef int *codes = <int *> malloc(naa * sizeof(int))
    cdef double *buf = <double *> malloc(3 * naa * sizeof(double))
    if codes == NULL or buf == NULL:
        free(codes)
        free(buf)
        raise MemoryError()

    try:
        for i in range(naa):
            codes[i] = _CODES[seq[i]]

        with nogil:
            _iupred(codes, naa, params, buf, buf + naa, buf + 2 * naa)

        return [buf[2 * naa + i] for i in range(naa)]
    finally:
        free(codes)
        free(buf)


def predict_long(str sequence):
    return _run(sequence, _LONG)


def predict_short(str sequence):
    return _run(sequence, _SHORT)
