# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/matthiasblum/idrpred/blob/main/LICENSE

"""
ESpritz (DisProt, NMR, and X-ray flavours), reimplemented from the
original binaries (disbinD, disbinN, disbinX)

Each flavour is an ensemble of two-stage bidirectional recursive neural
networks (BRNN). The first stage predicts disorder from the sequence
(one-hot encoding, or Atchley factors). The second stage refines
these predictions using averages of the first stage's output
over neighbouring blocks of residues. The disorder score is the
second stage's output, averaged over the models of the ensemble.
"""

import os

from libc.math cimport exp, sqrt, tanh
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy, memset

_DATADIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                        "espritz")
# Residue codes (alphabetical order)
_AA = "ACDEFGHIKLMNPQRSTVWY"


ctypedef struct Layer:
    int ny          # number of outputs
    int nin         # number of inputs
    int ncat        # number of inputs from the "categorical" input vector
    bint softmax    # softmax (otherwise tanh) activation
    double *w       # weights: ny * nin
    double *b       # biases: ny


ctypedef struct Network:
    Layer hidden
    Layer output


ctypedef struct BRNN:
    int nu          # size of the input vector of each residue
    int ny          # number of outputs
    int context     # half-width of the input window
    bint use_input  # the output network uses the input window
    int nf, nb      # size of forward and backward states
    int cof, cob    # half-width of the forward/backward state windows
    int stride      # distance between states in the windows
    int shortcuts   # number of previous states fed to state networks
    Network net_out
    Network net_f
    Network net_b


ctypedef struct Model:
    int nu, ny
    int nblocks     # number of blocks on each side of a residue (stage 2)
    int halfwidth   # half-width of blocks
    BRNN stage1
    BRNN stage2


cdef class _Tokens:
    cdef list tokens
    cdef Py_ssize_t i

    def __init__(self, str path):
        with open(path, "rt") as fh:
            self.tokens = fh.read().split()
        self.i = 0

    cdef int next_int(self) except? -1:
        v = int(self.tokens[self.i])
        self.i += 1
        return v

    cdef double next_float(self) except? -1:
        v = float(self.tokens[self.i])
        self.i += 1
        return v


cdef int _read_layer(_Tokens tk, Layer *layer, bint softmax) except -1:
    cdef int ny = tk.next_int()
    cdef int nic = tk.next_int()
    cdef int ni = tk.next_int()
    cdef int i, j, k, y, n = 0
    nk = [tk.next_int() for _ in range(nic + ni)]

    layer.ny = ny
    layer.nin = sum(nk)
    layer.ncat = sum(nk[:nic])
    layer.softmax = softmax
    layer.w = <double *> malloc(ny * layer.nin * sizeof(double))
    layer.b = <double *> malloc(ny * sizeof(double))
    if layer.w == NULL or layer.b == NULL:
        raise MemoryError()

    for y in range(ny):
        j = 0
        for i in range(nic + ni):
            for k in range(nk[i]):
                layer.w[y * layer.nin + j] = tk.next_float()
                j += 1
        layer.b[y] = tk.next_float()
    return 0


cdef int _read_network(_Tokens tk, Network *net, bint softmax) except -1:
    cdef int i
    # NOut, NHid, NIc, NI, and training parameters
    for i in range(7):
        tk.next_int()
    _read_layer(tk, &net.output, softmax)
    _read_layer(tk, &net.hidden, False)
    return 0


cdef int _read_brnn(_Tokens tk, BRNN *brnn) except -1:
    brnn.nu = tk.next_int()
    brnn.ny = tk.next_int()
    tk.next_int()                   # hidden units of the output network
    brnn.context = tk.next_int()
    brnn.use_input = tk.next_int() != 0
    brnn.nf = tk.next_int()
    brnn.nb = tk.next_int()
    tk.next_int()                   # hidden units of the state networks
    brnn.cof = tk.next_int()
    brnn.cob = tk.next_int()
    brnn.stride = tk.next_int()
    brnn.shortcuts = tk.next_int()
    tk.next_int()
    _read_network(tk, &brnn.net_out, True)
    _read_network(tk, &brnn.net_f, False)
    _read_network(tk, &brnn.net_b, False)
    return 0


cdef int _read_model(str path, Model *model) except -1:
    cdef _Tokens tk = _Tokens(path)
    cdef int i
    model.nu = tk.next_int()
    model.ny = tk.next_int()
    tk.next_int()
    tk.next_int()
    header = [tk.next_int() for _ in range(11)]
    model.nblocks = header[5]
    model.halfwidth = header[6]
    # Classification thresholds (unused)
    for i in range(model.ny - 1):
        tk.next_float()
    _read_brnn(tk, &model.stage1)
    _read_brnn(tk, &model.stage2)
    return 0


# Forward pass ----------------------------------------------------------------

cdef void _layer_forward(const Layer *layer, const double *inputs,
                         const double *reals, double *out) noexcept nogil:
    """
    Inputs are the concatenation of `ncat` values from `inputs`
    and `nin - ncat` values from `reals`.
    """
    cdef int y, j, imax = 0
    cdef double a, s = 0, vmax
    cdef const double *w
    cdef bint overflow = False

    for y in range(layer.ny):
        a = layer.b[y]
        w = layer.w + y * layer.nin
        for j in range(layer.ncat):
            a += w[j] * inputs[j]
        w += layer.ncat
        for j in range(layer.nin - layer.ncat):
            a += w[j] * reals[j]
        out[y] = a

    if not layer.softmax:
        for y in range(layer.ny):
            out[y] = tanh(out[y])
        return

    vmax = out[0]
    for y in range(layer.ny):
        if out[y] > 85:
            overflow = True
        else:
            s += exp(out[y])
        if out[y] > vmax:
            vmax = out[y]
            imax = y

    if overflow:
        for y in range(layer.ny):
            out[y] = 0.0001
        out[imax] = (layer.ny - 1) * -0.0001 + 1.0
    else:
        for y in range(layer.ny):
            out[y] = exp(out[y]) / s

    for y in range(layer.ny):
        if out[y] < 0.0001:
            out[y] = 0.0001


cdef void _network_forward(const Network *net, const double *inputs,
                           const double *reals, double *hidden,
                           double *out) noexcept nogil:
    _layer_forward(&net.hidden, inputs, reals, hidden)
    _layer_forward(&net.output, NULL, hidden, out)


cdef void _window(const BRNN *brnn, const double *inputs, int t, int length,
                  double *out) noexcept nogil:
    cdef int d, p, n = brnn.nu
    for d in range(-brnn.context, brnn.context + 1):
        p = t + d
        if 1 <= p <= length:
            memcpy(out, inputs + p * n, n * sizeof(double))
        else:
            memset(out, 0, n * sizeof(double))
        out += n


cdef int _brnn_predict(const BRNN *brnn, const double *inputs, int length,
                       double *outputs) noexcept nogil:
    """
    inputs: (length + 1) * nu values (position 0 is unused)
    outputs: (length + 1) * ny values (position 0 is unused)
    """
    cdef int t, d, k, p, nr
    cdef int nf = brnn.nf, nb = brnn.nb
    cdef int ns = brnn.shortcuts if brnn.shortcuts > 1 else 1
    cdef int nwin = (2 * brnn.context + 1) * brnn.nu
    cdef int nout = (2 * brnn.cof + 1) * nf + (2 * brnn.cob + 1) * nb
    cdef double *ff = <double *> calloc((length + 2) * nf, sizeof(double))
    cdef double *bb = <double *> calloc((length + 2) * nb, sizeof(double))
    cdef double *win = <double *> malloc(nwin * sizeof(double))
    cdef double *reals = <double *> malloc(
        max(ns * max(nf, nb), nout) * sizeof(double))
    cdef double *hidden = <double *> malloc(
        max(max(brnn.net_f.hidden.ny, brnn.net_b.hidden.ny),
            brnn.net_out.hidden.ny) * sizeof(double))
    cdef int rc = 0

    if (ff == NULL or bb == NULL or win == NULL or reals == NULL
            or hidden == NULL):
        rc = -1
    else:
        # Forward states: ff[0] is the initial (null) state
        for t in range(1, length + 1):
            _window(brnn, inputs, t, length, win)
            memcpy(reals, ff + (t - 1) * nf, nf * sizeof(double))
            for k in range(2, brnn.shortcuts + 1):
                if t - k >= 0:
                    memcpy(reals + (k - 1) * nf, ff + (t - k) * nf,
                           nf * sizeof(double))
                else:
                    memset(reals + (k - 1) * nf, 0, nf * sizeof(double))
            _network_forward(&brnn.net_f, win, reals, hidden, ff + t * nf)

        # Backward states: bb[length+1] is the initial (null) state
        for t in range(length, 0, -1):
            _window(brnn, inputs, t, length, win)
            memcpy(reals, bb + (t + 1) * nb, nb * sizeof(double))
            for k in range(2, brnn.shortcuts + 1):
                if t + k <= length + 1:
                    memcpy(reals + (k - 1) * nb, bb + (t + k) * nb,
                           nb * sizeof(double))
                else:
                    memset(reals + (k - 1) * nb, 0, nb * sizeof(double))
            _network_forward(&brnn.net_b, win, reals, hidden, bb + t * nb)

        # Outputs
        for t in range(1, length + 1):
            if brnn.use_input:
                _window(brnn, inputs, t, length, win)
            else:
                memset(win, 0, nwin * sizeof(double))
            nr = 0
            for d in range(-brnn.cof, brnn.cof + 1):
                p = t + d * brnn.stride
                if 0 <= p <= length:
                    memcpy(reals + nr, ff + p * nf, nf * sizeof(double))
                else:
                    memset(reals + nr, 0, nf * sizeof(double))
                nr += nf
            for d in range(-brnn.cob, brnn.cob + 1):
                p = t + d * brnn.stride
                if 1 <= p <= length + 1:
                    memcpy(reals + nr, bb + p * nb, nb * sizeof(double))
                else:
                    memset(reals + nr, 0, nb * sizeof(double))
                nr += nb
            _network_forward(&brnn.net_out, win, reals, hidden,
                             outputs + t * brnn.ny)

    free(ff)
    free(bb)
    free(win)
    free(reals)
    free(hidden)
    return rc


cdef int _model_predict(const Model *model, const int *codes, int length,
                        const double *atchley, int nmodels,
                        double *ensemble) noexcept nogil:
    cdef int t, k, d, p, j, center
    cdef int nu = model.nu, ny = model.ny
    cdef int width = 2 * model.halfwidth + 1
    cdef int dim2 = 2 * ny * (model.nblocks + 1)
    cdef double *in1 = <double *> calloc((length + 1) * nu, sizeof(double))
    cdef double *out1 = <double *> calloc((length + 1) * ny, sizeof(double))
    cdef double *in2 = <double *> calloc((length + 1) * dim2, sizeof(double))
    cdef double *out2 = <double *> calloc((length + 1) * ny, sizeof(double))
    cdef double *row
    cdef int rc = 0

    if in1 == NULL or out1 == NULL or in2 == NULL or out2 == NULL:
        rc = -1
    else:
        # Stage 1: input encoding
        for t in range(1, length + 1):
            if codes[t] < 0:
                continue
            if nu == 5:
                for k in range(5):
                    in1[t * nu + k] += atchley[codes[t] * 5 + k]
            else:
                in1[t * nu + codes[t]] = 1.0

        rc = _brnn_predict(&model.stage1, in1, length, out1)

    if rc == 0:
        # Stage 2: first stage's output at the residue, and averaged
        # over 2*nblocks+1 blocks of residues centered on the residue
        for t in range(1, length + 1):
            row = in2 + t * dim2
            for k in range(ny):
                row[k] = out1[t * ny + k]
            for d in range(-model.nblocks, model.nblocks + 1):
                center = t + d * width
                j = ny * (model.nblocks + d + 1)
                for p in range(center - model.halfwidth,
                               center + model.halfwidth + 1):
                    if 1 <= p <= length:
                        for k in range(ny):
                            row[j + k] += out1[p * ny + k] / width

        rc = _brnn_predict(&model.stage2, in2, length, out2)

    if rc == 0:
        for t in range(1, length + 1):
            for k in range(ny):
                ensemble[t * ny + k] += out2[t * ny + k] / nmodels

    free(in1)
    free(out1)
    free(in2)
    free(out2)
    return rc


# Python interface ------------------------------------------------------------

cdef int[256] _CODES
for _i in range(256):
    _CODES[_i] = -1
for _i, _aa in enumerate(_AA):
    _CODES[ord(_aa)] = _i


cdef double[100] _ATCHLEY


def _load_atchley():
    # Atchley factors, normalised (L2 norm over the 20 amino acids)
    cdef int i, k
    with open(os.path.join(_DATADIR, "atchley"), "rt") as fh:
        rows = [list(map(float, line.split()[:5])) for line in fh
                if line.strip()]
    for k in range(5):
        norm = 0.0
        for i in range(20):
            norm += rows[i][k] * rows[i][k]
        norm = sqrt(norm)
        for i in range(20):
            _ATCHLEY[i * 5 + k] = rows[i][k] / norm


_load_atchley()


cdef class _Ensemble:
    cdef Model *models
    cdef int n
    cdef int ny

    def __cinit__(self, str name):
        with open(os.path.join(_DATADIR, name), "rt") as fh:
            tokens = fh.read().split()

        self.n = int(tokens[0])
        self.models = <Model *> calloc(self.n, sizeof(Model))
        if self.models == NULL:
            raise MemoryError()

        for i in range(self.n):
            _read_model(os.path.join(_DATADIR, tokens[i + 1]),
                        &self.models[i])
        self.ny = self.models[0].ny

    def __dealloc__(self):
        # Models are loaded once and kept for the lifetime of the process
        pass

    def predict(self, str sequence):
        cdef int length = len(sequence), i, rc = 0
        if length == 0:
            return []

        cdef bytes seq = sequence.encode("ascii", "replace")
        cdef int *codes = <int *> malloc((length + 1) * sizeof(int))
        cdef double *ensemble = <double *> calloc((length + 1) * self.ny,
                                                  sizeof(double))
        if codes == NULL or ensemble == NULL:
            free(codes)
            free(ensemble)
            raise MemoryError()

        try:
            codes[0] = -1
            for i in range(length):
                codes[i + 1] = _CODES[seq[i]]

            with nogil:
                for i in range(self.n):
                    rc = _model_predict(&self.models[i], codes, length,
                                        _ATCHLEY, self.n, ensemble)
                    if rc != 0:
                        break

            if rc != 0:
                raise MemoryError()

            # Probability of the second class (disorder)
            return [ensemble[i * self.ny + 1] for i in range(1, length + 1)]
        finally:
            free(codes)
            free(ensemble)


_DISPROT = _Ensemble("ensembleD")
_NMR = _Ensemble("ensembleN")
_XRAY = _Ensemble("ensembleX")


def predict_disprot(str sequence):
    return _DISPROT.predict(sequence)


def predict_nmr(str sequence):
    return _NMR.predict(sequence)


def predict_xray(str sequence):
    return _XRAY.predict(sequence)
