from pathlib import Path

import pytest

import idrpred
from idrpred.idrpred import parse_fasta


DATA = Path(__file__).parent / "data"
# Human p53 (P04637)
P53 = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPP"
    "VAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQ"
    "LWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSV"
    "VVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEEN"
    "LRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG"
    "GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)


def _idrs(prediction):
    return [(s, e) for s, e, f in prediction.regions if f == "-"]


def test_p53():
    pred = idrpred.predict("p53", P53)
    # Same regions as MobiDB-lite 4.0
    assert _idrs(pred) == [(50, 96), (282, 325), (351, 393)]
    assert len(pred.consensus_scores) == len(P53)
    assert pred.consensus_states.count("1") == 47 + 44 + 43


def test_lowercase_sequence():
    upper = idrpred.predict("p53", P53)
    lower = idrpred.predict("p53", P53.lower())
    assert lower.regions == upper.regions
    assert lower.consensus_scores == upper.consensus_scores


@pytest.mark.parametrize("length", [1, 2, 3, 10, 19])
def test_short_sequences(length):
    # Too short for an IDR (minimum length: 20)
    pred = idrpred.predict("short", P53[:length])
    assert pred is not None
    assert pred.regions == []
    assert len(pred.consensus_scores) == length


def test_non_standard_residues():
    seq = P53[:50] + "U" + P53[51:100] + "XBZJO*-" + P53[107:]
    pred = idrpred.predict("p53", seq)
    assert pred is not None
    assert len(pred.consensus_scores) == len(seq)


def test_matches_mobidblite():
    # IDRs predicted by MobiDB-lite 4.0 for 200 S. cerevisiae proteins
    expected = {}
    with open(DATA / "yeast200.mobidblite-4.0.idr.tsv") as fh:
        for line in fh:
            seq_id, start, end = line.split()
            expected.setdefault(seq_id, []).append((int(start), int(end)))

    for seq_id, seq in parse_fasta(str(DATA / "yeast200.fasta")):
        pred = idrpred.predict(seq_id, seq)
        assert _idrs(pred) == expected.get(seq_id, []), seq_id

