# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/matthiasblum/idrpred/blob/main/LICENSE

import os
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import predict


@contextmanager
def open_file(path: str, mode: str):
    if path == "-":
        # Do not close standard streams
        yield sys.stdin if "r" in mode else sys.stdout
    else:
        with open(path, mode, encoding="utf-8") as fh:
            yield fh


def parse_fasta(file: str):
    seq_id = sequence = ""

    with open_file(file, "rt") as fh:
        for line in map(str.rstrip, fh):
            if line.startswith(">"):
                if seq_id and sequence:
                    yield seq_id, sequence.upper()
                seq_id = line[1:].split()[0]
                sequence = ""
            elif line:
                sequence += line

    if seq_id and sequence:
        yield seq_id, sequence.upper()


def run(file: str, threads: int, **kwargs):
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            fs = {}
            for seq_id, sequence in parse_fasta(file):
                f = executor.submit(predict, seq_id, sequence, **kwargs)
                fs[f] = (seq_id, sequence)

                if len(fs) == 1000:
                    for f in as_completed(fs):
                        seq_id, sequence = fs[f]
                        yield seq_id, sequence, f.result()

                    fs.clear()

            for f in as_completed(fs):
                seq_id, sequence = fs[f]
                yield seq_id, sequence, f.result()
    else:
        for seq_id, sequence in parse_fasta(file):
            yield seq_id, sequence, predict(seq_id, sequence, **kwargs)


def main():
    script = os.path.relpath(__file__)

    description = "A consensus-based predictor of intrinsically " \
                  "disordered regions in proteins."
    parser = ArgumentParser(prog=f"python {os.path.basename(script)}",
                            description=description)
    parser.add_argument("infile", nargs="?", default="-",
                        help="A file of sequences in FASTA format.")
    parser.add_argument("outfile", nargs="?", default="-",
                        help="Write the output of infile to outfile.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Generate consensus as long as at least "
                             "one predictor did not fail.")
    parser.add_argument("--skip-features", dest="find_features",
                        action="store_false", default=True,
                        help="Do not indentify sequence features, "
                             "such as domains of low complexity.")
    parser.add_argument("--format", choices=["regions", "caid"],
                        default="regions",
                        help="Output format: disordered regions and their "
                             "features (default), or per-residue scores "
                             "and states in CAID format.")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of parallel threads, default: 1.")
    args = parser.parse_args()

    if args.infile != "-" and not os.path.isfile(args.infile):
        parser.error(f"cannot open '{args.infile}': no such file")

    with open_file(args.outfile, "wt") as outfile:
        for seq_id, sequence, prediction in run(
                args.infile, args.threads,
                force=args.force,
                find_features=(args.find_features
                               and args.format == "regions")):
            if prediction is None:
                print(f"error in {seq_id}", file=sys.stderr)
                continue

            if args.format == "regions":
                for start, end, feature in prediction.regions:
                    outfile.write(f"{seq_id}\t{start}\t{end}\t{feature}\n")
                continue

            outfile.write(f">{seq_id}\n")
            for i, (aa, score, state) in enumerate(
                    zip(sequence, prediction.consensus_scores,
                        prediction.consensus_states)):
                outfile.write(f"{i + 1}\t{aa}\t{score:.3f}\t{state}\n")

if __name__ == "__main__":
    main()
