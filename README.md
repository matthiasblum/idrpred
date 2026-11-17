# IDRPred

[![Tests](https://github.com/matthiasblum/idrpred/actions/workflows/tests.yml/badge.svg)](https://github.com/matthiasblum/idrpred/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/713631556.svg)](https://zenodo.org/doi/10.5281/zenodo.13356721)
[![Docker Image Version (tag)](https://img.shields.io/docker/v/matblum/idrpred/latest?label=Docker)](https://hub.docker.com/r/matblum/idrpred)

IDRPred is an alternative implementation of [MobiDB-lite](https://github.com/BioComputingUP/MobiDB-lite) 
([Mehdiabadi , et al. 2025](https://doi.org/10.1093/bioinformatics/btaf297)), 
a method for identifying intrinsically disordered regions (IDRs) in proteins.
MobiDB-lite uses multiple predictors to derive a consensus, which is filtered 
for spurious short predictions in a second step.

MobiDB-lite runs the original predictors as external programs, starting about ten processes for each protein.
IDRPred reimplements these predictors in C/Cython with no external programs and no dependencies.
It reports the same disordered regions as MobiDB-lite, four to five times faster.

## Installation

```sh
pip install git+https://github.com/matthiasblum/idrpred
```

Building from source requires a C compiler.

### Docker

A Docker image of `idrpred` is available from [Docker Hub](https://hub.docker.com/r/matblum/idrpred).

## Usage

```sh
idrpred [options] [infile] [outfile]
```

Positional arguments:

* `infile`: The FASTA file of sequences to process. If `-` or not specified, read from standard input.
* `outfile`: The file of predicted intrinsically disordered regions. If `-` or not specified, write to standard output.

### Available options

| Options           | Description                                                                         |
|-------------------|-------------------------------------------------------------------------------------|
| `--format FORMAT` | Output format: `regions` (default) or `caid` (per-residue scores, see below)        |
| `--threads N`     | Process up to `N` sequences concurrently, default: `1`                              |
| `--force`         | Report a consensus as long as at least one predictor succeeded                      |
| `--skip-features` | Do not identify sequence features (composition, low complexity) within IDRs         |

Results are written in the order of the input sequences, regardless of the number of threads.

### Output formats

#### Regions (default)

One line per region: sequence ID, start, end (1-based, inclusive), and feature.
Intrinsically disordered regions have the feature `-`, and are followed by the sequence features found within them:

* Composition features ([Das and Pappu 2013](https://doi.org/10.1073/pnas.1304749110)): 
  `Polyampholyte`, `Positive Polyelectrolyte`, `Negative Polyelectrolyte`, `Cysteine-rich`, `Proline-rich`, `Glycine-rich`, and `Polar`.
  They are mutually exclusive: each residue has at most one.
* `Low complexity`: low-complexity regions identified by SEG ([Wootton 1994](https://doi.org/10.1016/0097-8485(94)85023-2)).
  They are reported independently of composition features, and may overlap them.

#### CAID

Per-residue scores and states, following the format of the [Critical Assessment of protein Intrinsic Disorder](https://caid.idpcentral.org/):

```
>sequence ID
1	M	0.875	1
2	S	0.750	1
```

Columns are the position, the residue, the score, and the binary state (`1`: disordered).
The score is the fraction of predictors agreeing that the residue is disordered, 
and the state is `1` if the residue belongs to a disordered region reported in the `regions` format.

## Predictors

Only predictors whose licence authorises distribution have been included in IDRPred.

| Method           | Reference                                                                 | Available |
|------------------|---------------------------------------------------------------------------|:---------:|
| ANCHOR           | [Dosztányi et al. 2009](https://doi.org/10.1093/bioinformatics/btp518)    |     ❌     |
| DisEMBL-465      | [Linding et al. 2003](https://doi.org/10.1016/j.str.2003.10.002)          |     ✔     |
| DisEMBL-HotLoops | [Linding et al. 2003](https://doi.org/10.1016/j.str.2003.10.002)          |     ✔     |
| DynaMine         | [Cilia et al. 2013](https://doi.org/10.1038/ncomms3741)                   |     ❌     |
| ESpritz-DisProt  | [Walsh et al. 2012](https://doi.org/10.1093/bioinformatics/btr682)        |     ✔     |
| ESpritz-NMR      | [Walsh et al. 2012](https://doi.org/10.1093/bioinformatics/btr682)        |     ✔     |
| ESpritz-Xray     | [Walsh et al. 2012](https://doi.org/10.1093/bioinformatics/btr682)        |     ✔     |
| FeSS             | [Piovesan et al. 2017](https://doi.org/10.1093/bioinformatics/btx085)     |     ❌     |
| GlobPlot         | [Linding et al. 2003](https://doi.org/10.1093/nar/gkg519)                 |     ✔     |
| IUPred-Long      | [Mészáros et al. 2018](https://doi.org/10.1093/nar/gky384)                |     ✔     |
| IUPred-Short     | [Mészáros et al. 2018](https://doi.org/10.1093/nar/gky384)                |     ✔     |
| JRONN            | [Yang et al. 2005](https://doi.org/10.1093/bioinformatics/bti534)         |     ❌     |
| Pfilt            | [Jones and Swindells 2002](https://doi.org/10.1016/s0968-0004(01)02039-4) |     ❌     |
| SEG              | [Wootton 1994](https://doi.org/10.1016/0097-8485(94)85023-2)              |     ✔     |
| VSL2b            | [Peng et al. 2006](https://doi.org/10.1186/1471-2105-7-208)               |     ❌     |

The predictors are reimplemented from the source code of the original programs 
(DisEMBL, IUPred, SEG, and the Savitzky-Golay filter of [TISEAN](https://www.pks.mpg.de/tisean/) 
([Hegger et al. 1999](https://doi.org/10.1063/1.166424)) used by DisEMBL and GlobPlot), 
and use their original parameters, weights, and models.
No source code is available for ESpritz: it was reimplemented from the original binaries, 
and uses the original models.
