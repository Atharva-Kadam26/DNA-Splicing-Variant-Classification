# Data

This project uses SpliceVarDB-derived genetic variant annotations
and the human reference genome (GRCh38/hg38).

## Data processing

The preprocessing workflow includes:

1. Cleaning SpliceVarDB variant annotations.
2. Extracting reference DNA sequences.
3. Constructing reference and alternate allele sequences.
4. Validating allele differences.
5. Creating gene-separated training, validation and test splits.

## Dataset summary

- Matched variants: 23,492
- Sequence length: 501 bp
- Positive class: SPLICE (1)
- Negative class: NORMAL (0)
- Splitting strategy: Gene-separated

## Data availability

Large reference genome files, raw genomic resources and
processed CSV datasets are not included in this repository.

Users should obtain the appropriate source datasets and reference
genome files before running the preprocessing scripts.
