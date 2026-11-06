# DNA Splicing Variant Classification Using CNNs and LSTMs

## Overview

This project investigates deep learning approaches for classifying DNA sequence variants associated with splicing.

Using variant annotations derived from **SpliceVarDB** and the human reference genome (GRCh38/hg38), the project constructs paired reference and alternate DNA sequences and trains deep learning models to classify variants into two categories:

- `SPLICE` (1)
- `NORMAL` (0)

The project explores Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs) to identify sequence patterns associated with splicing annotations.

## Dataset

- **Source:** SpliceVarDB-derived variant annotations
- **Reference genome:** GRCh38/hg38
- **Matched variants:** 23,492
- **Sequence length:** 501 bp
- **Input:** Paired reference and alternate DNA sequences
- **Classification:** Binary (`SPLICE` / `NORMAL`)
- **Unique genes:** 6,401

### Data splitting

A gene-separated splitting strategy was used to reduce gene overlap between training, validation, and test sets.

| Split | Samples |
|---|---:|
| Training | 16,497 |
| Validation | 3,638 |
| Test | 3,357 |
| **Total** | **23,492** |

No genes are shared between the three splits.

Large reference genome files, raw genomic resources, and processed datasets are excluded from this repository. Refer to [`data/README.md`](data/README.md) for data preparation information.

## Methodology

### 1. Data preprocessing

The preprocessing workflow includes:

1. Cleaning variant annotations.
2. Extracting reference DNA sequences.
3. Constructing reference and alternate allele sequences.
4. Validating reference and alternate alleles.
5. Creating gene-separated training, validation, and test splits.

### 2. CNN classification

The improved CNN uses paired reference and alternate sequence representations.

- One-hot encoding of DNA bases.
- Eight input channels: four reference and four alternate channels.
- Three 1D convolutional layers.
- Batch normalization and ReLU activation.
- Max pooling and dropout.
- Fully connected classification head.
- Binary cross-entropy with logits loss.
- AdamW optimization with learning-rate scheduling and early stopping.

### 3. Original LSTM classification

The original LSTM uses:

- Paired reference and alternate sequence representations.
- Eight input features per nucleotide position.
- Two-layer bidirectional LSTM.
- Hidden size of 64.
- Fully connected classification head.
- AdamW optimization and early stopping.

### 4. Mutation-aware LSTM

An additional LSTM implementation introduces a ninth input feature: a mutation-position indicator.

This feature explicitly identifies the nucleotide position where the reference and alternate sequences differ, allowing the model to incorporate mutation-location information.

The mutation-aware model is implemented separately from the original LSTM.

## Results

### CNN and original LSTM comparison

The following results were obtained on the held-out test set using a classification threshold of 0.5.

| Metric | Improved CNN | Original LSTM |
|---|---:|---:|
| Accuracy | 0.5874 | 0.5168 |
| Precision | 0.6530 | 0.5658 |
| Recall | 0.4947 | 0.4315 |
| F1-score | 0.5630 | 0.4896 |
| ROC-AUC | 0.6411 | 0.5329 |
| PR-AUC | 0.6738 | 0.5732 |

The improved CNN performed better than the original LSTM across the reported evaluation metrics.

These results suggest that the CNN learned more useful sequence patterns for this classification task under the tested configurations.

**Note:** The results above compare the improved CNN with the original LSTM. They do not represent the performance of the mutation-aware LSTM.

### Threshold analysis

The CNN was also evaluated at different classification thresholds using the validation set to examine the precision-recall trade-off.

A threshold of 0.5 was retained for the reported test-set comparison.

## Repository structure

```text
DL_IA2_splice_AI/
│
├── data/
│   ├── README.md
│   ├── build_variant_sequences.py
│   ├── clean_dataset.py
│   ├── extract_sequences.py
│   ├── inspect_mismatches.py
│   ├── prepare_splits.py
│   ├── validate_cleaned_alleles.py
│   ├── validate_variants.py
│   └── validate_variants_old.py
│
├── models/
│   ├── cnn_baseline.py
│   ├── evaluate_cnn.py
│   ├── lstm_model.py
│   ├── evaluate_lstm.py
│   ├── train_lstm_mutation_aware.py
│   └── threshold_analysis.py
│
├── results/
│   └── threshold_analysis.csv
│
├── scripts/
│   └── verify_splits.py
│
├── .gitignore
└── README.md
```

## Requirements

- Python 3
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install the required packages:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Running the project

First, obtain the required source data and reference genome, then run the preprocessing scripts.

Place the processed dataset at:

```text
data/variant_sequences_with_splits.csv
```

### Train the improved CNN

```bash
python models/cnn_baseline.py
```

### Evaluate the CNN

```bash
python models/evaluate_cnn.py
```

### Train the original LSTM

```bash
python models/lstm_model.py
```

### Train the mutation-aware LSTM

```bash
python models/train_lstm_mutation_aware.py
```

### Evaluate the LSTM

```bash
python models/evaluate_lstm.py
```

The evaluation scripts require their corresponding trained model checkpoints.

## Limitations

- Classification labels represent database annotations and should not automatically be interpreted as experimentally confirmed causal effects.
- Model performance depends on the quality and representation of the source annotations.
- Reported results reflect the tested architectures and training configurations.
- The mutation-aware LSTM requires independent test-set evaluation before its performance can be compared with the other models.

## Future directions

- Evaluate the mutation-aware LSTM on the held-out test set.
- Explore additional sequence-based deep learning architectures.
- Investigate class imbalance and decision-threshold selection.
- Improve model interpretability to identify sequence patterns relevant to splicing.

## Acknowledgements

This project uses variant information derived from SpliceVarDB and the GRCh38 human reference genome.
