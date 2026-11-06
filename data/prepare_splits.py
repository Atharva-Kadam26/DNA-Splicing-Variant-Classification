
import csv
import random
from collections import defaultdict, Counter

INPUT_FILE = "variant_sequences.csv"
ANNOTATION_FILE = "refGene.txt"
OUTPUT_FILE = "variant_sequences_with_splits.csv"

random.seed(42)

# Index annotations by transcript and chromosome.
annotations = defaultdict(list)

with open(ANNOTATION_FILE) as f:
    for line in f:
        fields = line.rstrip().split("\t")
        if len(fields) < 13:
            continue

        accession = fields[1].split(".")[0]
        chromosome = fields[2]
        start = int(fields[4])
        end = int(fields[5])
        gene = fields[12]

        annotations[(accession, chromosome)].append(
            (start, end, gene)
        )

rows = []
unmapped = 0
ambiguous = 0

with open(INPUT_FILE) as f:
    for row in csv.DictReader(f):
        accession = row["gene"].split(".")[0]
        chrom = row["chrom"]
        start = int(row["start"])

        candidate_genes = set()

        for tx_start, tx_end, gene in annotations.get(
            (accession, chrom), []
        ):
            if tx_start <= start < tx_end:
                candidate_genes.add(gene)

        if not candidate_genes:
            unmapped += 1
            continue

        if len(candidate_genes) != 1:
            ambiguous += 1
            continue

        row["gene_symbol"] = next(iter(candidate_genes))
        rows.append(row)

# Group variants by gene.
gene_rows = defaultdict(list)

for row in rows:
    gene_rows[row["gene_symbol"]].append(row)

genes = list(gene_rows)
random.shuffle(genes)

n_train = int(0.70 * len(genes))
n_val = int(0.15 * len(genes))

train_genes = set(genes[:n_train])
val_genes = set(genes[n_train:n_train + n_val])

for row in rows:
    gene = row["gene_symbol"]

    if gene in train_genes:
        row["split"] = "train"
    elif gene in val_genes:
        row["split"] = "validation"
    else:
        row["split"] = "test"

if rows:
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

print("Input variants:", sum(1 for _ in open(INPUT_FILE)) - 1)
print("Rows retained:", len(rows))
print("Unique genes:", len(gene_rows))
print("Unmapped variants:", unmapped)
print("Ambiguous variants:", ambiguous)
print("Split counts:", Counter(row["split"] for row in rows))

for split in ("train", "validation", "test"):
    print(
        split + " labels:",
        Counter(row["label"] for row in rows if row["split"] == split)
    )

print("Saved:", OUTPUT_FILE)

