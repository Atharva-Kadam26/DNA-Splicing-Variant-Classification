
import csv
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
filename = project_dir / "data" / "variant_sequences_with_splits.csv"

genes = {
    "train": set(),
    "validation": set(),
    "test": set()
}

with open(filename, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        split = row["split"]
        gene = row["gene_symbol"]

        if gene:
            genes[split].add(gene)

print("Unique genes:")
print("Train:", len(genes["train"]))
print("Validation:", len(genes["validation"]))
print("Test:", len(genes["test"]))

print("\nGene overlap:")
print("Train-validation:", len(genes["train"] & genes["validation"]))
print("Train-test:", len(genes["train"] & genes["test"]))
print("Validation-test:", len(genes["validation"] & genes["test"]))
