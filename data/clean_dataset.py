
from collections import defaultdict, Counter
import csv

input_file = "splicevardb.bed"
output_file = "cleaned_variants.csv"

# Group all annotations by genomic position.
positions = defaultdict(list)

with open(input_file) as f:
    for line in f:
        fields = line.rstrip().split("\t")
        if len(fields) < 5:
            continue

        chrom, start, end = fields[:3]
        hgvs = fields[3]
        score = int(fields[4])

        positions[(chrom, start, end)].append({
            "hgvs": hgvs,
            "score": score
        })

rows = []
excluded_conflicts = 0
excluded_uncertain = 0

for (chrom, start, end), records in positions.items():
    scores = {record["score"] for record in records}

    # Exclude every position with contradictory scores.
    if len(scores) > 1:
        excluded_conflicts += 1
        continue

    score = next(iter(scores))

    # Keep only the two classes for the initial binary task.
    if score not in (1, 3):
        excluded_uncertain += 1
        continue

    # One row per genomic position.
    rows.append({
        "chrom": chrom,
        "start": int(start),
        "end": int(end),
        "hgvs": records[0]["hgvs"],
        "score": score,
        "label": 1 if score == 3 else 0
    })

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "chrom", "start", "end", "hgvs", "score", "label"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("Cleaned records:", len(rows))
print("Conflicting positions excluded:", excluded_conflicts)
print("Uncertain-class positions excluded:", excluded_uncertain)
print("Class counts:", Counter(row["label"] for row in rows))
print("Saved to:", output_file)

