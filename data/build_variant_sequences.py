
import csv
import re
import subprocess
from collections import defaultdict

ANNOTATION_FILE = "refGene.txt"
INPUT_FILE = "reference_windows.csv"
OUTPUT_FILE = "variant_sequences.csv"

# Load transcript mappings, retaining all records.
annotations = defaultdict(list)

with open(ANNOTATION_FILE) as f:
    for line in f:
        fields = line.rstrip().split("\t")
        if len(fields) < 13:
            continue

        accession = fields[1].split(".")[0]
        chromosome = fields[2]
        strand = fields[3]
        tx_start = int(fields[4])
        tx_end = int(fields[5])

        annotations[(accession, chromosome)].append(
            (strand, tx_start, tx_end)
        )

complement = str.maketrans("ACGT", "TGCA")
counts = defaultdict(int)

with open(INPUT_FILE) as f, open(OUTPUT_FILE, "w", newline="") as out:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(out, fieldnames=[
        "chrom", "start", "hgvs", "label",
        "reference_sequence", "alternate_sequence",
        "gene", "strand"
    ])
    writer.writeheader()

    for row in reader:
        chrom = row["chrom"]
        start = int(row["start"])
        hgvs = row["hgvs"]
        sequence = row["reference_sequence"]

        match = re.match(
            r"([^:]+):c\..*?([ACGT])>([ACGT])$",
            hgvs
        )

        if not match:
            counts["unsupported_HGVS"] += 1
            continue

        accession = match.group(1).split(".")[0]
        transcript_ref, transcript_alt = match.group(2), match.group(3)

        candidates = {
            (strand, tx_start, tx_end, gene)
            for strand, tx_start, tx_end, gene in []
        }

        # Identify annotations overlapping the variant position.
        mappings = set()
        for strand, tx_start, tx_end in annotations.get(
            (accession, chrom), []
        ):
            if tx_start <= start < tx_end:
                mappings.add((strand, tx_start, tx_end))

        strands = {item[0] for item in mappings}

        if len(strands) != 1:
            counts["missing_or_ambiguous_mapping"] += 1
            continue

        strand = next(iter(strands))

        if strand == "-":
            genomic_ref = transcript_ref.translate(complement)
            genomic_alt = transcript_alt.translate(complement)
        else:
            genomic_ref = transcript_ref
            genomic_alt = transcript_alt

        # The reference sequence's center must match the genomic allele.
        if sequence[250] != genomic_ref:
            counts["reference_allele_mismatch"] += 1
            continue

        alternate = (
            sequence[:250] + genomic_alt + sequence[251:]
        )

        writer.writerow({
            "chrom": chrom,
            "start": start,
            "hgvs": hgvs,
            "label": row["label"],
            "reference_sequence": sequence,
            "alternate_sequence": alternate,
            "gene": accession,
            "strand": strand
        })

        counts["written"] += 1

print("Results:")
for key, value in sorted(counts.items()):
    print(key + ":", value)
print("Output:", OUTPUT_FILE)

