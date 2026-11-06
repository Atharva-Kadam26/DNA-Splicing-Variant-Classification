import csv
import re
from collections import Counter, defaultdict

# Load reference transcript mappings.
annotations = defaultdict(list)

with open("refGene.txt") as f:
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

counts = Counter()
examples = defaultdict(list)

with open("cleaned_variants.csv") as f:
    for row in csv.DictReader(f):
        chrom = row["chrom"]
        start = int(row["start"])
        hgvs = row["hgvs"]

        match = re.match(
            r"([^:]+):c\..*?([ACGT])>([ACGT])$",
            hgvs
        )

        if not match:
            counts["unsupported_HGVS"] += 1
            continue

        accession = match.group(1).split(".")[0]
        transcript_ref = match.group(2)
        transcript_alt = match.group(3)

        candidates = set()

        for strand, tx_start, tx_end in annotations.get(
            (accession, chrom), []
        ):
            if tx_start <= start < tx_end:
                candidates.add(strand)

        if len(candidates) == 0:
            counts["no_mapping"] += 1
            continue

        if len(candidates) > 1:
            counts["ambiguous_strand"] += 1
            continue

        strand = next(iter(candidates))

        if strand == "-":
            complement = str.maketrans("ACGT", "TGCA")
            genomic_ref = transcript_ref.translate(complement)
            genomic_alt = transcript_alt.translate(complement)
        else:
            genomic_ref = transcript_ref
            genomic_alt = transcript_alt

        # Validate against the actual genomic reference.
        import subprocess

        result = subprocess.run(
            [
                "samtools", "faidx", "hg38.fa",
                f"{chrom}:{start + 1}-{start + 1}"
            ],
            capture_output=True,
            text=True
        )

        sequence = [
            line.strip()
            for line in result.stdout.splitlines()
            if not line.startswith(">")
        ]

        observed = "".join(sequence).upper()

        if not observed:
            counts["reference_lookup_failed"] += 1
        elif observed != genomic_ref:
            counts["reference_mismatch"] += 1
            if len(examples["reference_mismatch"]) < 10:
                examples["reference_mismatch"].append(
                    (chrom, start, hgvs, strand,
                     genomic_ref, observed)
                )
        else:
            counts["reference_match"] += 1

        if genomic_ref == genomic_alt:
            counts["invalid_same_allele"] += 1

print("Allele validation summary:")
for label, count in sorted(counts.items()):
    print(f"{label}: {count}")

for label, records in examples.items():
    print(f"\nExamples of {label}:")
    for record in records:
        print(record)
