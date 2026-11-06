import re
from collections import Counter

# Load transcript strands from UCSC RefSeq annotation
strands = {}

with open("refGene.txt") as f:
    for line in f:
        fields = line.rstrip().split("\t")
        transcript = fields[1].split(".")[0]
        strand = fields[3]
        strands[transcript] = strand

# Load BED records
variants = []

with open("splicevardb.bed") as f:
    for line in f:
        fields = line.rstrip().split("\t")
        variants.append(fields)

# Load extracted reference bases, preserving record order
reference_bases = []

with open("reference_bases.fa") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith(">"):
            reference_bases.append(line.upper())

complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

counts = Counter()

for fields, genomic_base in zip(variants, reference_bases):
    annotation = fields[3]
    score = fields[4]

    counts["total"] += 1

    # Match transcript-level single-nucleotide substitutions
    match = re.search(r":c\..*?([ACGT])>([ACGT])$", annotation)

    if not match:
        counts["unsupported_variant_format"] += 1
        continue

    transcript_ref = match.group(1)
    transcript_alt = match.group(2)
    transcript = annotation.split(":")[0].split(".")[0]

    strand = strands.get(transcript)

    if strand not in ("+", "-"):
        counts["transcript_strand_missing"] += 1
        continue

    expected_genomic_ref = transcript_ref

    if strand == "-":
        expected_genomic_ref = complement[transcript_ref]

    counts["parsed_snv"] += 1

    if genomic_base == expected_genomic_ref:
        counts["reference_match"] += 1
    else:
        counts["reference_mismatch"] += 1
        counts["mismatch_score_" + score] += 1

print("Validation summary")
for key, value in sorted(counts.items()):
    print(f"{key}: {value}")
