
import re
from collections import defaultdict, Counter

BED_FILE = "splicevardb.bed"
REFGENE_FILE = "refGene.txt"
REFERENCE_FILE = "reference_bases.fa"

# Store transcript annotation mappings.
annotations = defaultdict(list)

with open(REFGENE_FILE) as f:
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

# Read reference bases in the same order as the BED records.
reference_bases = []

with open(REFERENCE_FILE) as f:
    for line in f:
        if line.startswith(">"):
            continue

        reference_bases.append(line.strip().upper())

counts = Counter()
total = 0

with open(BED_FILE) as f:
    for line in f:
        fields = line.rstrip().split("\t")

        if len(fields) < 5:
            continue

        chromosome = fields[0]
        start = int(fields[1])
        hgvs = fields[3]
        score = int(fields[4])

        total += 1

        # Each BED interval should have exactly one reference base.
        if total > len(reference_bases):
            counts["missing_reference_sequence"] += 1
            continue

        genomic_ref = reference_bases[total - 1]

        # Extract the transcript-level SNV alleles.
        match = re.search(r"([ACGT])>([ACGT])$", hgvs)

        if not match:
            counts["unsupported_variant_format"] += 1
            continue

        transcript_ref = match.group(1)

        # Get transcript accession, including removal of version suffix.
        accession_match = re.match(r"([^:]+):c\.", hgvs)

        if not accession_match:
            counts["unsupported_variant_format"] += 1
            continue

        accession = accession_match.group(1).split(".")[0]

        # Find annotation records overlapping the genomic position.
        candidates = []

        for strand, tx_start, tx_end in annotations.get(
            (accession, chromosome), []
        ):
            if tx_start <= start < tx_end:
                candidates.append(strand)

        candidates = set(candidates)

        if not candidates:
            counts["no_matching_transcript_mapping"] += 1
            continue

        if len(candidates) != 1:
            counts["ambiguous_strand_mapping"] += 1
            continue

        strand = next(iter(candidates))

        # Complement the transcript allele on the minus strand.
        if strand == "-":
            transcript_ref = transcript_ref.translate(
                str.maketrans("ACGT", "TGCA")
            )

        if transcript_ref == genomic_ref:
            counts["reference_match"] += 1
        else:
            counts["reference_mismatch"] += 1

        counts["score_" + str(score)] += 1

print("Total BED records:", total)
print("\nValidation counts:")
for label, count in sorted(counts.items()):
    print(label + ":", count)

print("\nReference bases:", len(reference_bases))

