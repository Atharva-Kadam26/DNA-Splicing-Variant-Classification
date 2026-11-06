import re

complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

strands = {}
with open("refGene.txt") as f:
    for line in f:
        fields = line.rstrip().split("\t")
        transcript = fields[1].split(".")[0]
        strands[transcript] = fields[3]

with open("splicevardb.bed") as bed, open("reference_bases.fa") as fasta:
    records = [line.rstrip().split("\t") for line in bed]
    bases = [line.strip().upper() for line in fasta if not line.startswith(">")]

shown = 0
for fields, genomic_ref in zip(records, bases):
    annotation = fields[3]
    match = re.search(r":c\..*?([ACGT])>([ACGT])$", annotation)
    if not match:
        continue

    transcript = annotation.split(":")[0].split(".")[0]
    strand = strands.get(transcript)
    if strand not in ("+", "-"):
        continue

    expected = match.group(1)
    if strand == "-":
        expected = complement[expected]

    if genomic_ref != expected:
        print(
            fields[0], fields[1], fields[2],
            annotation, "score=" + fields[4],
            "expected=" + expected, "observed=" + genomic_ref,
            "strand=" + strand,
            sep="\t"
        )
        shown += 1
        if shown == 20:
            break
