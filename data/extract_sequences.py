
import csv
import subprocess

INPUT_FILE = "cleaned_variants.csv"
REFERENCE = "hg38.fa"
OUTPUT_FILE = "reference_windows.csv"
WINDOW = 501
FLANK = WINDOW // 2

with open(INPUT_FILE) as f:
    variants = list(csv.DictReader(f))

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "chrom", "start", "end", "hgvs",
            "score", "label", "reference_sequence"
        ]
    )
    writer.writeheader()

    written = 0
    skipped = 0

    for variant in variants:
        chrom = variant["chrom"]
        start = int(variant["start"])

        # BED start is zero-based. samtools faidx is one-based.
        region_start = start + 1 - FLANK
        region_end = start + 1 + FLANK

        if region_start < 1:
            skipped += 1
            continue

        region = f"{chrom}:{region_start}-{region_end}"

        result = subprocess.run(
            ["samtools", "faidx", REFERENCE, region],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            skipped += 1
            continue

        sequence = "".join(
            line.strip()
            for line in result.stdout.splitlines()
            if not line.startswith(">")
        ).upper()

        if len(sequence) != WINDOW:
            skipped += 1
            continue

        writer.writerow({
            "chrom": chrom,
            "start": variant["start"],
            "end": variant["end"],
            "hgvs": variant["hgvs"],
            "score": variant["score"],
            "label": variant["label"],
            "reference_sequence": sequence
        })

        written += 1

print("Input variants:", len(variants))
print("Windows written:", written)
print("Skipped:", skipped)
print("Output:", OUTPUT_FILE)
print("Expected window length:", WINDOW)

