
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from evaluate_cnn import NeuralNetwork, encode_sequence


# --------------------------------------------------
# 1. Validation dataset
# --------------------------------------------------

class SplicingDataset(Dataset):
    def __init__(self, filename):
        self.data = []

        with open(filename, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] == "validation":
                    self.data.append(row)

        print(f"Validation samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        reference = encode_sequence(row["reference_sequence"])
        alternate = encode_sequence(row["alternate_sequence"])

        sequence = torch.cat((reference, alternate), dim=0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return sequence, label


# --------------------------------------------------
# 2. Evaluate different thresholds
# --------------------------------------------------

def main():
    project_dir = Path(__file__).resolve().parent.parent

    data_file = project_dir / "data" / "variant_sequences_with_splits.csv"
    model_file = project_dir / "models" / "cnn_improved.pth"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = SplicingDataset(str(data_file))
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model = NeuralNetwork().to(device)
    model.load_state_dict(
        torch.load(
            model_file,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)

            logits = model(sequences)
            probabilities = torch.sigmoid(logits)

            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\nValidation Threshold Analysis")
    print("-" * 76)
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
    )

    results = []

    for threshold in thresholds:
        predictions = [
            1 if probability >= threshold else 0
            for probability in all_probabilities
        ]

        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(
            all_labels, predictions, zero_division=0
        )
        recall = recall_score(
            all_labels, predictions, zero_division=0
        )
        f1 = f1_score(
            all_labels, predictions, zero_division=0
        )

        results.append((threshold, accuracy, precision, recall, f1))

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

    best_result = max(results, key=lambda result: result[4])

    print("\nBest threshold by validation F1-score:")
    print(f"Threshold: {best_result[0]:.2f}")
    print(f"Accuracy:  {best_result[1]:.4f}")
    print(f"Precision: {best_result[2]:.4f}")
    print(f"Recall:    {best_result[3]:.4f}")
    print(f"F1-score:  {best_result[4]:.4f}")

    output_file = project_dir / "results" / "threshold_analysis.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "threshold", "accuracy", "precision", "recall", "f1_score"
        ])
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
