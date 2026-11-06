
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

def encode_sequence(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    encoded = torch.zeros(4, len(sequence))

    for position, base in enumerate(sequence):
        if base in mapping:
            encoded[mapping[base], position] = 1.0

    return encoded


class SplicingDataset(Dataset):
    def __init__(self, filename, split):
        self.data = []

        with open(filename, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] == split:
                    self.data.append(row)

        print(f"{split.capitalize()} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        reference = encode_sequence(row["reference_sequence"])
        alternate = encode_sequence(row["alternate_sequence"])

        # Eight channels: four reference + four alternate.
        sequence = torch.cat((reference, alternate), dim=0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return sequence, label


# --------------------------------------------------
# 2. Improved CNN architecture
# Must match the architecture used during training.
# --------------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)


# --------------------------------------------------
# 3. Evaluation
# --------------------------------------------------

def main():
    project_dir = Path(__file__).resolve().parent.parent

    data_file = project_dir / "data" / "variant_sequences_with_splits.csv"
    model_file = project_dir / "models" / "cnn_improved.pth"

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    # Change to "validation" for model selection.
    split = "test"

    dataset = SplicingDataset(str(data_file), split)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

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
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    # Standard classification threshold.
    predictions = [
        1 if probability >= 0.5 else 0
        for probability in all_probabilities
    ]

    print(f"\nEvaluation split: {split}")
    print(f"Samples evaluated: {len(all_labels)}")

    print("\nClassification Metrics")
    print("----------------------")
    print(f"Accuracy:  {accuracy_score(all_labels, predictions):.4f}")
    print(
        f"Precision: {precision_score(all_labels, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(all_labels, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(all_labels, predictions, zero_division=0):.4f}"
    )

    if len(set(all_labels)) == 2:
        print(
            f"ROC-AUC:   {roc_auc_score(all_labels, all_probabilities):.4f}"
        )
        print(
            f"PR-AUC:    {average_precision_score(all_labels, all_probabilities):.4f}"
        )

    print("\nConfusion Matrix")
    print("----------------")
    print("Rows = actual classes; columns = predicted classes")
    print("Class order: 0 (NORMAL), 1 (SPLICE)")
    print(confusion_matrix(all_labels, predictions, labels=[0, 1]))

    print("\nClassification Report")
    print("---------------------")
    print(
        classification_report(
            all_labels,
            predictions,
            labels=[0, 1],
            target_names=["NORMAL", "SPLICE"],
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
