
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

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

        # Combine reference and alternate one-hot encodings.
        x = torch.cat((reference, alternate), dim=0)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        return x, label


# --------------------------------------------------
# 2. DNA one-hot encoding
# --------------------------------------------------

def encode_sequence(sequence):
    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }

    encoded = torch.zeros(
        4, len(sequence),
        dtype=torch.float32
    )

    for position, base in enumerate(sequence):
        if base in mapping:
            encoded[mapping[base], position] = 1.0

    return encoded


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

def load_data():
    project_dir = Path(__file__).resolve().parent.parent
    filename = project_dir / "data" / "variant_sequences_with_splits.csv"

    if not filename.exists():
        raise FileNotFoundError(f"Dataset not found: {filename}")

    training_data = SplicingDataset(str(filename), "train")
    validation_data = SplicingDataset(str(filename), "validation")
    test_data = SplicingDataset(str(filename), "test")

    return training_data, validation_data, test_data


# --------------------------------------------------
# 4. Improved CNN
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
# 5. Training
# --------------------------------------------------

def train(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(sequences)
        loss = loss_function(logits, labels)

        loss.backward()

        # Prevent unusually large gradient updates.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = loss_function(logits, labels)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# --------------------------------------------------
# 7. Main training pipeline
# --------------------------------------------------

def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    training_data, validation_data, test_data = load_data()

    batch_size = 64

    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    model = NeuralNetwork().to(device)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    epochs = 30
    patience = 5
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "cnn_improved.pth"

    print("\nStarting improved CNN training...\n")

    for epoch in range(epochs):
        training_loss = train(
            model,
            train_loader,
            loss_function,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            device
        )

        scheduler.step(validation_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {training_loss:.4f} "
            f"Validation Loss: {validation_loss:.4f} "
            f"Validation Accuracy: {validation_accuracy:.4f} "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered.")
            break

    print(f"\nBest model saved to: {model_path}")

    # Evaluate the held-out test set only after model selection.
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device
    )

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
