
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. DNA encoding
# --------------------------------------------------

def encode_sequence(sequence):
    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }

    encoded = torch.zeros(len(sequence), 4, dtype=torch.float32)

    for position, base in enumerate(sequence):
        if base in mapping:
            encoded[position, mapping[base]] = 1.0

    return encoded


# --------------------------------------------------
# 2. Dataset
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

        # Concatenate the two sequences along the feature dimension.
        sequence = torch.cat((reference, alternate), dim=1)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        return sequence, label


# --------------------------------------------------
# 3. LSTM model
# --------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        # Concatenate the final forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        return self.classifier(combined).squeeze(1)


# --------------------------------------------------
# 4. Training
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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# 5. Evaluation
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

            predictions = (torch.sigmoid(logits) >= 0.5).float()

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

def main():
    torch.manual_seed(42)

    project_dir = Path(__file__).resolve().parent.parent

    data_file = (
        project_dir / "data" / "variant_sequences_with_splits.csv"
    )

    model_file = project_dir / "models" / "lstm_model.pth"

    training_data = SplicingDataset(str(data_file), "train")
    validation_data = SplicingDataset(str(data_file), "validation")

    train_loader = DataLoader(
        training_data,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    model = LSTMModel().to(device)

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

    epochs = 20
    patience = 5

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print("\nStarting LSTM training...\n")

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
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_file)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered.")
            break

    print(f"\nBest LSTM model saved to: {model_file}")


if __name__ == "__main__":
    main()
