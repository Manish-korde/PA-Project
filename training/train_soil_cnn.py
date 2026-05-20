from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Dataset" / "soil_image_datset" / "Soil-image-dataset"
ORIGINAL_DATA_DIR = DATA_ROOT / "Orignal-Dataset"
MODELS_DIR = ROOT / "Models"
MODEL_PATH = MODELS_DIR / "soil_cnn_updated_model.pth"
ENCODER_PATH = MODELS_DIR / "soil_label_encoder_7class.pkl"
METRICS_PATH = MODELS_DIR / "soil_cnn_updated_model_metrics.json"
RANDOM_STATE = 42
TARGET_TRAIN_SAMPLES = 8000


class SoilCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class SoilImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform: transforms.Compose):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = default_loader(image_path)
        return self.transform(image), label


class AugmentedOversampledDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform: transforms.Compose, target_size: int):
        if not samples:
            raise ValueError("Training samples cannot be empty.")
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return self.target_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index % len(self.samples)]
        image = default_loader(image_path)
        return self.transform(image), label


def detect_runtime() -> dict[str, object]:
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    return {
        "framework": "PyTorch",
        "pytorch_version": torch.__version__,
        "gpu_available": gpu_available,
        "device": device_name,
        "gpu_count": torch.cuda.device_count() if gpu_available else 0,
    }


def collect_class_names(root_dir: Path) -> list[str]:
    return sorted(path.name for path in root_dir.iterdir() if path.is_dir())


def collect_samples(root_dir: Path, class_names: list[str]) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            samples.append((str(image_path), class_index))
    return samples


def summarize_split(split_name: str, samples: list[tuple[str, int]], class_names: list[str]) -> None:
    counts = Counter(label for _, label in samples)
    print(f"\n{split_name} split: {len(samples)} images")
    for class_index, class_name in enumerate(class_names):
        print(f"  {class_name}: {counts.get(class_index, 0)}")


def build_datasets() -> tuple[Dataset, SoilImageDataset, SoilImageDataset, list[str], int]:
    class_names = collect_class_names(ORIGINAL_DATA_DIR)
    all_samples = collect_samples(ORIGINAL_DATA_DIR, class_names)
    labels = [label for _, label in all_samples]

    train_samples, temp_samples = train_test_split(
        all_samples,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    temp_labels = [label for _, label in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    summarize_split("Train source", train_samples, class_names)
    summarize_split("Validation", val_samples, class_names)
    summarize_split("Test", test_samples, class_names)
    print(f"\nTrain augmented target: {TARGET_TRAIN_SAMPLES} samples per epoch")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return (
        AugmentedOversampledDataset(train_samples, train_transform, TARGET_TRAIN_SAMPLES),
        SoilImageDataset(val_samples, eval_transform),
        SoilImageDataset(test_samples, eval_transform),
        class_names,
        len(train_samples),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    runtime = detect_runtime()
    print("Runtime Info:")
    print(json.dumps(runtime, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, class_names, train_source_count = build_datasets()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    joblib.dump(label_encoder, ENCODER_PATH)

    model = SoilCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest model test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}")

    metrics = {
        "runtime": runtime,
        "class_names": class_names,
        "split_policy": {
            "source": str(ORIGINAL_DATA_DIR),
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "split_type": "stratified",
            "random_state": RANDOM_STATE,
            "augmentation": "train only",
        },
        "counts": {
            "train": len(train_dataset),
            "train_source_images": train_source_count,
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "epochs_ran": args.epochs,
        "best_epoch": best_epoch,
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTraining complete. Metrics saved to:", METRICS_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
