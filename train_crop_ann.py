from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Dataset" / "solit_dataset" / "Soil-Climate-data.csv"
MODELS_DIR = ROOT / "Models"
MODEL_PATH = MODELS_DIR / "crop_ann_multiclass_model.pth"
LABEL_ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "crop_ann_preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "crop_ann_multiclass_metrics.json"
COMPARISON_PATH = MODELS_DIR / "crop_recommendation_model_comparison.json"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["Soil_Type", "Irrigation_Available"]
NUMERIC_FEATURES = [
    "Farm_Size_Acres",
    "Soil_pH",
    "Soil_Nitrogen",
    "Soil_Organic_Matter",
    "Temperature",
    "Rainfall",
    "Humidity",
]
TARGET_COLUMN = "Crop_Type"


class CropANN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset() -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    df = pd.read_csv(CSV_PATH)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    return df, y.tolist(), label_encoder


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )


def split_dataset(df: pd.DataFrame, y: list[int]):
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_prob.extend(probabilities.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, y_true, y_pred, y_prob


def update_comparison(metrics: dict[str, object], class_names: list[str], train_size: int, val_size: int, test_size: int) -> None:
    if COMPARISON_PATH.exists():
        try:
            payload = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    models = [item for item in payload.get("models", []) if item.get("Model") != metrics["model_name"]]
    models.append(
        {
            "Model": metrics["model_name"],
            "Best Validation Accuracy": metrics["best_val_accuracy"],
            "Test Accuracy": metrics["test_accuracy"],
            "Top-3 Accuracy": metrics["top3_accuracy"],
            "Test Log Loss": metrics["test_log_loss"],
        }
    )
    models.sort(key=lambda item: item["Test Accuracy"], reverse=True)

    payload.update(
        {
            "task": "crop recommendation",
            "crop_type_count": len(class_names),
            "crop_types": class_names,
            "train_size": train_size,
            "validation_size": val_size,
            "test_size": test_size,
            "models": models,
            "best_model": models[0]["Model"] if models else None,
        }
    )
    COMPARISON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, y, label_encoder = load_dataset()
    class_names = list(label_encoder.classes_)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, y)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_val_processed = X_val_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    train_ds = TensorDataset(
        torch.tensor(X_train_processed, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_processed, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_processed, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = CropANN(input_size=X_train_processed.shape[1], num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

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
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_true, val_pred, _ = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_true, val_pred)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_true, test_pred, test_prob = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_true, test_pred)
    test_log_loss = log_loss(test_true, test_prob)
    top3_acc = top_k_accuracy_score(test_true, test_prob, k=3, labels=list(range(len(class_names))))

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metrics = {
        "model_name": "ANN",
        "task": "crop recommendation",
        "framework": "PyTorch (ANN)",
        "crop_type_count": len(class_names),
        "crop_types": class_names,
        "features": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "excluded_columns": ["Compatible"],
        "split_policy": {
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "split_type": "stratified",
            "random_state": RANDOM_STATE,
        },
        "counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "top3_accuracy": top3_acc,
        "test_log_loss": test_log_loss,
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "label_encoder_path": str(LABEL_ENCODER_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_comparison(metrics, class_names, len(X_train), len(X_val), len(X_test))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
