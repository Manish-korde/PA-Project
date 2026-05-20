from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE, RandomOverSampler

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
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_and_filter_dataset() -> tuple[pd.DataFrame, LabelEncoder]:
    df = pd.read_csv(CSV_PATH)
    # Filter for compatible pairings only (The Ground Truth signal)
    df_clean = df[df['Compatible'] == 1].copy()
    print(f"Original rows: {len(df)} | Cleaned (Compatible=1) rows: {len(df_clean)}")
    
    label_encoder = LabelEncoder()
    df_clean[TARGET_COLUMN] = label_encoder.fit_transform(df_clean[TARGET_COLUMN])
    return df_clean, label_encoder

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
        ]
    )

def to_numpy(x):
    if hasattr(x, "values"):
        return x.values
    return np.array(x)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_prob.extend(probabilities.cpu().tolist())
    return running_loss / len(loader.dataset), y_true, y_pred, y_prob

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, label_encoder = load_and_filter_dataset()
    class_names = list(label_encoder.classes_)
    
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    # Preprocessing
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"): X_processed = X_processed.toarray()

    # 1. SPLIT FIRST (Keep test/val sets pure)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train)

    # 2. OVERSAMPLE ONLY THE TRAINING SET
    print(f"Original Training size: {len(X_train)}")
    print("Applying SMOTE + Random Oversampling to TRAINING signal only...")
    
    # Target 400 samples per class in training (19 classes * 400 = 7600 rows)
    sampling_strategy = {i: 400 for i in range(len(class_names))}
    oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    
    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
    print(f"Resampled Training size: {len(X_train_res)}")

    # 3. Create Tensors
    train_ds = TensorDataset(torch.tensor(to_numpy(X_train_res), dtype=torch.float32), torch.tensor(to_numpy(y_train_res), dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(to_numpy(X_val), dtype=torch.float32), torch.tensor(to_numpy(y_val), dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(to_numpy(X_test), dtype=torch.float32), torch.tensor(to_numpy(y_test), dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = CropANN(input_size=X_train.shape[1], num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs)
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()

        _, val_true, val_pred, _ = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_true, val_pred)
        scheduler.step(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH))
    test_loss, test_true, test_pred, test_prob = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_true, test_pred)
    
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metrics = {
        "model_name": "ANN_FINAL",
        "test_accuracy": test_acc,
        "val_accuracy": best_val_acc,
        "samples_trained": len(X_train_res),
        "improvements": ["Filtering (Compatible=1)", "SMOTE Oversampling", "Batch Normalization"]
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
