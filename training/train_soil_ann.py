from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Dataset" / "solit_dataset" / "Soil-Climate-data.csv"
MODELS_DIR = ROOT / "Models"
MODEL_PATH = MODELS_DIR / "soil_ann_model_7class.pth"
ENCODER_PATH = MODELS_DIR / "soil_ann_label_encoder_7class.pkl"
SCALER_PATH = MODELS_DIR / "soil_ann_scaler.pkl"
METRICS_PATH = MODELS_DIR / "soil_ann_model_7class_metrics.json"

TARGET_CLASSES = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
]

class SoilANN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SoilANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def preprocess_data():
    df = pd.read_csv(CSV_PATH)
    features = ["Soil_pH", "Soil_Nitrogen", "Soil_Organic_Matter", "Temperature", "Rainfall", "Humidity"]
    X = df[features].values
    
    mapping = {
        "Alluvial soils": "Alluvial_Soil",
        "Black soils": "Black_Soil",
        "Laterite soils": "Laterite_Soil",
        "Red and Yellow soils": "Red_Soil",
    }
    y_raw = df["Soil_Type"].map(mapping).fillna("Unknown")
    
    encoder = LabelEncoder()
    encoder.fit(TARGET_CLASSES)
    y = encoder.transform(y_raw)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, encoder

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, scaler, encoder = preprocess_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SoilANN(input_size=6, num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print(f"Training Soil ANN on {len(X_train)} samples...")

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs)
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Val Acc: {val_acc:.4f}")

    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "framework": "PyTorch (ANN)",
        "input_features": 6,
        "classes": TARGET_CLASSES,
        "final_accuracy": best_acc,
        "model_path": str(MODEL_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
