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

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Dataset" / "solit_dataset" / "Soil-Climate-data.csv"
MODELS_DIR = ROOT / "Models"
MODEL_PATH = MODELS_DIR / "soil_ann_model_7class.pth"
ENCODER_PATH = MODELS_DIR / "soil_ann_label_encoder_7class.pkl"
SCALER_PATH = MODELS_DIR / "soil_ann_scaler.pkl"
METRICS_PATH = MODELS_DIR / "soil_ann_model_7class_metrics.json"

# The 7 target classes as defined in app.py
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
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def preprocess_data():
    df = pd.read_csv(CSV_PATH)
    
    # Feature columns
    features = ["Soil_pH", "Soil_Nitrogen", "Soil_Organic_Matter", "Temperature", "Rainfall", "Humidity"]
    X = df[features].values
    
    # Label mapping to match the 7 classes
    # Original CSV has: ['Red and Yellow soils', 'Alluvial soils', 'Laterite soils', 'Black soils']
    mapping = {
        "Alluvial soils": "Alluvial_Soil",
        "Black soils": "Black_Soil",
        "Laterite soils": "Laterite_Soil",
        "Red and Yellow soils": "Red_Soil", # Defaulting 'Red and Yellow' to 'Red'
    }
    
    y_raw = df["Soil_Type"].map(mapping).fillna("Unknown")
    
    # We want a LabelEncoder that knows about ALL 7 classes
    encoder = LabelEncoder()
    encoder.fit(TARGET_CLASSES)
    
    # Convert labels to indices
    y = encoder.transform(y_raw)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, encoder

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, scaler, encoder = preprocess_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SoilANN(input_size=6, num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training ANN on {len(X_train)} samples for {args.epochs} epochs...")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
        
        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "framework": "PyTorch (ANN)",
        "input_features": 6,
        "classes": TARGET_CLASSES,
        "final_accuracy": val_acc,
        "model_path": str(MODEL_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
