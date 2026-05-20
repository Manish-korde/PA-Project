import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Dataset" / "Crop_recommendation.csv"
MODELS_DIR = ROOT / "Models" / "final models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = list(le.classes_)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. Random Forest (The Accuracy King) ---
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # RF handles raw numbers well
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# --- 2. ANN (The AI Specialist) ---
print("Training ANN...")
class CropANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropANN(7, len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_t = torch.FloatTensor(X_train_scaled).to(device)
y_t = torch.LongTensor(y_train).to(device)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    test_outputs = model(torch.FloatTensor(X_test_scaled).to(device))
    ann_acc = accuracy_score(y_test, test_outputs.argmax(1).cpu().numpy())
print(f"ANN Accuracy: {ann_acc:.4f}")

# Save Everything
joblib.dump(rf_model, MODELS_DIR / "crop_rf_model.pkl")
torch.save(model.state_dict(), MODELS_DIR / "crop_ann_model.pth")
joblib.dump(scaler, MODELS_DIR / "crop_scaler.pkl")
joblib.dump(le, MODELS_DIR / "crop_label_encoder.pkl")

# Metrics for UI
metrics = {
    "rf_accuracy": rf_acc,
    "ann_accuracy": ann_acc,
    "best_model": "RandomForest" if rf_acc > ann_acc else "ANN",
    "features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    "crops": class_names
}
with open(MODELS_DIR / "crop_pro_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nDone! Pro models saved to 'Models/final models/'")
