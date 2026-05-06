import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("""# Model Visualizations and Comparisons
This notebook visualizes the performance of various models trained in the system:
1. **Tabular Models (Crop Recommendation)**: ANN vs Random Forest (RF)
2. **Image Models (Soil Classification)**: CNN vs ResNet

Models are loaded from pre-trained `.pth` and `.pkl` files without retraining."""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import json
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')"""))

# Load Crop Data
cells.append(nbf.v4.new_markdown_cell("""## 1. Tabular Models (Crop Recommendation)"""))
cells.append(nbf.v4.new_code_cell("""# Load crop data
crop_df = pd.read_csv('../Dataset/Crop_recommendation.csv')
X_crop = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = crop_df['label']

# Load Scaler and Label Encoder
crop_scaler = joblib.load('../Models/final models/crop_scaler.pkl')
crop_le = joblib.load('../Models/final models/crop_label_encoder.pkl')

y_crop_encoded = crop_le.transform(y_crop)
class_names_crop = list(crop_le.classes_)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_crop, y_crop_encoded, test_size=0.2, random_state=42
)
X_test_scaled = crop_scaler.transform(X_test_c)"""))

# Load ANN and RF
cells.append(nbf.v4.new_code_cell("""# Load Random Forest
rf_model = joblib.load('../Models/final models/crop_rf_model.pkl')
rf_preds = rf_model.predict(X_test_c) # RF was trained on raw features
rf_acc = accuracy_score(y_test_c, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Define ANN 
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

dnn_model = CropANN(7, len(class_names_crop))
dnn_model.load_state_dict(torch.load('../Models/final models/crop_ann_model.pth', map_location=device))
dnn_model.to(device)
dnn_model.eval()

with torch.no_grad():
    X_t = torch.FloatTensor(X_test_scaled).to(device)
    dnn_outputs = dnn_model(X_t)
    _, dnn_preds = torch.max(dnn_outputs, 1)
    dnn_preds = dnn_preds.cpu().numpy()

dnn_acc = accuracy_score(y_test_c, dnn_preds)
print(f"ANN Accuracy: {dnn_acc:.4f}")"""))

# Visualize Tabular
cells.append(nbf.v4.new_code_cell("""# Confusion Matrices for Crop Models
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(confusion_matrix(y_test_c, rf_preds), annot=False, cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(confusion_matrix(y_test_c, dnn_preds), annot=False, cmap='Greens', ax=axes[1])
axes[1].set_title('ANN Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()"""))

# Load Image Data
cells.append(nbf.v4.new_markdown_cell("""## 2. Image Models (Soil Classification)"""))
cells.append(nbf.v4.new_code_cell("""# Setup Image Data
class SoilImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = default_loader(image_path)
        return self.transform(image), label

def collect_class_names(root_dir):
    return sorted(path.name for path in os.scandir(root_dir) if path.is_dir())

def collect_samples(root_dir, class_names):
    samples = []
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        for image_name in os.listdir(class_dir):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                samples.append((os.path.join(class_dir, image_name), class_index))
    return samples

data_dir = '../Dataset/soil_image_datset/Soil-image-dataset/Orignal-Dataset'
class_names_img = collect_class_names(data_dir)
all_samples = collect_samples(data_dir, class_names_img)
labels = [label for _, label in all_samples]

_, temp_samples = train_test_split(all_samples, test_size=0.30, random_state=42, stratify=labels)
temp_labels = [label for _, label in temp_samples]
_, test_samples = train_test_split(temp_samples, test_size=0.50, random_state=42, stratify=temp_labels)

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = SoilImageDataset(test_samples, eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test Set Size: {len(test_dataset)}")
print(f"Classes: {class_names_img}")"""))

# Load CNN and ResNet
cells.append(nbf.v4.new_code_cell("""# Define Custom CNN
class SoilCNN(nn.Module):
    def __init__(self, num_classes):
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

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load CNN
cnn_model = SoilCNN(len(class_names_img))
try:
    cnn_model.load_state_dict(torch.load('../Models/final models/soil_cnn_updated_model.pth', map_location=device))
    cnn_model.to(device)
    cnn_model.eval()
    print("CNN Model loaded successfully.")
except Exception as e:
    print("Could not load CNN:", e)

# Load ResNet
resnet_model = models.resnet18(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(class_names_img))
try:
    # Trying models folder
    res_path = '../Models/soil_resnet_updated_model.pth'
    if not os.path.exists(res_path):
        res_path = '../Models/final models/soil_resnet_updated_model.pth'
    resnet_model.load_state_dict(torch.load(res_path, map_location=device))
    resnet_model.to(device)
    resnet_model.eval()
    print("ResNet Model loaded successfully.")
except Exception as e:
    print("Could not load ResNet (ensure it is trained and saved as 'soil_resnet_updated_model.pth'):", e)"""))

# Evaluate and visualize Image models
cells.append(nbf.v4.new_code_cell("""# Evaluate models
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

print("Evaluating CNN...")
y_true_img, cnn_preds = get_predictions(cnn_model, test_loader)
print("Evaluating ResNet...")
try:
    _, resnet_preds = get_predictions(resnet_model, test_loader)
except Exception as e:
    print("Skipping ResNet eval due to missing weights.")
    resnet_preds = None

cnn_acc = accuracy_score(y_true_img, cnn_preds)
res_acc = accuracy_score(y_true_img, resnet_preds) if resnet_preds is not None else 0
print(f"CNN Accuracy: {cnn_acc:.4f}")
if resnet_preds is not None:
    print(f"ResNet Accuracy: {res_acc:.4f}")"""))

# Confusion matrices for image models
cells.append(nbf.v4.new_code_cell("""# Visualizations
fig, axes = plt.subplots(1, 2 if resnet_preds is not None else 1, figsize=(18 if resnet_preds is not None else 8, 8))

if resnet_preds is not None:
    sns.heatmap(confusion_matrix(y_true_img, cnn_preds), annot=True, fmt='d', cmap='Oranges', ax=axes[0],
                xticklabels=class_names_img, yticklabels=class_names_img)
    axes[0].set_title('Custom CNN Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(confusion_matrix(y_true_img, resnet_preds), annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=class_names_img, yticklabels=class_names_img)
    axes[1].set_title('ResNet Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
else:
    sns.heatmap(confusion_matrix(y_true_img, cnn_preds), annot=True, fmt='d', cmap='Oranges', ax=axes,
                xticklabels=class_names_img, yticklabels=class_names_img)
    axes.set_title('Custom CNN Confusion Matrix')
    axes.set_xlabel('Predicted')
    axes.set_ylabel('True')

plt.tight_layout()
plt.show()"""))

# Model Comparisons
cells.append(nbf.v4.new_markdown_cell("""## 3. Accuracy Comparisons"""))
cells.append(nbf.v4.new_code_cell("""# Bar chart comparison
models_list = ['Random Forest', 'ANN (DNN)', 'Custom CNN', 'ResNet']
acc_list = [rf_acc, dnn_acc, cnn_acc, res_acc if resnet_preds is not None else 0]

plt.figure(figsize=(10, 6))
sns.barplot(x=models_list, y=acc_list, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
for i, v in enumerate(acc_list):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
plt.show()"""))

nb['cells'] = cells
os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/model_visualizations.ipynb')
print("Notebook created at notebooks/model_visualizations.ipynb")
