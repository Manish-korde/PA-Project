from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from PIL import Image

from agent import FarmAgent
from analysis.image_analysis import ImageAnalysisService
from analysis.soil_analysis import SoilAnalysisService

# Global Agent Instance
# Soil Profiles Database (Scientific Baselines for Farmers)
SOIL_PROFILES = {
    "Alluvial_Soil": {
        "description": "Rich in minerals, particularly potash. Highly fertile and best for grains.",
        "best_crops": "Rice, Wheat, Sugarcane, Pulses, Oilseeds",
        "weather": "Warm with moderate to high rainfall.",
        "baselines": {"soil_nitrogen": 80, "soil_phosphorus": 40, "soil_potassium": 40, "soil_ph": 7.0, "temperature": 25, "rainfall": 1200, "humidity": 70, "soil_organic_matter": 1.5}
    },
    "Arid_Soil": {
        "description": "Sandy texture, low organic matter, high salt content. Requires irrigation.",
        "best_crops": "Bajra, Guar, Fodder, Melons",
        "weather": "Hot and dry with very low rainfall.",
        "baselines": {"soil_nitrogen": 20, "soil_phosphorus": 10, "soil_potassium": 30, "soil_ph": 8.2, "temperature": 35, "rainfall": 300, "humidity": 20, "soil_organic_matter": 0.2}
    },
    "Black_Soil": {
        "description": "High clay content, excellent water retention. Rich in iron, lime, and calcium.",
        "best_crops": "Cotton, Soyabean, Chillies, Jowar",
        "weather": "Tropical with moderate rainfall.",
        "baselines": {"soil_nitrogen": 60, "soil_phosphorus": 50, "soil_potassium": 80, "soil_ph": 7.8, "temperature": 28, "rainfall": 800, "humidity": 50, "soil_organic_matter": 1.0}
    },
    "Laterite_Soil": {
        "description": "Leached soil, acidic, poor in fertility. Good for plantation crops.",
        "best_crops": "Cashew, Rubber, Tea, Coffee",
        "weather": "Heavy rainfall and high humidity.",
        "baselines": {"soil_nitrogen": 30, "soil_phosphorus": 20, "soil_potassium": 20, "soil_ph": 5.2, "temperature": 26, "rainfall": 2000, "humidity": 80, "soil_organic_matter": 2.5}
    },
    "Mountain_Soil": {
        "description": "Rich in organic matter (humus) but acidic. Found in hilly regions.",
        "best_crops": "Tea, Coffee, Spices, Apple, Saffron",
        "weather": "Cooler temperatures with high moisture.",
        "baselines": {"soil_nitrogen": 90, "soil_phosphorus": 30, "soil_potassium": 40, "soil_ph": 5.8, "temperature": 18, "rainfall": 1500, "humidity": 60, "soil_organic_matter": 4.0}
    },
    "Red_Soil": {
        "description": "Formed from crystalline rocks. Rich in iron but low in nitrogen and phosphorus.",
        "best_crops": "Tobacco, Groundnut, Ragi, Potato",
        "weather": "Warm with seasonal rainfall.",
        "baselines": {"soil_nitrogen": 40, "soil_phosphorus": 20, "soil_potassium": 30, "soil_ph": 6.0, "temperature": 27, "rainfall": 1000, "humidity": 40, "soil_organic_matter": 0.8}
    },
    "Yellow_Soil": {
        "description": "Hydrated form of red soil. Slightly more fertile than red soil.",
        "best_crops": "Rice, Maize, Groundnut",
        "weather": "Moderate warmth and rainfall.",
        "baselines": {"soil_nitrogen": 50, "soil_phosphorus": 30, "soil_potassium": 40, "soil_ph": 6.4, "temperature": 26, "rainfall": 1100, "humidity": 50, "soil_organic_matter": 1.2}
    }
}


FARM_AGENT = FarmAgent()



ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
MODELS_DIR = ROOT / "Models"
FINAL_MODELS_DIR = MODELS_DIR / "final models"
DATASET_DIR = ROOT / "Dataset"
SOIL_ANALYSIS = SoilAnalysisService(DATASET_DIR / "solit_dataset" / "Soil-Climate-data.csv")
IMAGE_ANALYSIS = ImageAnalysisService(
    DATASET_DIR / "soil_image_datset", 
    FINAL_MODELS_DIR / "soil_cnn_updated_model_metrics.json"
)
def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]

SOIL_MODEL_PATH = None
SOIL_ENCODER_PATH = first_existing_path(
    FINAL_MODELS_DIR / "soil_label_encoder_7class.pkl",
    MODELS_DIR / "soil_label_encoder_7class.pkl"
)

SOIL_ANN_MODEL_PATH = MODELS_DIR / "soil_ann_model_7class.pth"
SOIL_ANN_ENCODER_PATH = first_existing_path(
    FINAL_MODELS_DIR / "soil_ann_label_encoder_7class.pkl",
    MODELS_DIR / "soil_ann_label_encoder_7class.pkl"
)
SOIL_ANN_SCALER_PATH = MODELS_DIR / "soil_ann_scaler.pkl"
SOIL_ANN_METRICS_PATH = MODELS_DIR / "soil_ann_model_7class_metrics.json"

CROP_PRO_METRICS_PATH = FINAL_MODELS_DIR / "crop_pro_metrics.json"
CROP_ANN_MODEL_PATH = FINAL_MODELS_DIR / "crop_ann_model.pth"
CROP_RF_MODEL_PATH = FINAL_MODELS_DIR / "crop_rf_model.pkl"
CROP_LABEL_ENCODER_PATH = FINAL_MODELS_DIR / "crop_label_encoder.pkl"
CROP_SCALER_PATH = FINAL_MODELS_DIR / "crop_scaler.pkl"

SOIL_DATASET_PATH = DATASET_DIR / "soil_image_datset"
TARGET_SOIL_CLASSES = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil",
]





SOIL_MODEL_PATH = first_existing_path(
    FINAL_MODELS_DIR / "soil_cnn_updated_model.pth",
    MODELS_DIR / "soil_cnn_updated_model.pth",
    MODELS_DIR / "soil_cnn_model_7class.pth",
)
SOIL_MODEL_METRICS_PATH = first_existing_path(
    FINAL_MODELS_DIR / "soil_cnn_updated_model_metrics.json",
    MODELS_DIR / "soil_cnn_updated_model_metrics.json",
    MODELS_DIR / "soil_cnn_model_7class_metrics.json",
)

try:
    SOIL_ENCODER = joblib.load(SOIL_ENCODER_PATH)
    SOIL_ENCODER_CLASSES = list(getattr(SOIL_ENCODER, "classes_", []))
except Exception:
    SOIL_ENCODER = None
    SOIL_ENCODER_CLASSES = []

try:
    SOIL_ANN_ENCODER = joblib.load(SOIL_ANN_ENCODER_PATH)
    SOIL_ANN_SCALER = joblib.load(SOIL_ANN_SCALER_PATH)
    SOIL_ANN_CLASSES = list(getattr(SOIL_ANN_ENCODER, "classes_", []))
except Exception:
    SOIL_ANN_ENCODER = None
    SOIL_ANN_SCALER = None
    SOIL_ANN_CLASSES = []

try:
    CROP_LABEL_ENCODER = joblib.load(CROP_LABEL_ENCODER_PATH)
    CROP_CLASSES = list(getattr(CROP_LABEL_ENCODER, "classes_", []))
except Exception:
    CROP_LABEL_ENCODER = None
    CROP_CLASSES = []

try:
    CROP_RF_MODEL = joblib.load(CROP_RF_MODEL_PATH)
except Exception:
    CROP_RF_MODEL = None

try:
    CROP_TREE_PREPROCESSOR = joblib.load(CROP_TREE_PREPROCESSOR_PATH)
except Exception:
    CROP_TREE_PREPROCESSOR = None

try:
    CROP_ANN_PREPROCESSOR = joblib.load(CROP_ANN_PREPROCESSOR_PATH)
except Exception:
    CROP_ANN_PREPROCESSOR = None

try:
    CROP_PRO_METRICS = json.loads(CROP_PRO_METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    CROP_PRO_METRICS = {}

try:
    CROP_SCALER = joblib.load(CROP_SCALER_PATH)
except Exception:
    CROP_SCALER = None

try:
    SOIL_MODEL_METRICS = json.loads(SOIL_MODEL_METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    SOIL_MODEL_METRICS = {}

try:
    SOIL_ANN_METRICS = json.loads(SOIL_ANN_METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    SOIL_ANN_METRICS = {}

try:
    CROP_COMPARISON = json.loads(CROP_COMPARISON_PATH.read_text(encoding="utf-8"))
except Exception:
    CROP_COMPARISON = {}

SOIL_ENCODER_ERROR = ""

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    TORCH_IMPORT_ERROR = ""


class SoilCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SoilCNN, self).__init__()
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
        x = self.classifier(x)
        return x


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

    def forward(self, x):
        return self.net(x)


def normalize_label(label: str) -> str:
    lowered = label.strip().lower().replace("-", " ").replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    synonyms = {
        "alluvial soil": "Alluvial_Soil",
        "alluvial soils": "Alluvial_Soil",
        "arid soil": "Arid_Soil",
        "arid soils": "Arid_Soil",
        "black soil": "Black_Soil",
        "black soils": "Black_Soil",
        "laterite soil": "Laterite_Soil",
        "laterite soils": "Laterite_Soil",
        "mountain soil": "Mountain_Soil",
        "mountain soils": "Mountain_Soil",
        "red soil": "Red_Soil",
        "red soils": "Red_Soil",
        "yellow soil": "Yellow_Soil",
        "yellow soils": "Yellow_Soil",
        "red and yellow soil": "Red_Soil",
        "red and yellow soils": "Red_Soil",
    }
    return synonyms.get(lowered, label)


def pytorch_runtime_info() -> dict[str, Any]:
    info = {
        "available": torch is not None,
        "version": getattr(torch, "__version__", None) if torch is not None else None,
        "gpu_available": torch.cuda.is_available() if torch is not None else False,
        "device_name": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else "CPU",
        "warning": "",
    }
    if torch is None:
        info["warning"] = TORCH_IMPORT_ERROR or "PyTorch import failed."
        return info

    if not info["gpu_available"]:
        info["warning"] = "PyTorch is installed but GPU is not available. Native Windows GPU usually works if torch+cuXXX is installed."
    return info


def dataset_class_counts(base_path: Path) -> dict[str, int]:
    if not base_path.exists():
        return {}

    counts: dict[str, int] = {}
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            counts[item.name] = sum(1 for child in item.iterdir() if child.is_file())
    return counts


def inspect_soil_model_binary() -> dict[str, Any]:
    if not SOIL_MODEL_PATH.exists():
        return {"exists": False}

    output_classes = len(SOIL_ENCODER_CLASSES) if SOIL_ENCODER_CLASSES else None

    return {
        "exists": True,
        "path": str(SOIL_MODEL_PATH),
        "size_mb": round(SOIL_MODEL_PATH.stat().st_size / (1024 * 1024), 2),
        "input_shape_hint": [224, 224, 3],
        "framework": "PyTorch",
        "output_classes": output_classes,
        "metrics_path": str(SOIL_MODEL_METRICS_PATH) if SOIL_MODEL_METRICS_PATH.exists() else None,
        "best_val_accuracy": SOIL_MODEL_METRICS.get("best_val_accuracy"),
        "test_accuracy": SOIL_MODEL_METRICS.get("test_accuracy"),
    }


def recommendation_audit() -> dict[str, Any]:
    best_crop_model = CROP_PRO_METRICS.get("best_model")
    crop_ready = bool(CROP_PRO_METRICS and CROP_LABEL_ENCODER and CROP_CLASSES and CROP_SCALER)
    
    if best_crop_model == "RandomForest":
        crop_ready = crop_ready and CROP_RF_MODEL is not None
    elif best_crop_model == "ANN":
        crop_ready = crop_ready and torch is not None and CROP_ANN_MODEL_PATH.exists()
    
    legacy_ann_ready = bool(
        SOIL_ANN_MODEL_PATH.exists()
        and SOIL_ANN_ENCODER is not None
        and SOIL_ANN_SCALER is not None
    )

    warnings: list[str] = []
    active_mode = "crop_recommendation" if crop_ready else "legacy_soil_ann"

    if not crop_ready:
        warnings.append(
            "Crop recommendation models are not trained yet. The web app is using the older tabular soil ANN as a fallback."
        )
        warnings.append(
            "The legacy ANN predicts soil labels from six numeric features and is not the final crop recommendation model from the revised plan."
        )
    if legacy_ann_ready and SOIL_ANN_METRICS.get("final_accuracy") is not None:
        warnings.append(
            f"Legacy ANN recorded accuracy is {SOIL_ANN_METRICS['final_accuracy']:.3f}, so its outputs should be treated as provisional."
        )
    if crop_ready and CROP_PRO_METRICS.get("best_model"):
        warnings.append(
            f"Pro Crop recommendation artifacts detected. Preferred model: {CROP_PRO_METRICS['best_model']}."
        )

    return {
        "active_mode": active_mode,
        "crop_recommendation_ready": crop_ready,
        "legacy_ann_ready": legacy_ann_ready,
        "crop_type_count": len(CROP_PRO_METRICS.get("crops", [])),
        "best_crop_model": best_crop_model,
        "crop_models": [
            {"Model": "RandomForest", "Accuracy": CROP_PRO_METRICS.get("rf_accuracy")},
            {"Model": "ANN", "Accuracy": CROP_PRO_METRICS.get("ann_accuracy")}
        ],
        "legacy_ann_metrics": SOIL_ANN_METRICS,
        "warnings": warnings,
        "runnable": crop_ready or legacy_ann_ready,
    }


def soil_audit() -> dict[str, Any]:
    original_counts = dataset_class_counts(SOIL_DATASET_PATH / "Orignal-Dataset")
    augmented_counts = dataset_class_counts(SOIL_DATASET_PATH / "CyAUG-Dataset")
    model_file = inspect_soil_model_binary()
    normalized_encoder_classes = [normalize_label(label) for label in SOIL_ENCODER_CLASSES]
    runtime = pytorch_runtime_info()

    warnings: list[str] = []
    if original_counts and len(original_counts) != len(SOIL_ENCODER_CLASSES):
        warnings.append(
            f"Dataset folders contain {len(original_counts)} classes, but the saved label encoder contains {len(SOIL_ENCODER_CLASSES)} classes."
        )
    if normalized_encoder_classes:
        expected = set(original_counts.keys())
        encoded = set(normalized_encoder_classes)
        if encoded and encoded != expected:
            warnings.append("Label names in the encoder do not match the class folders on disk.")
    if TORCH_IMPORT_ERROR:
        warnings.append("PyTorch is not installed locally, so the CNN cannot be executed from this workspace.")
    if SOIL_ENCODER_ERROR:
        warnings.append(f"Encoder load produced a compatibility issue: {SOIL_ENCODER_ERROR}")
    if runtime["warning"]:
        warnings.append(runtime["warning"])

    return {
        "model_file": model_file,
        "target_classes": TARGET_SOIL_CLASSES,
        "encoder_classes": SOIL_ENCODER_CLASSES,
        "normalized_encoder_classes": normalized_encoder_classes,
        "original_dataset_counts": original_counts,
        "augmented_dataset_counts": augmented_counts,
        "runtime": runtime,
        "warnings": warnings,
        "runnable": bool(
            torch is not None
            and SOIL_MODEL_PATH.exists()
            and SOIL_ENCODER_CLASSES
            and len(SOIL_ENCODER_CLASSES) > 0
        ),
    }


SOIL_AUDIT = soil_audit()
RECOMMENDATION_AUDIT = recommendation_audit()


def parse_crop_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        data = {
            "N": float(payload["soil_nitrogen"]),
            "P": float(payload["soil_phosphorus"]),
            "K": float(payload["soil_potassium"]),
            "temperature": float(payload["temperature"]),
            "humidity": float(payload["humidity"]),
            "ph": float(payload["soil_ph"]),
            "rainfall": float(payload["rainfall"]),
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or invalid crop recommendation parameters: {exc}")
    return data


def predict_crop_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    if not RECOMMENDATION_AUDIT["crop_recommendation_ready"]:
        raise RuntimeError("Crop recommendation models are not ready.")

    data_dict = parse_crop_payload(payload)
    features_list = [[data_dict[f] for f in CROP_PRO_METRICS["features"]]]
    best_model = RECOMMENDATION_AUDIT["best_crop_model"]

    if best_model == "RandomForest":
        probabilities = CROP_RF_MODEL.predict_proba(features_list)[0]
        model_name = "Random Forest (Pro)"
    elif best_model == "ANN":
        transformed = CROP_SCALER.transform(features_list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CropANN(input_size=7, num_classes=len(CROP_CLASSES)).to(device)
        model.load_state_dict(torch.load(CROP_ANN_MODEL_PATH, map_location=device))
        model.eval()
        input_tensor = torch.tensor(transformed, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        model_name = "ANN (Pro)"
    else:
        raise RuntimeError(f"Unsupported crop recommendation model: {best_model}")

    ranked = sorted(
        (
            {"label": CROP_CLASSES[index], "probability": round(float(probabilities[index]), 4)}
            for index in range(len(CROP_CLASSES))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "mode": "crop_recommendation",
        "model_name": model_name,
        "label": ranked[0]["label"],
        "top_probability": ranked[0]["probability"],
        "ranked_predictions": ranked,
        "warning": (
            f"Current best crop model is {model_name}. Accuracy: "
            f"{CROP_PRO_METRICS.get('rf_accuracy', 0):.3f} (RF) / {CROP_PRO_METRICS.get('ann_accuracy', 0):.3f} (ANN)."
        ),
    }


def predict_soil(payload: dict[str, Any]) -> dict[str, Any]:
    if not SOIL_AUDIT["runnable"]:
        raise RuntimeError("Soil CNN is not ready.")

    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise ValueError("No image provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(SOIL_ENCODER_CLASSES)
    model = SoilCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(SOIL_MODEL_PATH, map_location=device))
    model.eval()

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        top_index = int(probabilities.argmax())

    ranked = sorted(
        (
            {"label": SOIL_ENCODER_CLASSES[index], "probability": round(float(probabilities[index]), 4)}
            for index in range(len(SOIL_ENCODER_CLASSES))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    # --- Visual Decision Support: Micro-Texture Enhancement ---
    # This pipeline enhances soil micro-textures, cracks, and mineral hues 
    # to help agronomists visually verify the soil condition,
    # compensating for low-quality smartphone camera captures.
    from PIL import ImageEnhance, ImageFilter
    
    # 1. Noise Reduction
    enhanced_img = image.filter(ImageFilter.SMOOTH_MORE)
    # 2. Micro-Texture Synthesis
    enhancer = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(1.5)
    # 3. Additional enhancements for visible effect
    enhancer = ImageEnhance.Brightness(enhanced_img)
    enhanced_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(enhanced_img)
    enhanced_img = enhancer.enhance(1.2)
    
    # Save enhanced image to base64
    buffered = io.BytesIO()
    enhanced_img.save(buffered, format="JPEG")
    enhanced_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    soil_type = SOIL_ENCODER_CLASSES[top_index]
    profile = SOIL_PROFILES.get(soil_type, {})

    return {
        "label": soil_type,
        "top_probability": ranked[0]["probability"],
        "ranked_predictions": ranked,
        "enhanced_image": f"data:image/jpeg;base64,{enhanced_base64}",
        "profile": profile
    }


def predict_soil_tabular(payload: dict[str, Any]) -> dict[str, Any]:
    # 1. Get Crop Recommendation (New Data)
    crop_result = {}
    if RECOMMENDATION_AUDIT["crop_recommendation_ready"]:
        crop_result = predict_crop_recommendation(payload)

    # 2. Get Soil Identification (Old Data or Direct Input)
    soil_label = payload.get("soil_type", "").strip()
    
    # Fallback to ANN if the user didn't select or scan anything
    if not soil_label or soil_label == "Manually select soil type":
        soil_label = "Unknown"
        if SOIL_ANN_MODEL_PATH.exists() and SOIL_ANN_ENCODER:
            try:
                # Features: pH, Nitrogen, Organic_Matter, Temp, Rainfall, Humidity
                data = [
                    float(payload["soil_ph"]),
                    float(payload["soil_nitrogen"]),
                    float(payload["soil_organic_matter"]),
                    float(payload["temperature"]),
                    float(payload["rainfall"]),
                    float(payload["humidity"]),
                ]
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                num_classes = len(SOIL_ANN_CLASSES)
                model = SoilANN(input_size=6, num_classes=num_classes).to(device)
                model.load_state_dict(torch.load(SOIL_ANN_MODEL_PATH, map_location=device))
                model.eval()
                
                scaled = SOIL_ANN_SCALER.transform([data])
                input_tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    idx = int(outputs.argmax(dim=1)[0])
                    soil_label = SOIL_ANN_CLASSES[idx]
            except Exception:
                soil_label = "N/A (Pending Image Scan)"
        else:
            soil_label = "N/A (Pending Image Scan)"

    # 3. Generate Agent Action Plan
    action_plan = FARM_AGENT.generate_action_plan(
        features=payload,
        crop_result=crop_result,
        soil_result=soil_label
    )

    return {
        "mode": "hybrid_recommendation",
        "crop_label": crop_result.get("label", "N/A"),
        "soil_label": soil_label,
        "crop_confidence": crop_result.get("top_probability", 0),
        "warning": crop_result.get("warning", ""),
        "action_plan": action_plan,
        "ranked_predictions": crop_result.get("ranked_predictions", [])
    }

    # Scale
    input_scaled = SOIL_ANN_SCALER.transform([data])
    input_tensor = torch.FloatTensor(input_scaled).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        top_index = int(probabilities.argmax())

    ranked = sorted(
        (
            {"label": SOIL_ANN_CLASSES[index], "probability": round(float(probabilities[index]), 4)}
            for index in range(len(SOIL_ANN_CLASSES))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "mode": "legacy_soil_ann",
        "model_name": "Legacy Soil ANN",
        "label": SOIL_ANN_CLASSES[top_index],
        "top_probability": ranked[0]["probability"],
        "ranked_predictions": ranked,
        "warning": "This tab is still using the legacy tabular soil ANN until crop recommendation models are trained and integrated.",
    }


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
        query = urllib.parse.parse_qs(parsed.query or "")

        if path == "/api/status":
            json_response(
                self,
                HTTPStatus.OK,
                {
                    "soil_audit": SOIL_AUDIT,
                    "recommendation_audit": RECOMMENDATION_AUDIT,
                    "agent": FARM_AGENT.status(),
                    "soil_analysis": {"plot_count": len(SOIL_ANALYSIS.list_plots())},
                    "server_urls": {
                        "local": "http://127.0.0.1:8000",
                        "browser_hint": "Use http://127.0.0.1:8000 or http://localhost:8000 in the browser, not http://0.0.0.0:8000.",
                    },
                },
            )
            return

        if path == "/api/augmented-samples":
            import random
            from PIL import ImageEnhance, ImageOps
            import numpy as np

            def apply_live_augmentation(image):
                # 1. Random Rotation
                img = image.rotate(random.uniform(-25, 25))
                # 2. Random Horizontal Flip
                if random.random() > 0.5:
                    img = ImageOps.mirror(img)
                # 3. Color Jittering (Brightness/Contrast)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.7, 1.3))
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
                # 4. Simulated GAN Noise / Texture Grain
                data = np.array(img).astype(np.float32)
                noise = np.random.normal(0, 5, data.shape)
                data = np.clip(data + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(data)

            base_p = ROOT / "Dataset" / "soil_image_datset" / "Soil-image-dataset" / "Orignal-Dataset"
            if not base_p.exists():
                base_p = ROOT / "Dataset" / "soil_image_datset" / "Orignal-Dataset"

            classes = [d for d in os.listdir(base_p) if os.path.isdir(base_p / d)]
            chosen_class = random.choice(classes)
            class_path = base_p / chosen_class
            
            all_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            source_file = random.choice(all_files)
            
            with Image.open(class_path / source_file) as img:
                img = img.convert("RGB").resize((224, 224))
                
                # Original
                buffered_orig = io.BytesIO()
                img.save(buffered_orig, format="JPEG")
                orig_b64 = base64.b64encode(buffered_orig.getvalue()).decode('utf-8')
                
                samples = [{"type": "Original Source", "data": orig_b64}]
                
                # Generate 5 Live Augmented variations
                for i in range(5):
                    aug_img = apply_live_augmentation(img)
                    buf = io.BytesIO()
                    aug_img.save(buf, format="JPEG")
                    samples.append({
                        "type": f"In-Memory Augment {i+1}", 
                        "data": base64.b64encode(buf.getvalue()).decode('utf-8')
                    })

            json_response(self, HTTPStatus.OK, {"class": chosen_class, "samples": samples})
            return

        if path == "/api/image-metrics/list":
            json_response(self, HTTPStatus.OK, {"plots": IMAGE_ANALYSIS.list_plots()})
            return

        if path == "/api/image-metrics/plot":
            name = (query.get("name") or [None])[0]
            dpi = (query.get("dpi") or [None])[0]
            if not name:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Missing query param: name"})
                return
            try:
                png = IMAGE_ANALYSIS.render_plot(str(name), dpi=int(dpi) if dpi else 160)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            except KeyError:
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "Unknown plot name"})
                return
            except Exception as exc:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

        if path == "/api/soil/analysis/list":
            json_response(self, HTTPStatus.OK, {"plots": SOIL_ANALYSIS.list_plots()})
            return

        if path == "/api/soil/analysis/plot":
            name = (query.get("name") or [None])[0]
            dpi = (query.get("dpi") or [None])[0]
            if not name:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Missing query param: name"})
                return
            try:
                png = SOIL_ANALYSIS.render_plot(str(name), dpi=int(dpi) if dpi else 160)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            except KeyError:
                json_response(self, HTTPStatus.NOT_FOUND, {"error": "Unknown plot name"})
                return
            except Exception as exc:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

        if path.startswith("/api/soil/analysis"):
            json_response(
                self,
                HTTPStatus.NOT_FOUND,
                {"error": "Unknown soil analysis endpoint", "path": path},
            )
            return

        if path == "/" or path.startswith("/web/"):
            relative = "index.html" if path == "/" else path.removeprefix("/web/")
            file_path = STATIC_DIR / relative
            if file_path.is_file():
                content_type, _ = mimetypes.guess_type(file_path.name)
                body = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"})
            return

        try:
            if path == "/api/soil/predict":
                json_response(self, HTTPStatus.OK, predict_soil(payload))
                return
            if path == "/api/soil/predict-tabular":
                json_response(self, HTTPStatus.OK, predict_soil_tabular(payload))
                return
            if path == "/api/crop/recommend":
                json_response(self, HTTPStatus.OK, predict_crop_recommendation(payload))
                return
            if path == "/api/agent/chat":
                reply = FARM_AGENT.chat(payload.get("message", ""))
                json_response(self, HTTPStatus.OK, {"reply": reply})
                return
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print("Serving on:")
    print(f"  http://127.0.0.1:{port}")
    print(f"  http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
