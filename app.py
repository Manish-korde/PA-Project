from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
MODELS_DIR = ROOT / "Models"
FINAL_MODELS_DIR = MODELS_DIR / "final models"
DATASET_DIR = ROOT / "Dataset"
SOIL_MODEL_PATH = None
SOIL_ENCODER_PATH = MODELS_DIR / "soil_label_encoder_7class.pkl"

SOIL_ANN_MODEL_PATH = MODELS_DIR / "soil_ann_model_7class.pth"
SOIL_ANN_ENCODER_PATH = MODELS_DIR / "soil_ann_label_encoder_7class.pkl"
SOIL_ANN_SCALER_PATH = MODELS_DIR / "soil_ann_scaler.pkl"
SOIL_ANN_METRICS_PATH = MODELS_DIR / "soil_ann_model_7class_metrics.json"

CROP_COMPARISON_PATH = MODELS_DIR / "crop_recommendation_model_comparison.json"
CROP_ANN_MODEL_PATH = MODELS_DIR / "crop_ann_multiclass_model.pth"
CROP_RF_MODEL_PATH = MODELS_DIR / "crop_random_forest_model.pkl"
CROP_BOOST_MODEL_PATH = MODELS_DIR / "crop_boosting_model.pkl"
CROP_LABEL_ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
CROP_TREE_PREPROCESSOR_PATH = MODELS_DIR / "crop_tree_preprocessor.pkl"
CROP_ANN_PREPROCESSOR_PATH = MODELS_DIR / "crop_ann_preprocessor.pkl"

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


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


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
    best_crop_model = CROP_COMPARISON.get("best_model")
    crop_ready = bool(CROP_COMPARISON and CROP_LABEL_ENCODER and CROP_CLASSES)
    if best_crop_model == "ANN":
        crop_ready = crop_ready and torch is not None and CROP_ANN_MODEL_PATH.exists() and CROP_ANN_PREPROCESSOR is not None
    elif best_crop_model == "RandomForest":
        crop_ready = crop_ready and CROP_RF_MODEL is not None and CROP_TREE_PREPROCESSOR is not None
    elif best_crop_model:
        crop_ready = False
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
    if crop_ready and CROP_COMPARISON.get("best_model"):
        warnings.append(
            f"Crop recommendation artifacts detected. Preferred model: {CROP_COMPARISON['best_model']}."
        )

    return {
        "active_mode": active_mode,
        "crop_recommendation_ready": crop_ready,
        "legacy_ann_ready": legacy_ann_ready,
        "crop_type_count": CROP_COMPARISON.get("crop_type_count"),
        "best_crop_model": best_crop_model,
        "crop_models": CROP_COMPARISON.get("models", []),
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


def parse_crop_payload(payload: dict[str, Any]) -> pd.DataFrame:
    try:
        row = {
            "Soil_Type": str(payload["soil_type"]),
            "Irrigation_Available": int(payload["irrigation_available"]),
            "Farm_Size_Acres": float(payload["farm_size_acres"]),
            "Soil_pH": float(payload["soil_ph"]),
            "Soil_Nitrogen": float(payload["soil_nitrogen"]),
            "Soil_Organic_Matter": float(payload["soil_organic_matter"]),
            "Temperature": float(payload["temperature"]),
            "Rainfall": float(payload["rainfall"]),
            "Humidity": float(payload["humidity"]),
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or invalid crop recommendation parameters: {exc}")
    return pd.DataFrame([row])


def predict_crop_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    if not RECOMMENDATION_AUDIT["crop_recommendation_ready"]:
        raise RuntimeError("Crop recommendation models are not ready.")

    features_df = parse_crop_payload(payload)
    best_model = RECOMMENDATION_AUDIT["best_crop_model"]

    if best_model == "RandomForest":
        transformed = CROP_TREE_PREPROCESSOR.transform(features_df)
        probabilities = CROP_RF_MODEL.predict_proba(transformed)[0]
        model_name = "RandomForest"
    elif best_model == "ANN":
        transformed = CROP_ANN_PREPROCESSOR.transform(features_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CropANN(input_size=transformed.shape[1], num_classes=len(CROP_CLASSES)).to(device)
        model.load_state_dict(torch.load(CROP_ANN_MODEL_PATH, map_location=device))
        model.eval()
        input_tensor = torch.tensor(transformed, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        model_name = "ANN"
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
            f"Current best crop model is {model_name}, but overall accuracy is low "
            f"({CROP_COMPARISON['models'][0]['Test Accuracy']:.3f} test accuracy), so treat recommendations cautiously."
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

    return {
        "label": SOIL_ENCODER_CLASSES[top_index],
        "top_probability": ranked[0]["probability"],
        "ranked_predictions": ranked,
    }


def predict_soil_tabular(payload: dict[str, Any]) -> dict[str, Any]:
    if RECOMMENDATION_AUDIT["crop_recommendation_ready"]:
        return predict_crop_recommendation(payload)

    if not SOIL_ANN_MODEL_PATH.exists() or not SOIL_ANN_ENCODER:
        raise RuntimeError("Soil ANN model is not trained yet.")

    # Features: pH, Nitrogen, Organic_Matter, Temp, Rainfall, Humidity
    try:
        data = [
            float(payload["soil_ph"]),
            float(payload["soil_nitrogen"]),
            float(payload["soil_organic_matter"]),
            float(payload["temperature"]),
            float(payload["rainfall"]),
            float(payload["humidity"]),
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or invalid parameters: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(SOIL_ANN_CLASSES)
    model = SoilANN(input_size=6, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(SOIL_ANN_MODEL_PATH, map_location=device))
    model.eval()

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
        if self.path == "/api/status":
            json_response(
                self,
                HTTPStatus.OK,
                {
                    "soil_audit": SOIL_AUDIT,
                    "recommendation_audit": RECOMMENDATION_AUDIT,
                    "server_urls": {
                        "local": "http://127.0.0.1:8000",
                        "browser_hint": "Use http://127.0.0.1:8000 or http://localhost:8000 in the browser, not http://0.0.0.0:8000.",
                    },
                },
            )
            return

        if self.path == "/" or self.path.startswith("/web/"):
            relative = "index.html" if self.path == "/" else self.path.removeprefix("/web/")
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
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"})
            return

        try:
            if self.path == "/api/soil/predict":
                json_response(self, HTTPStatus.OK, predict_soil(payload))
                return
            if self.path == "/api/soil/predict-tabular":
                json_response(self, HTTPStatus.OK, predict_soil_tabular(payload))
                return
            json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    host = "0.0.0.0"
    port = 8000
    server = ThreadingHTTPServer((host, port), AppHandler)
    print("Serving on:")
    print("  http://127.0.0.1:8000")
    print("  http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
