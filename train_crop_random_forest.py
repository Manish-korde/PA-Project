from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Dataset" / "solit_dataset" / "Soil-Climate-data.csv"
MODELS_DIR = ROOT / "Models"
MODEL_PATH = MODELS_DIR / "crop_random_forest_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "crop_label_encoder.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "crop_tree_preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "crop_random_forest_metrics.json"
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


def load_dataset() -> tuple[pd.DataFrame, list[int], LabelEncoder]:
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, y, label_encoder = load_dataset()
    class_names = list(label_encoder.classes_)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, y)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train_processed, y_train)

    val_prob = model.predict_proba(X_val_processed)
    test_prob = model.predict_proba(X_test_processed)
    val_pred = val_prob.argmax(axis=1)
    test_pred = test_prob.argmax(axis=1)

    metrics = {
        "model_name": "RandomForest",
        "task": "crop recommendation",
        "framework": "scikit-learn",
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
        "best_val_accuracy": accuracy_score(y_val, val_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "top3_accuracy": top_k_accuracy_score(y_test, test_prob, k=3, labels=list(range(len(class_names)))),
        "test_log_loss": log_loss(y_test, test_prob),
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "label_encoder_path": str(LABEL_ENCODER_PATH),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_comparison(metrics, class_names, len(X_train), len(X_val), len(X_test))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
