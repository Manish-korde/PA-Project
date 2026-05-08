from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class PlotSpec:
    name: str
    title: str
    build: Callable[[pd.DataFrame, int], bytes]


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def _png_bytes(fig, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    fig.clf()
    return buf.getvalue()


def _safe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def _maybe_import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def _build_eda_distributions(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    if sns:
        sns.countplot(x="Crop_Type", data=df, ax=ax1)
        ax1.set_title("Crop Type Distribution")
        ax1.tick_params(axis="x", rotation=90)

        sns.countplot(x="Soil_Type", data=df, ax=ax2)
        ax2.set_title("Soil Type Distribution")
        ax2.tick_params(axis="x", rotation=90)

        sns.countplot(x="Irrigation_Available", data=df, ax=ax3)
        ax3.set_title("Irrigation Availability")

        sns.histplot(df["Soil_pH"], kde=True, ax=ax4)
        ax4.set_title("Soil pH Distribution")
    else:
        df["Crop_Type"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Crop Type Distribution")
        ax1.tick_params(axis="x", rotation=90)
        df["Soil_Type"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Soil Type Distribution")
        ax2.tick_params(axis="x", rotation=90)
        df["Irrigation_Available"].value_counts().plot(kind="bar", ax=ax3)
        ax3.set_title("Irrigation Availability")
        df["Soil_pH"].plot(kind="hist", ax=ax4, bins=30)
        ax4.set_title("Soil pH Distribution")

    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_correlation_heatmap(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()
    numeric = df.select_dtypes(include=["float64", "int64", "int32"]).copy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    corr = numeric.corr(numeric_only=True)

    if sns:
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    else:
        ax.imshow(corr.values, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_outlier_boxplot(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()

    cols = ["Soil_pH", "Soil_Nitrogen", "Rainfall", "Temperature", "Humidity"]
    available = [c for c in cols if c in df.columns]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.boxplot(data=df[available], ax=ax)
    else:
        ax.boxplot([df[c].values for c in available], labels=available)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Boxplot for Outlier Detection")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_target_distribution(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns:
        sns.countplot(x="Compatible", data=df, ax=ax)
    else:
        df["Compatible"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Target (Compatible) Distribution")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _prepare_ml(df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    y = df["Compatible"].astype(int)
    X = df.drop(columns=["Compatible"])

    categorical = [c for c in ["Crop_Type", "Soil_Type", "Irrigation_Available"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def _build_classification_accuracy(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    preprocessor, X_train, X_test, y_train, y_test = _prepare_ml(df)

    models: list[tuple[str, Any]] = [
        ("LR", LogisticRegression(max_iter=500)),
        ("DT", DecisionTreeClassifier(random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ]

    rows: list[tuple[str, float]] = []
    for name, estimator in models:
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append((name, float(accuracy_score(y_test, pred))))

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([r[0] for r in rows], [r[1] for r in rows])
    ax.set_ylim(0, 1)
    ax.set_title("Classification Model Accuracy Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    for i, (_, acc) in enumerate(rows):
        ax.text(i, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_confusion_matrices(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    preprocessor, X_train, X_test, y_train, y_test = _prepare_ml(df)
    models: list[tuple[str, Any]] = [
        ("LogReg", LogisticRegression(max_iter=500)),
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ("RandomForest", RandomForestClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ]

    fig = plt.figure(figsize=(12, 9))
    for i, (name, estimator) in enumerate(models):
        ax = fig.add_subplot(2, 2, i + 1)
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        if sns:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        else:
            ax.imshow(cm, cmap="Blues")
            for (r, c), val in ((r, c, cm[r, c]) for r in range(cm.shape[0]) for c in range(cm.shape[1])):
                ax.text(c, r, str(val), ha="center", va="center", color="black")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices (Binary Compatible)", y=1.02)
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_kmeans_elbow(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    from sklearn.cluster import KMeans

    X_cluster = df.select_dtypes(include=["float64", "int64", "int32"]).copy()
    wcss: list[float] = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init="auto")
        kmeans.fit(X_cluster)
        wcss.append(float(kmeans.inertia_))

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, 11), wcss, marker="o")
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_kmeans_scatter(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()
    from sklearn.cluster import KMeans

    X_cluster = df.select_dtypes(include=["float64", "int64", "int32"]).copy()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_cluster)
    plot_df = df.copy()
    plot_df["Cluster"] = clusters

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns and "Temperature" in plot_df.columns and "Rainfall" in plot_df.columns:
        sns.scatterplot(x=plot_df["Temperature"], y=plot_df["Rainfall"], hue=plot_df["Cluster"], ax=ax, s=20)
    else:
        ax.scatter(plot_df["Temperature"], plot_df["Rainfall"], c=plot_df["Cluster"], s=10)
    ax.set_title("KMeans Clustering (Temperature vs Rainfall)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Rainfall")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_hierarchical_dendrogram(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage  # type: ignore
    except Exception:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.01, 0.5, "SciPy not installed; dendrogram unavailable.", transform=ax.transAxes)
        ax.set_axis_off()
        return _png_bytes(fig, dpi)

    X_cluster = df.select_dtypes(include=["float64", "int64", "int32"]).sample(n=min(500, len(df)), random_state=42)
    linked = linkage(X_cluster, method="ward")

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(linked, ax=ax, no_labels=True, color_threshold=None)
    ax.set_title("Hierarchical Clustering Dendrogram (sampled)")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_hierarchical_scatter(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    sns = _maybe_import_seaborn()
    from sklearn.cluster import AgglomerativeClustering

    X_cluster = df.select_dtypes(include=["float64", "int64", "int32"]).copy()
    hc = AgglomerativeClustering(n_clusters=3)
    clusters = hc.fit_predict(X_cluster)
    plot_df = df.copy()
    plot_df["HC_Cluster"] = clusters

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    if sns and "Temperature" in plot_df.columns and "Rainfall" in plot_df.columns:
        sns.scatterplot(x=plot_df["Temperature"], y=plot_df["Rainfall"], hue=plot_df["HC_Cluster"], ax=ax, s=20)
    else:
        ax.scatter(plot_df["Temperature"], plot_df["Rainfall"], c=plot_df["HC_Cluster"], s=10)
    ax.set_title("Hierarchical Clustering (Temperature vs Rainfall)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Rainfall")
    fig.tight_layout()
    return _png_bytes(fig, dpi)


def _build_ann_dnn_curves(df: pd.DataFrame, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline

    preprocessor, X_train, X_test, y_train, y_test = _prepare_ml(df)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    def train_mlp(hidden_layer_sizes: tuple[int, ...]) -> dict[str, Any]:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=40,
            early_stopping=True,
            n_iter_no_change=6,
            validation_fraction=0.2,
            random_state=42,
        )
        pipe = Pipeline([("prep", preprocessor), ("model", mlp)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_test)
        test_acc = float(accuracy_score(y_test, pred))
        curve = getattr(mlp, "validation_scores_", None) or []
        return {"pipe": pipe, "test_acc": test_acc, "val_curve": list(curve), "loss_curve": list(getattr(mlp, "loss_curve_", []))}

    ann = train_mlp((64, 32))
    dnn = train_mlp((128, 64, 32))

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if ann["val_curve"] and dnn["val_curve"]:
        ax1.plot(ann["val_curve"], label=f"ANN val acc (test {ann['test_acc']:.2f})")
        ax1.plot(dnn["val_curve"], label=f"DNN val acc (test {dnn['test_acc']:.2f})")
        ax1.set_title("ANN vs DNN Validation Accuracy (MLPClassifier)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
    else:
        ax1.text(0.01, 0.5, "Validation curves unavailable in this environment.", transform=ax1.transAxes)
        ax1.set_axis_off()

    ax2.bar(["ANN", "DNN"], [ann["test_acc"], dnn["test_acc"]])
    ax2.set_ylim(0, 1)
    ax2.set_title("ANN vs DNN Test Accuracy")
    for i, v in enumerate([ann["test_acc"], dnn["test_acc"]]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return _png_bytes(fig, dpi)


class SoilAnalysisService:
    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path
        self._lock = threading.Lock()
        self._df: Optional[pd.DataFrame] = None
        self._plot_cache: dict[tuple[str, int], bytes] = {}

        self._plots: list[PlotSpec] = [
            PlotSpec("eda_distributions", "Distributions", _build_eda_distributions),
            PlotSpec("correlation_heatmap", "Correlation Heatmap", _build_correlation_heatmap),
            PlotSpec("outlier_boxplot", "Outlier Boxplot", _build_outlier_boxplot),
            PlotSpec("target_distribution", "Target Distribution", _build_target_distribution),
            PlotSpec("classification_accuracy", "Classifier Accuracy", _build_classification_accuracy),
            PlotSpec("confusion_matrices", "Confusion Matrices", _build_confusion_matrices),
            PlotSpec("kmeans_elbow", "KMeans Elbow", _build_kmeans_elbow),
            PlotSpec("kmeans_scatter", "KMeans Scatter", _build_kmeans_scatter),
            PlotSpec("hierarchical_dendrogram", "Hierarchical Dendrogram", _build_hierarchical_dendrogram),
            PlotSpec("hierarchical_scatter", "Hierarchical Scatter", _build_hierarchical_scatter),
            PlotSpec("ann_dnn_curves", "ANN/DNN Curves", _build_ann_dnn_curves),
        ]

    def list_plots(self) -> list[dict[str, str]]:
        return [{"name": plot.name, "title": plot.title} for plot in self._plots]

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self._df = _load_dataframe(self._dataset_path)
        return self._df

    def render_plot(self, name: str, dpi: int = 160) -> bytes:
        dpi = max(80, min(int(dpi), 400))
        with self._lock:
            cache_key = (name, dpi)
            if cache_key in self._plot_cache:
                return self._plot_cache[cache_key]
            df = self._ensure_loaded()
            spec = next((p for p in self._plots if p.name == name), None)
            if not spec:
                raise KeyError(name)
            png = spec.build(df, dpi)
            self._plot_cache[cache_key] = png
            return png

    def clear_cache(self) -> None:
        with self._lock:
            self._plot_cache.clear()

    def render_plot_base64(self, name: str) -> str:
        return base64.b64encode(self.render_plot(name)).decode("ascii")

