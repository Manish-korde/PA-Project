from __future__ import annotations
import base64
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class PlotSpec:
    name: str
    title: str
    description: str
    build: Callable[[Any], bytes]

def _png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()

def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _maybe_import_seaborn():
    try:
        import seaborn as sns
        return sns
    except Exception:
        return None

def _build_image_distribution(dataset_path: Path) -> bytes:
    plt = _safe_import_matplotlib()
    
    # We look for the dataset structure
    search_path = dataset_path / "Soil-image-dataset" / "Orignal-Dataset"
    if not search_path.exists():
        search_path = dataset_path / "Orignal-Dataset"
    
    if not search_path.exists():
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"Dataset not found at {search_path}", ha='center')
        return _png_bytes(fig)

    folders = [f for f in os.listdir(search_path) if os.path.isdir(search_path / f)]
    counts = [len(os.listdir(search_path / f)) for f in folders]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    colors = plt.cm.viridis([i/len(folders) for i in range(len(folders))])
    bars = ax.bar(folders, counts, color=colors)
    ax.set_title("Number of Images per Soil Type (Source Dataset)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Soil Type", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    return _png_bytes(fig)

def _build_training_summary(metrics: dict) -> bytes:
    plt = _safe_import_matplotlib()
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    labels = ['Train Accuracy', 'Val Accuracy', 'Test Accuracy']
    values = [
        metrics.get('final_train_accuracy', 0),
        metrics.get('final_val_accuracy', 0),
        metrics.get('test_accuracy', 0)
    ]
    
    # Filter out zeros
    plot_data = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not plot_data:
        plt.text(0.5, 0.5, "No accuracy data available in metrics", ha='center')
        return _png_bytes(fig)
    
    p_labels, p_values = zip(*plot_data)
    
    bars = ax.bar(p_labels, p_values, color=['#4ade80', '#3b82f6', '#f59e0b'])
    ax.set_ylim(0, 1.1)
    ax.set_title("CNN Model Performance Comparison", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Accuracy Score", fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add info text
    info_text = f"Epochs Ran: {metrics.get('epochs_ran', 'N/A')}\nBest Epoch: {metrics.get('best_epoch', 'N/A')}"
    plt.figtext(0.15, 0.8, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout()
    return _png_bytes(fig)

def _build_data_split(metrics: dict) -> bytes:
    plt = _safe_import_matplotlib()
    
    counts = metrics.get('counts', {})
    if not counts:
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No split data available", ha='center')
        return _png_bytes(fig)
        
    labels = ['Train', 'Validation', 'Test']
    sizes = [counts.get('train', 0), counts.get('val', 0), counts.get('test', 0)]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    colors = ['#10b981', '#6366f1', '#f59e0b']
    explode = (0.05, 0, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=140, colors=colors, textprops={'fontsize': 12})
    ax.set_title("Dataset Split Overview", fontsize=14, fontweight='bold', pad=20)
    
    # Add raw counts as legend
    legend_labels = [f"{l}: {s} images" for l, s in zip(labels, sizes)]
    ax.legend(legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    fig.tight_layout()
    return _png_bytes(fig)

class ImageAnalysisService:
    def __init__(self, dataset_path: Path, metrics_path: Path) -> None:
        self._dataset_path = dataset_path
        self._metrics_path = metrics_path
        self._lock = threading.Lock()
        self._metrics: Optional[dict] = None
        self._plot_cache: dict[str, bytes] = {}
        
        self._plots = [
            PlotSpec(
                "image_distribution", 
                "Dataset Distribution", 
                "Visualizes the number of images available for each soil class in the source dataset.",
                _build_image_distribution
            ),
            PlotSpec(
                "training_summary", 
                "Model Performance", 
                "Compares the training, validation, and final test accuracy of the CNN model.",
                _build_training_summary
            ),
            PlotSpec(
                "data_split", 
                "Data Split Overview", 
                "Shows how the data was partitioned into training, validation, and test sets.",
                _build_data_split
            )
        ]

    def _load_metrics(self) -> dict:
        if self._metrics is None:
            if self._metrics_path.exists():
                self._metrics = json.loads(self._metrics_path.read_text(encoding="utf-8"))
            else:
                self._metrics = {}
        return self._metrics

    def list_plots(self) -> list[dict[str, str]]:
        return [{"name": p.name, "title": p.title, "description": p.description} for p in self._plots]

    def render_plot(self, name: str) -> bytes:
        with self._lock:
            if name in self._plot_cache:
                return self._plot_cache[name]
            
            spec = next((p for p in self._plots if p.name == name), None)
            if not spec:
                raise KeyError(name)
            
            if name == "image_distribution":
                png = spec.build(self._dataset_path)
            else:
                png = spec.build(self._load_metrics())
                
            self._plot_cache[name] = png
            return png
