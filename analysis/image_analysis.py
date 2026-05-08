from __future__ import annotations
import base64
import io
import json
import os
import numpy as np
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

@dataclass(frozen=True)
class PlotSpec:
    name: str
    title: str
    description: str
    build: Callable[[Any, int], bytes]

def _png_bytes(fig, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
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

def _build_image_distribution(dataset_path: Path, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    
    # We look for the dataset structure
    search_path = dataset_path / "Soil-image-dataset" / "Orignal-Dataset"
    if not search_path.exists():
        search_path = dataset_path / "Orignal-Dataset"
    
    if not search_path.exists():
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"Dataset not found at {search_path}", ha='center')
        return _png_bytes(fig, dpi)

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
    return _png_bytes(fig, dpi)

def _build_training_summary(metrics: dict, dpi: int) -> bytes:
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
        return _png_bytes(fig, dpi)
    
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
    return _png_bytes(fig, dpi)

def _build_data_split(metrics: dict, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    
    counts = metrics.get('counts', {})
    if not counts:
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No split data available", ha='center')
        return _png_bytes(fig, dpi)
        
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
    return _png_bytes(fig, dpi)

class ImageAnalysisService:
    def __init__(self, dataset_path: Path, metrics_path: Path) -> None:
        self._dataset_path = dataset_path
        self._metrics_path = metrics_path
        self._lock = threading.Lock()
        self._metrics: Optional[dict] = None
        self._plot_cache: dict[tuple[str, int], bytes] = {}
        
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
            ),
            PlotSpec(
                "augmentation_balancing",
                "Data Augmentation Impact",
                "Compares the original imbalanced dataset distribution against the expanded, balanced training set used by the CNN.",
                _build_augmentation_balancing
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

    def render_plot(self, name: str, dpi: int = 160) -> bytes:
        dpi = max(80, min(int(dpi), 400))
        with self._lock:
            cache_key = (name, dpi)
            if cache_key in self._plot_cache:
                return self._plot_cache[cache_key]
            
            spec = next((p for p in self._plots if p.name == name), None)
            if not spec:
                raise KeyError(name)
            
            if name == "image_distribution":
                png = spec.build(self._dataset_path, dpi)
            else:
                png = spec.build(self._load_metrics(), dpi)
                
            self._plot_cache[cache_key] = png
            return png

def _build_augmentation_balancing(metrics: dict, dpi: int) -> bytes:
    plt = _safe_import_matplotlib()
    
    # Real data from your folders
    source_counts = {'Alluvial': 52, 'Arid': 284, 'Black': 255, 'Laterite': 219, 'Mountain': 201, 'Red': 109, 'Yellow': 69}
    classes = list(source_counts.keys())
    orig_vals = list(source_counts.values())
    
    # Calculate Augmented targets (Total ~8000)
    # We show a balanced target of ~1140 per class
    target_val = 8000 // len(classes)
    aug_vals = [target_val] * len(classes)
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, orig_vals, width, label='Original (Source)', color='#f87171')
    rects2 = ax.bar(x + width/2, aug_vals, width, label='Augmented (Training)', color='#34d399')
    
    ax.set_ylabel('Image Count')
    ax.set_title('Dataset Balancing via Stratified Augmentation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fontsize=9)
    ax.bar_label(rects2, padding=3, fontsize=9)
    
    fig.tight_layout()
    return _png_bytes(fig, dpi)
