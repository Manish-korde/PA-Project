# Soil CNN Updated Model Report

## Model Artifact

- Model file: `C:\Users\manis\OneDrive\Desktop\PA-project\Models\soil_cnn_updated_model.pth`
- Metrics file: `C:\Users\manis\OneDrive\Desktop\PA-project\Models\soil_cnn_updated_model_metrics.json`
- Encoder file: `C:\Users\manis\OneDrive\Desktop\PA-project\Models\soil_label_encoder_7class.pkl`

## Runtime

- Framework: `PyTorch`
- PyTorch version: `2.5.1+cu121`
- GPU available: `true`
- Device: `NVIDIA GeForce RTX 4050 Laptop GPU`
- GPU count: `1`

## Classes

- `Alluvial_Soil`
- `Arid_Soil`
- `Black_Soil`
- `Laterite_Soil`
- `Mountain_Soil`
- `Red_Soil`
- `Yellow_Soil`

## Data Split

- Source dataset: `C:\Users\manis\OneDrive\Desktop\PA-project\Dataset\soil_image_datset\Soil-image-dataset\Orignal-Dataset`
- Split type: `stratified`
- Random state: `42`
- Train fraction: `0.70`
- Validation fraction: `0.15`
- Test fraction: `0.15`
- Augmentation: `train only`

## Sample Counts

- Train samples per epoch after augmentation: `8000`
- Train source images: `832`
- Validation images: `178`
- Test images: `179`

## Training Results

- Epochs run: `20`
- Best epoch: `14`
- Final train accuracy: `0.996625`
- Final validation accuracy: `0.8146067415730337`
- Best validation accuracy: `0.8314606741573034`
- Test accuracy: `0.8715083798882681`

## Notes

- The train count of `8000` is an augmented effective epoch size created from `832` real training images.
- Validation and test results are measured on untouched real images only.
