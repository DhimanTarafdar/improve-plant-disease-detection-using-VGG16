# Plant Disease Detection using CNN (VGG16)

A deep learning project that detects plant diseases from leaf images using **Transfer Learning with VGG16**. Trained on the **PlantVillage dataset**, this model can identify **38 different plant disease classes** with high accuracy.

---

## What This Project Does

Given a photo of a plant leaf, the model tells you:
- **What disease** the plant has (or if it's healthy)
- **How confident** it is (in %)
- The **top 3 most likely predictions**

---

## Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) (via KaggleHub)
- **Classes:** 38 (covering multiple crops and disease types)
- **Split:** Separate `train/` and `val/` folders

---

## Model Architecture

Uses **VGG16** pretrained on ImageNet — the convolutional base is kept frozen (as a feature extractor), and a custom classifier is attached on top:

```
VGG16 Feature Extractor (frozen)
        ↓
Linear(25088 → 4096) + ReLU + Dropout(0.5)
        ↓
Linear(4096 → 1024) + ReLU + Dropout(0.4)
        ↓
Linear(1024 → 38)   ← output classes
```

---

## Training Details

| Setting | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs (Phase 1) | 10 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau |

### Data Augmentation (Training)
Random crop, horizontal & vertical flip, rotation (±30°), and color jitter (brightness, contrast, saturation, hue) — to make the model robust to real-world lighting and orientation variation.

---

## Two-Phase Training Strategy

### Phase 1 — Classifier Training
Only the new classifier layers are trained. The VGG16 convolutional backbone stays frozen. This is fast and avoids overfitting early on.

### Phase 2 — Fine-Tuning
The best Phase 1 model is loaded, then the **last 2 convolutional blocks** (layers 17–30) of VGG16 are unfrozen and trained with a much smaller learning rate (`0.0001`) to gently adapt the pretrained features to plant imagery.

> Phase 1 Best Validation Accuracy baseline: **84.60%**

---

## 🧪 Inference

The model can predict on:
- **Validation set images** (from the dataset)
- **Your own uploaded images** (via Google Colab file upload)

Output includes the predicted class name, confidence percentage, and top-3 predictions with probabilities.

---

## 💾 Saved Models

| File | Description |
|---|---|
| `best_plant_disease_vgg16.pth` | Best model from Phase 1 |
| `best_plant_disease_finetuned.pth` | Best model from Phase 2 (if improved) |
| `plant_disease_finetuned.pth` | Final fine-tuned model (downloadable) |

---

## 🚀 How to Run

1. Open the notebook in **Google Colab** (GPU recommended)
2. Install dependencies:
   ```bash
   pip install kagglehub
   ```
3. Run all cells top to bottom:
   - Downloads dataset automatically via KaggleHub
   - Trains Phase 1 (classifier only)
   - Fine-tunes Phase 2 (last conv blocks)
   - Runs inference on test images
4. Upload your own leaf images when prompted to test the model

---

## 🛠️ Requirements

```
torch
torchvision
kagglehub
Pillow
matplotlib
```

---

## 📊 Training Plots

The notebook generates loss and accuracy curves for both phases, including a reference line showing the Phase 1 best accuracy — so you can visually confirm whether fine-tuning improved performance.

---

## 📁 Project Structure

```
├── Improve_Plant_Disease_Detection_using_CNN.ipynb   # Main notebook
├── best_plant_disease_vgg16.pth                      # Phase 1 best weights
├── best_plant_disease_finetuned.pth                  # Phase 2 best weights
└── plant_disease_finetuned.pth                       # Final model export
```
