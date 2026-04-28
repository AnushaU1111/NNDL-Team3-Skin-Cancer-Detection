# 🔬 Deep Learning for Skin Cancer Detection
### An Ensemble Approach on HAM10000

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![NC State](https://img.shields.io/badge/NC%20State-CSC%20591-CC0000)](https://ncsu.edu)

> Binary classification of dermoscopic images (Benign vs. Malignant) using a weighted ensemble of EfficientNet-B3, EfficientNet-B5, and EVA02 Vision Transformer — achieving **AUC 0.9640** and **93.9% malignant recall** on HAM10000.

---

## 📋 Table of Contents

- [Results](#-results)
- [Dataset](#-dataset)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Pipeline](#-training-pipeline)
- [Evaluation](#-evaluation)
- [References](#-references)

---

## 🏆 Results

| Model | AUC | Recall | F1 | Accuracy |
|-------|-----|--------|----|----------|
| Logistic Regression (baseline) | 0.756 | 89.1% | 0.725 | 70.3% |
| EfficientNet-B3 | 0.9306 | 96.0% | 0.581 | 73.0% |
| EfficientNet-B5 | 0.9331 | 94.2% | 0.625 | 77.9% |
| B5 + Multiclass Pretrain | 0.9364 | 97.3% | 0.575 | 77.4% |
| EVA02 Vision Transformer | 0.9623 | 84.0% | 0.801 | — |
| **Weighted Ensemble (final)** | **0.9640** | **93.9%** | **0.730** | **86.5%** |

> Decision threshold set to **0.40** to prioritize recall in a clinical screening context.  
> Only **18 missed cancers** out of 293 malignant cases in the test set.

---


## 📊 Dataset

**HAM10000** (Human Against Machine with 10,000 training images)  
[Kaggle — skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

| Class | Label | Count | Category |
|-------|-------|-------|----------|
| Melanocytic Nevi | NV | 6,705 | Benign |
| Melanoma | MEL | 1,113 | **Malignant** |
| Benign Keratosis | BKL | 1,099 | Benign |
| Basal Cell Carcinoma | BCC | 514 | **Malignant** |
| Actinic Keratosis | AKIEC | 327 | **Malignant** |
| Dermatofibroma | DF | 115 | Benign |
| Vascular Lesions | VASC | 142 | Benign |

**Binary target:** Malignant = MEL + BCC + AKIEC (1,954) vs Benign = everything else (8,061)  
**Split:** 70% train (7,010) / 15% val (1,502) / 15% test (1,503) — stratified

---

## 🧠 Models

### Phase 1 — Logistic Regression Baseline
Flatten 64×64 images → StandardScaler → PCA → Logistic Regression  
Simple linear pipeline to establish a performance floor.

### Phase 2 — Deep Learning Ensemble

#### Stage 1: Multiclass Pretraining
EfficientNet-B5 trained on all 7 lesion classes with CrossEntropyLoss for 15 epochs. Forces the backbone to learn rich cross-lesion skin texture features before binary specialization.

#### Stage 2: Binary Fine-Tuning with Progressive Unfreeze
Pretrained backbone transferred to a binary head trained with Focal Loss. At epoch 10, full backbone unfrozen with halved learning rate (ULMFiT strategy).

#### Stage 3: EVA02 Vision Transformer
Attention-based model (21.7M params) pretrained on ImageNet-22k. Architecturally orthogonal to EfficientNet — uncorrelated errors maximize ensemble gain.

#### Stage 4: Weighted TTA Ensemble
5-pass test-time augmentation per model. Final prediction = weighted average of all 4 models, weighted by individual validation AUC.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision timm --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas numpy matplotlib Pillow
```

> **GPU recommended.** Training on CPU is possible but will take several hours per epoch.  
> For Google Colab: use Runtime → T4 GPU and run `skin_cancer_advanced.ipynb` directly.

---

## 🚀 Usage

### Run the baseline
```bash
python baseline_logistic_regression.py
```

### Run full ensemble (Google Colab recommended)
Open `Skin Cancer Detection model.ipynb` in Colab with T4 GPU selected.

```python
# Download dataset via Kaggle API
!pip install kaggle
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
!unzip skin-cancer-mnist-ham10000.zip -d data/HAM10000
```

---

## 🔧 Training Pipeline

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Batch size | 16 (B5), 32 (B3) |
| Optimizer | AdamW |
| Scheduler | OneCycleLR (cosine) |
| Backbone LR | 5e-5 |
| Head LR | 5e-4 |
| Weight decay | 1e-4 |
| Epochs | 40 (early stopping, patience=10) |
| Precision | AMP fp16 |

### Training innovations

- **Focal Loss** (α=0.75, γ=2.0) — concentrates gradient on hard malignant cases
- **WeightedRandomSampler** — rebalances every mini-batch to handle 4:1 class imbalance
- **MixUp** (α=0.4) + **CutMix** (α=1.0) — stochastic augmentation per batch
- **Progressive unfreeze** — backbone unfrozen at epoch 10 with halved LR
- **Label smoothing** = 0.1 — prevents overconfidence on noisy labels
- **TTA × 5 passes** — test-time augmentation averaged at inference

---

## 📈 Evaluation

### Decision threshold

The default threshold of 0.5 maximizes F1 (= 0.808). We use **threshold = 0.40** to prioritize recall in a clinical screening context — missing a malignant lesion is far more costly than a false alarm.

| Threshold | Recall | Precision | F1 |
|-----------|--------|-----------|-----|
| 0.40 | **93.9%** | 46.8% | 0.730 |
| 0.50 | 80.0% | 60.5% | **0.808** |

### Reproduce evaluation

```python
# Load checkpoint and run TTA ensemble
ckpt = torch.load('checkpoints/best_model_vit.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

prob_ensemble = predict_tta(model, test_df, n_passes=5)
y_pred = (prob_ensemble >= 0.40).astype(int)
```

---

## 📚 References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). **The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.** *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161

2. Tan, M., & Le, Q. V. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** *ICML 2019*. https://arxiv.org/abs/1905.11946

---

## 👩‍💻 Author

Aditya Purohit, Anusha Upadhyay, Swasti Sadanand
CSC 525 Neural Networks & Deep Learning — Spring 2026

---

*This project was developed as part of the NC State AI Student Symposium 2026.*