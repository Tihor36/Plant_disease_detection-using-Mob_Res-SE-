# 🌿 Mob-Res + SE-Attention: Plant Disease Diagnosis

A lightweight and explainable deep learning model for plant disease classification, built on the **Mob-Res** architecture with novel **Squeeze-and-Excitation (SE) Attention** enhancements.

> **Based on:** *"A lightweight and explainable CNN model for empowering plant disease diagnosis"* — Scientific Reports, 2025

---

## 📌 Highlights

- **Dual-path architecture** — Residual blocks + MobileNetV2 (ImageNet pre-trained)
- **SE-Attention at two strategic points** — spatial feature maps & fused feature vector
- **Two-phase training** — frozen warm-up → fine-tuning with lower LR
- **Explainability** — Grad-CAM, Grad-CAM++, and LIME visualizations
- **38-class** plant disease classification on [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## 🏗️ Architecture

```
Input (128×128×3)
    │
    ├── Path 1: ResBlock(64) → Pool → ResBlock(128) → Pool
    │           → ResBlock(256) [+ SE-Block on spatial features]
    │           → GAP → 256-d
    │
    ├── Path 2: MobileNetV2 (ImageNet, fine-tuned top layers)
    │           → GAP → 1280-d
    │
    └── Concatenate (1536-d)
             │
        ┌────┴────┐
        │ SE-Block │  ← Channel attention on fused features
        └────┬────┘
             │
        Dropout(0.3) → Dense(38, softmax)
```

### Key Improvements Over Original Mob-Res

| Aspect | Original Mob-Res | This Work (Mob-Res + SE) |
|---|---|---|
| Feature recalibration | None | SE block on spatial features (32×32×256) |
| Fusion attention | Simple concatenation | Channel attention on fused 1536-d vector |
| MobileNetV2 training | Fully frozen | Top 30 layers fine-tuned (Phase 2) |
| Training strategy | Single phase | Two-phase (frozen → fine-tune with 10× lower LR) |
| Regularization | — | Dropout (0.3) + LR scheduling |

---

## 📂 Repository Structure

```
├── Mob_Res_SE_Attention.ipynb   # Main notebook (run on Google Colab with GPU)
├── Mob_Res_Original.ipynb       # Original Mob-Res implementation (for comparison)
└── README.md
```

---

## 🚀 Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `Mob_Res_SE_Attention.ipynb` to Colab, or push this repo to GitHub and open directly.

### 2. Enable GPU

Go to **Runtime → Change runtime type → GPU (T4)** → Save

### 3. Run All Cells

Execute cells in order. The notebook will:

1. **Install dependencies** — `lime`, `opencv`, `scikit-learn`, `kagglehub`
2. **Download PlantVillage dataset** — 38 classes, ~54,000 images (color)
3. **Load data** with augmentation — rotation, flips, shifts, zoom (80/20 train-val split)
4. **Build the model** — Mob-Res + SE-Attention (~3.8M parameters)
5. **Train in 2 phases:**
   - Phase 1 (10 epochs): MobileNetV2 frozen, LR = 0.001
   - Phase 2 (30 epochs): Top 30 layers unfrozen, LR = 0.0001
6. **Evaluate** — classification report with per-class precision, recall, F1
7. **Visualize explainability** — Grad-CAM, Grad-CAM++, LIME
8. **Save model** — `.keras` format

**Estimated runtime:** ~60 min on a T4 GPU

---

## 📊 Training Strategy

### Phase 1 — Warm-Up (10 epochs)

- MobileNetV2 weights **frozen**
- Only residual blocks, SE blocks, and classifier are trained
- Optimizer: Adam (LR = 0.001, β₁ = β₂ = 0.9)
- EarlyStopping: patience = 5

### Phase 2 — Fine-Tuning (30 epochs)

- Top 30 MobileNetV2 layers **unfrozen**
- Optimizer: Adam (LR = 0.0001 — 10× lower)
- ReduceLROnPlateau: factor = 0.5, patience = 3
- EarlyStopping: patience = 10

---

## 🔍 Explainability

The notebook includes three explainability methods to interpret model predictions:

| Method | What It Shows |
|---|---|
| **Grad-CAM** | Heatmap of regions the model focuses on per path |
| **Grad-CAM++** | Higher-order gradients for more precise localization |
| **LIME** | Superpixel importance — which regions support/oppose the prediction |

All visualizations are generated for both Path 1 (Residual + SE) and Path 2 (MobileNetV2).

---

## 🛠️ Tech Stack

| Component | Version / Detail |
|---|---|
| Python | 3.10+ |
| TensorFlow / Keras | 2.x |
| MobileNetV2 | ImageNet pre-trained |
| Dataset | [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (color, 38 classes) |
| Explainability | Grad-CAM, Grad-CAM++, LIME |
| Platform | Google Colab (T4 GPU) |

---

## 📄 Citation



```bibtex
@article{moussafir2025lightweight,
  title={A lightweight and explainable CNN model for empowering plant disease diagnosis},
  author={Moussafir, M. and others},
  journal={Scientific Reports},
  year={2025},
  publisher={Nature Publishing Group}
}
```

---

## 📜 License

This project is for academic and research purposes.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) by Abdallah Ali
- Original Mob-Res architecture from *Scientific Reports, 2025*
- SE-Net: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) (Hu et al., 2018)

