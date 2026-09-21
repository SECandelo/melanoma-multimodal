# Multimodal Melanoma Classification with Interpretable Deep Learning

Skin-lesion malignancy classification on **ISIC 2020**, built as a research-grade,
reproducible pipeline: patient-level data splitting, transfer learning with
EfficientNet-B3, an image + clinical-metadata multimodal variant, multi-seed
robustness analysis, threshold selection on validation, and Grad-CAM
interpretability with an explicit confounder audit.

> **TL;DR** — The image-only model reaches **AUC 0.894** on a held-out,
> patient-disjoint test set. Adding clinical metadata (sex, age, anatomical site)
> does **not** improve discrimination beyond run-to-run noise
> (ΔAUC −0.015 ± 0.025 across seeds) — a negative result that is reported
> honestly rather than hidden. Grad-CAM shows the network attends to the lesion
> on confident correct predictions, and exposes dataset confounders
> (surgical-ink markings, rulers, hairs) on the failures.

---

## 1. Motivation

Melanoma is curable when caught early, and dermoscopy-based triage is a natural
target for deep learning. Two questions drive this project:

1. **Does a modern CNN generalise across patients** — not just across images —
   on a strongly imbalanced screening dataset (~1.8 % malignant)?
2. **Do routinely available clinical metadata add information** on top of the
   image, or is the pixel signal already sufficient?

The second question matters practically: metadata are cheap, so a real gain
would justify collecting them. The finding here is that, on this dataset, they
do not — which is itself worth stating.

## 2. Dataset

| | |
|---|---|
| Source | ISIC 2020 Challenge (training set) |
| Images | 33,126 dermoscopic images |
| Positives | ~1.8 % malignant (severe class imbalance) |
| Metadata | patient ID, sex, approximate age, anatomical site |

**Patient-level splitting.** Because a single patient contributes many lesions,
a random image-level split would leak patient identity across train/val/test and
inflate performance. Splits are produced with `StratifiedGroupKFold`, grouping on
`patient_id` and stratifying on the label, so **no patient appears in more than
one split**.

## 3. Methods

- **Backbone:** EfficientNet-B3 (`timm`, ImageNet-pretrained), transfer learning.
- **Image model:** backbone → dropout → linear head (1 logit).
- **Multimodal model:** image features ⊕ a small MLP over metadata
  (sex, age z-scored, one-hot anatomical site; explicit missingness flags),
  fused into a shared head.
- **Imbalance:** `BCEWithLogitsLoss` with `pos_weight = (1−p)/p`.
- **Optimisation:** AdamW, mixed precision (AMP), early stopping on validation AUC.
- **Primary metric:** ROC-AUC (threshold-independent — the right choice under
  heavy imbalance). Sensitivity/specificity are reported at an operating point
  selected on the validation set (see §4).

Two configurations are used and **declared**:

| Config | Resolution | Epochs | Used for |
|---|---|---|---|
| `CFG`   | 256 px | 20 | Main reported models, ROC & Grad-CAM figures |
| `STUDY` | 224 px | 12 | Multi-seed robustness study (reduced for compute budget) |

Within the multi-seed study **all seeds and both models share the identical
`STUDY` configuration**, so the baseline-vs-multimodal comparison is fair.

## 4. Results

### Discrimination (held-out test, patient-disjoint)

| Model | Test AUC (main, 256/20) | Multi-seed AUC (224/12, n=3) |
|---|---|---|
| Image-only (EfficientNet-B3) | **0.894** | 0.887 ± 0.012 |
| Multimodal (+ metadata)      | 0.894 | 0.872 ± 0.030 |
| **Δ (multimodal − baseline)** | ~0.000 | **−0.015 ± 0.025** |

The metadata provide **no discriminative gain**: the difference is smaller than
the seed-to-seed standard deviation. The multimodal model is also *less* stable
(higher variance), consistent with a small head learning noise from
low-information features.

### Operating point (image-only model)

Threshold chosen on the **validation** set via Youden's J, then applied to test:

| | |
|---|---|
| Threshold | 0.656 |
| Sensitivity | 0.840 |
| Specificity | 0.795 |

At the naïve 0.5 threshold the model looks insensitive; the correct,
leakage-free operating point recovers a balanced 0.84 / 0.80 trade-off.

*(ROC comparison: `results/figures_comparison.png`.)*

## 5. Interpretability — Grad-CAM

Grad-CAM was implemented **from scratch** (forward/backward hooks on the last
convolutional layer, `backbone.conv_head`) rather than with a library, to keep
full control of the method.

*(Figure: `results/gradcam.png`.)*

Two readings:

- **Where it works:** on confident, correct malignant cases the activation is
  compact and centred **on the lesion** — evidence the model uses the right
  signal, not background.
- **Where it fails / confounders:** on low-probability outputs the map is diffuse
  and edge-dominated (Grad-CAM of a near-zero score is not informative — a known
  property of the method). More importantly, one malignant example shows the map
  drawn toward **surgical-ink marking**: clinicians mark suspicious lesions more
  often, so ink is a spurious cue the model can exploit. Rulers and hairs are
  further known ISIC confounders. Flagging these is part of the analysis, not an
  afterthought.

## 6. Key takeaways

- A patient-level-honest EfficientNet-B3 reaches **AUC 0.894** on ISIC 2020.
- **Clinical metadata do not help** here — reported transparently with
  multi-seed statistics rather than a single lucky run.
- Correct **operating-point selection** (val-chosen threshold) matters more than
  chasing a few thousandths of AUC.
- Interpretability doubles as **debugging**: Grad-CAM surfaces dataset
  confounders that a headline metric would hide.

## 7. Limitations & next steps

- ISIC 2020 confounders (ink, rulers, hairs) are not removed; a hair-augmentation
  / inpainting ablation would test robustness directly.
- Single-dataset evaluation — external validation (e.g. HAM10000) would test
  generalisation.
- Metadata fusion is simple concatenation; the negative result is specific to
  this design and dataset, not a universal claim.

## 8. Reproducibility

```
# 1. environment (Colab / Kaggle: cv2, timm, albumentations preinstalled)
pip install timm albumentations opencv-python

# 2. data: ISIC 2020 train set -> CSV_PATH, IMAGES_DIR

# 3. run, in order:
#    - setup cell (configs, datasets, models, train/eval functions)
#    - main models (image-only + multimodal, size 256)  -> saves *_best.pt, test arrays
#    - multi-seed study (seeds 42/1/7)                   -> multiseed_log.json
#    - operating-point cell (val-chosen threshold)
#    - Grad-CAM cell                                     -> results/gradcam.png
```

All artefacts (weights, test predictions, logs, figures) are written to a
persistent results directory so the analysis survives session resets.

### Single-image inference

Once trained, the model is portable: everything it learned lives in the saved
weights (`results/classifier_best.pt`), so `predict.py` runs on any machine —
CPU-only, offline — without the dataset or any retraining.

```
python predict.py --image lesion.jpg                 # label + malignancy probability
python predict.py --image lesion.jpg --gradcam       # also saves a Grad-CAM overlay
python predict.py --image lesion.jpg --threshold 0.5 # override the operating point
```

The decision threshold defaults to 0.656 (chosen on the validation set), not
0.5, and is exposed as a CLI argument. **Domain note:** the network is trained
on ISIC dermoscopic images — a phone photo of a mole is out-of-domain and will
not give a reliable prediction. This is a demonstration project, not a medical
device.

## 9. Repository structure

```
.
├── notebook.ipynb            # end-to-end pipeline (Colab)
├── results/
│   ├── classifier_best.pt    # image-only weights
│   ├── multimodal_best.pt    # multimodal weights
│   ├── *_test_y.npy / *_test_p.npy   # saved test labels & predictions
│   ├── multiseed_log.json    # per-seed AUCs
│   ├── figures_comparison.png
│   └── gradcam.png
└── README.md
```

---

*Built with PyTorch, timm, albumentations and scikit-learn.
Author: Simone Edoardo Candelo.*
