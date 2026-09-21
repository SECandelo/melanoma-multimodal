#!/usr/bin/env python3
"""
predict.py - single-image inference with the trained image-only melanoma
classifier (EfficientNet-B3, ISIC 2020).

Examples
--------
    python predict.py --image neo.jpg
    python predict.py --image neo.jpg --gradcam
    python predict.py --image neo.jpg --weights results/classifier_best.pt --threshold 0.66

Notes
-----
* The model is trained on ISIC 2020 *dermoscopic* images. A phone photo of a
  mole is out-of-domain and will NOT give a reliable prediction.
* This is a demonstration project, not a medical device.
"""
import argparse, os, sys
import cv2, numpy as np, torch, torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------------------------------------------- defaults
BACKBONE  = "efficientnet_b3"
SIZE      = 256          # must match the training resolution
THRESHOLD = 0.656        # chosen on the validation set (Youden's J), not 0.5
WEIGHTS   = "results/classifier_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- model
class Net(nn.Module):
    """EfficientNet backbone + dropout + single-logit head (image-only)."""
    def __init__(self, backbone=BACKBONE, dropout=0.3):
        super().__init__()
        # pretrained=False: we load our own fine-tuned weights right after
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.backbone.num_features, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def load_model(weights, backbone=BACKBONE):
    if not os.path.isfile(weights):
        sys.exit(f"[error] weights not found: {weights}")
    model = Net(backbone).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()                       # inference mode: no dropout / BN update
    return model


# ----------------------------------------------------------------- preprocessing
def make_tfm(size=SIZE):
    # SAME transform as validation: resize + normalize, NO augmentation
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


def load_image(path, size=SIZE):
    if not os.path.isfile(path):
        sys.exit(f"[error] image not found: {path}")
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"[error] could not read image (unsupported/corrupt): {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = make_tfm(size)(image=rgb)["image"].unsqueeze(0)   # (1, 3, H, W)
    return rgb, x


# ----------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, x, threshold=THRESHOLD):
    prob = torch.sigmoid(model(x.to(device))).item()      # probability in [0, 1]
    label = "MALIGNANT" if prob >= threshold else "benign"
    return label, prob


# ----------------------------------------------------------------- grad-cam
def gradcam(model, x, size=SIZE):
    """From-scratch Grad-CAM on the last conv layer (backbone.conv_head)."""
    feat, grad = {}, {}
    layer = model.backbone.conv_head
    h1 = layer.register_forward_hook(lambda m, i, o: feat.__setitem__("v", o.detach()))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grad.__setitem__("v", go[0].detach()))

    model.zero_grad()
    score = model(x.to(device)).squeeze()   # scalar logit for "malignant"
    score.backward()                        # gradient flows back to conv_head

    A_ = feat["v"][0]                        # (C, h, w) activations
    G  = grad["v"][0]                        # (C, h, w) gradients
    w  = G.mean(dim=(1, 2))                  # (C,) channel importance weights
    cam = torch.relu((w[:, None, None] * A_).sum(0))   # (h, w) weighted sum + ReLU
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]

    h1.remove(); h2.remove()
    return cam.cpu().numpy()


def save_gradcam(rgb, cam, label, prob, out, size=SIZE):
    import matplotlib
    matplotlib.use("Agg")                    # headless-safe (no display needed)
    import matplotlib.pyplot as plt
    disp = cv2.resize(rgb, (size, size))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(disp);            ax[0].set_title("input");    ax[0].axis("off")
    ax[1].imshow(disp)
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title("Grad-CAM");   ax[1].axis("off")
    fig.suptitle(f"{label}  |  p(malignant) = {prob:.1%}")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description="Melanoma image classifier - single-image inference.")
    ap.add_argument("--image", required=True, help="path to the dermoscopic image")
    ap.add_argument("--weights", default=WEIGHTS, help=f"model weights (default: {WEIGHTS})")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"decision threshold (default: {THRESHOLD}, chosen on validation)")
    ap.add_argument("--size", type=int, default=SIZE, help=f"input size (default: {SIZE})")
    ap.add_argument("--backbone", default=BACKBONE, help=f"timm backbone (default: {BACKBONE})")
    ap.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM overlay")
    ap.add_argument("--out", default="gradcam_pred.png", help="Grad-CAM output path")
    args = ap.parse_args()

    model = load_model(args.weights, args.backbone)
    rgb, x = load_image(args.image, args.size)
    label, prob = predict(model, x, args.threshold)

    print(f"\n  image      : {args.image}")
    print(f"  prediction : {label}")
    print(f"  p(malignant): {prob:.1%}   (threshold {args.threshold})\n")

    if args.gradcam:
        cam = gradcam(model, x, args.size)
        save_gradcam(rgb, cam, label, prob, args.out, args.size)
        print(f"  Grad-CAM saved to: {args.out}\n")


if __name__ == "__main__":
    main()
