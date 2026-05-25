# Phase 4 — Anomaly Detection Training (PatchCore + EfficientAD)

> **Scope:** Backend + model training only. Builds two complementary **unsupervised** anomaly detectors that operate on the component ROIs cropped by the Phase 3 detector. PatchCore (DINOv2 features, memory bank) provides high-fidelity scoring; EfficientAD (student-teacher on raw patches) provides real-time scoring. The two scores feed the Phase 6 fault decision unit. Crucially, anomaly models train on **normal-only** data — they generalize to faults the labeled classifier (Phase 5) has never seen, enabling open-world detection of the long tail.

---

## 1. Phase objective

The fault classifier (Phase 5) can only flag faults that appear in the labeled set. Industrial UPS systems generate **novel** failure modes constantly: a new capacitor vendor with a slightly different bulge geometry, an unusual leakage pattern, an unseen burn mark. Anomaly detection covers this open-world tail.

| Model | Role | Latency target |
|---|---|---|
| **PatchCore** (DINOv2-L/14 features, coreset memory bank) | "Slow but accurate" — primary anomaly signal | ≤ 120 ms / crop on RTX 3060 |
| **EfficientAD** (small student-teacher on raw 256² patches) | "Fast and dense" — pixel-level heatmap, real-time | ≤ 15 ms / crop on RTX 3060 |
| **Score fusion** (Phase 6) | Calibrated, per-component, OR-of-experts gate | n/a |

Deliverables:

| Deliverable | Artifact |
|---|---|
| Normals-only dataset (per component) | `dataset_v2/normals_only/<component>/{train/good,val/{good,defect}}/` (Phase 2 populates skeleton) |
| Per-component PatchCore memory bank | `runs/anomaly/patchcore/<component>/{memory_bank.pt, threshold.json, metadata.json}` |
| Per-component EfficientAD weights | `runs/anomaly/efficientad/<component>/{teacher.pt, student.pt, autoencoder.pt, threshold.json, metadata.json}` |
| DINOv2 feature extractor (frozen, shared) | `models/dinov2/dinov2_vitl14.pth` (downloaded once) |
| Training scripts | `scripts/anomaly/train_patchcore.py`, `train_efficientad.py` |
| Eval pack per component | `runs/anomaly/<model>/<component>/evaluation/{auroc_image.json, auroc_pixel.json, pr_curve.png}` |
| Inference adapter | `powervision/anomaly/{patchcore.py, efficientad.py, fusion.py}` |
| ONNX export for EfficientAD | `runs/anomaly/efficientad/<component>/student.onnx` |
| MLflow registry | `powervision-patchcore/<component>/v<N>`, `powervision-efficientad/<component>/v<N>` |

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Format |
|---|---|---|
| Component ROI crops (normal only for train) | `dataset_v2/normals_only/<component>/train/good/*.jpg` | JPEG |
| Component ROI crops (good + defect for val) | `dataset_v2/normals_only/<component>/val/{good,defect}/*.jpg` | JPEG |
| Anomaly ground-truth masks (val/test) | `dataset_v2/labels/anomaly_masks/<split>/*.png` mapped to crops | PNG 0/255 |
| Component taxonomy | `taxonomy/component_taxonomy.yaml` | YAML |
| Phase 2 preproc pipeline | `powervision.preproc.pipeline.serve_preprocess_classifier` | Python |
| DINOv2 ViT-L/14 weights | Hugging Face / Facebook AI Research | PyTorch checkpoint |

### 2.2 Outputs

```text
runs/anomaly/
├── patchcore/
│   ├── electrolytic_capacitor/
│   │   ├── memory_bank.pt          # coreset features (float16, shape: [K, D])
│   │   ├── threshold.json          # {"score_threshold": 0.42, "method": "fpr_at_recall_95"}
│   │   ├── metadata.json           # {taxonomy_version, dinov2_revision, train_count, ...}
│   │   └── evaluation/
│   │       ├── auroc_image.json
│   │       ├── auroc_pixel.json
│   │       ├── pr_curve.png
│   │       └── heatmap_examples/
│   ├── battery_terminal_post/...
│   └── ...
└── efficientad/
    ├── electrolytic_capacitor/
    │   ├── teacher.pt
    │   ├── student.pt
    │   ├── student.onnx
    │   ├── autoencoder.pt
    │   ├── pdn_normalization.json  # per-feature mean/std for student-teacher diff
    │   ├── threshold.json
    │   ├── metadata.json
    │   └── evaluation/
    │       ├── auroc_image.json
    │       ├── auroc_pixel.json
    │       └── pixel_pro_curve.png
    └── ...

models/dinov2/dinov2_vitl14.pth    # shared, frozen
```

### 2.3 Format contracts

- **Memory bank**: PyTorch tensor saved via `torch.save(dict(features=tensor, names=list[str]))`. Shape: `[K, D]` where `K = coreset_size` (default 1024 per component), `D = DINOv2 patch dim = 1024`. Stored in `float16` for memory efficiency.
- **Threshold JSON**: `{"score_threshold": float, "method": "<calibration_method>", "fpr_at_threshold": float, "tpr_at_threshold": float, "computed_at": "<iso8601>"}`. Computed by the calibration script (§5.3).
- **Per-component artifacts** are mutually independent — different UPS sites can serve different versions per component without orchestration.

---

## 3. End-to-end dataset creation pipeline

### 3.1 Why normal-only?

PatchCore and EfficientAD are unsupervised: they learn what "normal" looks like and flag anything that differs. Two consequences:

1. The training set must contain **only** normal examples; even a few defective images at training time poison the memory bank / teacher.
2. Validation needs **both** normal and defective examples to choose a score threshold and evaluate AUROC.

### 3.2 Defining "normal" per component

A crop counts as normal if **all** of the following hold:

- The crop's source image has `is_normal == True` in `manifest.parquet` (no fault bboxes anywhere in the image).
- The component bbox does **not** overlap any fault bbox in the source image (IoU > 0 with any fault bbox disqualifies).
- The component is fully visible (CVAT `occlusion < 0.25`, `truncation < 0.25`).
- The crop passes IQA (`iqa_check` from Phase 2 §6.5).

`scripts/anomaly/build_anomaly_dataset.py` enforces all four.

### 3.3 Dataset population script

```python
"""Build dataset_v2/normals_only/<component>/{train/good,val/{good,defect}}/
from the curated dataset_v2/ + manifest.parquet."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd, cv2, numpy as np

from powervision.preproc.quality import iqa_check

ROOT = Path("dataset_v2")
TAX  = Path("taxonomy")

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0

def _yolo_to_xyxy(line, w, h):
    c, cx, cy, bw, bh = line.split(); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
    x1 = (cx - bw/2) * w; x2 = (cx + bw/2) * w
    y1 = (cy - bh/2) * h; y2 = (cy + bh/2) * h
    return int(c), (x1, y1, x2, y2)

def build_for_component(component_id: int, component_name: str, df: pd.DataFrame,
                         pad_ratio: float = 0.15, min_side: int = 96):
    out = ROOT / "normals_only" / component_name
    for sub in ("train/good", "val/good", "val/defect", "test/good", "test/defect"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        img_path = ROOT / row["rgb_path"]
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        cmp_lbl = ROOT / "labels" / "components" / row["split"] / f"{row['image_id']}.txt"
        flt_lbl = ROOT / "labels" / "faults"     / row["split"] / f"{row['image_id']}.txt"
        if not cmp_lbl.exists(): continue
        cmp_boxes = [(_yolo_to_xyxy(l, w, h)) for l in cmp_lbl.read_text().splitlines() if l.strip()]
        flt_boxes = ([(_yolo_to_xyxy(l, w, h)) for l in flt_lbl.read_text().splitlines() if l.strip()]
                     if flt_lbl.exists() else [])

        for idx, (cid, cbox) in enumerate(cmp_boxes):
            if cid != component_id: continue
            overlaps_fault = any(_bbox_iou(cbox, fbox) > 0 for _, fbox in flt_boxes)

            # Pad and crop
            x1, y1, x2, y2 = cbox; bw, bh = x2-x1, y2-y1
            x1 = max(0, int(x1 - bw*pad_ratio)); x2 = min(w, int(x2 + bw*pad_ratio))
            y1 = max(0, int(y1 - bh*pad_ratio)); y2 = min(h, int(y2 + bh*pad_ratio))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < min_side: continue
            ok, _ = iqa_check(crop)
            if not ok: continue

            stem = f"{row['image_id']}__{idx:02d}.jpg"
            if row["is_normal"] and not overlaps_fault:
                dest = out / row["split"] / "good" / stem
            elif overlaps_fault:
                # Only val/test use defects (train must be normal-only)
                if row["split"] == "train": continue
                dest = out / row["split"] / "defect" / stem
            else:
                continue
            cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_v2/manifest.parquet")
    ap.add_argument("--taxonomy", default="taxonomy/component_taxonomy.yaml")
    args = ap.parse_args()
    import yaml
    comps = yaml.safe_load(Path(args.taxonomy).read_text())["components"]
    df = pd.read_parquet(args.manifest)
    for c in comps:
        build_for_component(c["id"], c["name"], df)

if __name__ == "__main__":
    main()
```

### 3.4 Minimum data requirements per component

| Set | Floor | Recommended |
|---|---|---|
| `train/good` | 50 | 300–1000 |
| `val/good`   | 20 | 100 |
| `val/defect` | 10 (mix of fault types) | 50 |
| `test/good`  | 20 | 100 |
| `test/defect`| 10 | 50 |

Components failing the floor → **skip anomaly training**; that component falls back to detection-only (Phase 3) + rule engine in Phase 6.

### 3.5 Class balancing inside `val/defect`

The val/defect bucket is a mix of different fault types (it is unsupervised — model does not know which fault). For meaningful threshold calibration, ensure the defect bucket spans the **fault modes** linked to that component in `fault_taxonomy.yaml`. The build script logs `dataset_v2/normals_only/<component>/_distribution.json`:

```json
{
  "train_good": 412,
  "val_good": 88,
  "val_defect_by_fault_id": {"9": 22, "10": 18, "12": 5},
  "test_good": 96,
  "test_defect_by_fault_id": {"9": 24, "10": 19, "12": 5}
}
```

### 3.6 Augmentation strategy

**Training-side augmentation for normal-only models is delicate.** Heavy augmentation distorts the "normal" distribution, raising false-positive rates. Use the minimal recipe:

```python
# powervision/anomaly/aug.py
import albumentations as A

NORMAL_TRAIN = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(256, 256, border_mode=0, value=(0, 0, 0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
    A.Rotate(limit=5, border_mode=0, p=0.3),
])

EVAL = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(256, 256, border_mode=0, value=(0, 0, 0)),
])
```

Heavy aug (cutout, blur, mosaic) is **forbidden** here. If under-fitting, collect more good crops.

### 3.7 Dataset versioning

`dataset_v2/normals_only/` is DVC-tracked. Each component subdirectory bumps independently when:

- New `train/good` crops added → bump `vN.M.+1` (e.g. `electrolytic_capacitor-v2.0.1`)
- Threshold recalibrated only → bump in the **model** registry, not dataset

---

## 4. Model 1 — PatchCore

### 4.1 Architecture

```
Input crop (256×256 BGR)
  → preprocess: BGR→RGB, ImageNet normalize, resize to 256 (already)
  → DINOv2-L/14 patch embeddings (frozen)
       output: [B, N_patches=256, D=1024]   (16×16 patches at 14px each ≈ 224, padded to 256)
  → Layer hooks: average features from layers 2 & 3 (blocks 11 and 21 of ViT-L)
  → Locally-aware feature aggregation (Patchcore's neighborhood pooling)
  → Memory bank: coreset-subsampled normal patch features (K=1024, D=1024)
At inference:
  → For each query patch: nearest-neighbor distance to memory bank
  → Image score = max(patch_scores)
  → Heatmap   = bilinear up-sampled patch_scores
```

### 4.2 Why DINOv2?

Original PatchCore uses ImageNet-pretrained WideResNet-50. DINOv2-L/14 has been the strongest publicly available self-supervised vision backbone for the last 18 months and consistently produces ~3–8 AUROC points improvement on MVTec-AD-like industrial datasets. Frozen DINOv2 features generalize across components without any fine-tuning, which keeps the per-component pipeline lightweight (only the memory bank differs per component).

### 4.3 Training script: `scripts/anomaly/train_patchcore.py`

```python
"""Build a PatchCore memory bank for one component.

  python scripts/anomaly/train_patchcore.py \
    --component electrolytic_capacitor \
    --coreset 1024 \
    --device cuda:0
"""
from __future__ import annotations
import argparse, json, math
from datetime import datetime
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

from powervision.anomaly.dinov2_backbone import DINOv2Patches
from powervision.anomaly.coreset import greedy_coreset
from powervision.anomaly.dataset import NormalOnlyDataset, EvalDataset
from powervision.anomaly.calibrate import calibrate_threshold

ROOT = Path("dataset_v2/normals_only")
OUT  = Path("runs/anomaly/patchcore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True)
    ap.add_argument("--coreset", type=int, default=1024)
    ap.add_argument("--neigh", type=int, default=3, help="locally-aware pooling neighborhood radius")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    comp_dir = ROOT / args.component
    out_dir  = OUT / args.component
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    backbone = DINOv2Patches(model="dinov2_vitl14", layers=(2, 3), device=device).eval()

    ds = NormalOnlyDataset(comp_dir / "train" / "good", size=256)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    all_features: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            feats = backbone(x)                          # [B, H*W, D]
            feats = _locally_aware_pool(feats, args.neigh)
            all_features.append(feats.reshape(-1, feats.shape[-1]).cpu())
    feats = torch.cat(all_features, dim=0)               # [total_patches, D]

    # Coreset subsampling — greedy k-center, target K
    idx = greedy_coreset(feats.numpy(), k=min(args.coreset, len(feats)))
    bank = feats[idx].to(torch.float16)

    # Persist
    torch.save({"features": bank, "neigh": args.neigh}, out_dir / "memory_bank.pt")

    # Calibrate threshold on val
    eval_ds = EvalDataset(comp_dir / "val", size=256)
    threshold_info = calibrate_threshold(
        scorer=_make_patchcore_scorer(bank, backbone, args.neigh),
        eval_dataset=eval_ds,
        target_recall=0.95,
        device=device,
    )
    (out_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2))

    (out_dir / "metadata.json").write_text(json.dumps({
        "taxonomy_version": "2.0.0",
        "preproc_version": "2.0.0",
        "backbone": "dinov2_vitl14",
        "backbone_layers": [2, 3],
        "coreset_size": int(bank.shape[0]),
        "feature_dim": int(bank.shape[1]),
        "neigh": args.neigh,
        "train_count": len(ds),
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))

def _locally_aware_pool(feats: torch.Tensor, neigh: int) -> torch.Tensor:
    """Average pool over a (2*neigh+1)² neighborhood on the 2D patch grid."""
    B, N, D = feats.shape; H = W = int(math.sqrt(N))
    x = feats.transpose(1, 2).reshape(B, D, H, W)
    x = F.avg_pool2d(x, kernel_size=2*neigh+1, stride=1, padding=neigh)
    return x.reshape(B, D, -1).transpose(1, 2)

def _make_patchcore_scorer(bank, backbone, neigh):
    bank = bank.float()
    def score(img_chw_tensor: torch.Tensor):
        with torch.inference_mode():
            feats = backbone(img_chw_tensor.unsqueeze(0))
            feats = _locally_aware_pool(feats, neigh)
            q = feats.reshape(-1, feats.shape[-1])      # [N, D]
            # Pairwise L2 to bank, take min across bank
            d2 = torch.cdist(q, bank.to(q.device))      # [N, K]
            patch_scores, _ = d2.min(dim=1)
            image_score = patch_scores.max().item()
            H = W = int(math.sqrt(q.shape[0]))
            heatmap = patch_scores.reshape(H, W).cpu().numpy()
            return image_score, heatmap
    return score

if __name__ == "__main__":
    main()
```

### 4.4 Coreset selection

```python
# powervision/anomaly/coreset.py
"""Greedy k-center sampling — preserves PatchCore's quality with manageable bank size."""
import numpy as np
from sklearn.random_projection import SparseRandomProjection

def greedy_coreset(features: np.ndarray, k: int, eps: int = 256, seed: int = 42) -> np.ndarray:
    """Project to `eps` dims, greedy farthest-point sampling to k indices."""
    rng = np.random.default_rng(seed)
    proj = SparseRandomProjection(n_components=eps, random_state=seed)
    F = proj.fit_transform(features)
    n = F.shape[0]
    if k >= n: return np.arange(n)
    selected = [int(rng.integers(0, n))]
    dist = np.linalg.norm(F - F[selected[0]], axis=1)
    for _ in range(1, k):
        next_i = int(dist.argmax())
        selected.append(next_i)
        new_d = np.linalg.norm(F - F[next_i], axis=1)
        dist = np.minimum(dist, new_d)
    return np.asarray(selected, dtype=np.int64)
```

### 4.5 Hyperparameters

| Hyperparameter | Default | Rationale |
|---|---|---|
| Backbone | `dinov2_vitl14` | Best open backbone for industrial AD as of 2026 |
| Layers tapped | 2, 3 (mid/late blocks) | Lower layers = textures, upper = semantics; mid-late mix is the empirical sweet spot |
| Image size | 256 | Multiples of patch size (14); large enough for component detail |
| Coreset K | 1024 | ~1% of typical patch pool; reduces inference cost without AUROC loss |
| Neighborhood `neigh` | 3 | Smooths spurious single-patch noise |
| Random projection dim | 256 | Coreset selection speed without quality drop |

### 4.6 Compute requirements

| Stage | Compute | Time per component |
|---|---|---|
| Feature extraction (1000 train crops) | RTX 3060 | ~90 s |
| Coreset selection (50k patches → K=1024) | CPU | ~20 s |
| Per-component total | — | ~2 min |
| 30 components total | RTX 3060 | ~1 h |

DINOv2 ViT-L/14 fits comfortably in 6 GB VRAM at batch 16.

---

## 5. Model 2 — EfficientAD

### 5.1 Architecture

Student-teacher with autoencoder regularization (from "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies").

```
Teacher (frozen, pretrained on ImageNet via knowledge distillation from WideResNet-101)
  └── Patch Description Network (PDN-small): 4 conv blocks, ~3 M params
  └── Outputs 384-d feature map at stride 4
Student (trained)
  └── Same PDN architecture, 2× the channels (~6 M params), random init
  └── Trained to match teacher features ON NORMAL DATA ONLY
Autoencoder (trained)
  └── 8-layer conv U-Net, ~7 M params
  └── Reconstructs the teacher's feature map → catches global anomalies that
       local student misses

At inference, two anomaly maps:
  - Local map:  L2(student - teacher)
  - Global map: L2(student - autoencoder)
  - Combined:  pixelwise max of normalized maps
```

### 5.2 Training script: `scripts/anomaly/train_efficientad.py`

```python
"""Train EfficientAD (student + autoencoder) on one component's normal data.

  python scripts/anomaly/train_efficientad.py \
    --component electrolytic_capacitor \
    --epochs 200 \
    --batch 8 \
    --device cuda:0
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader

from powervision.anomaly.dataset import NormalOnlyDataset, EvalDataset
from powervision.anomaly.pdn      import PDN_S, AutoEncoder, TeacherKD
from powervision.anomaly.calibrate import calibrate_threshold

ROOT = Path("dataset_v2/normals_only")
OUT  = Path("runs/anomaly/efficientad")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    comp_dir = ROOT / args.component
    out_dir  = OUT / args.component
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Frozen teacher (pretrained PDN-S; checkpoint distributed once at models/efficientad/teacher_pdn_s.pth)
    teacher = TeacherKD.load_pretrained("models/efficientad/teacher_pdn_s.pth").to(device).eval()
    student = PDN_S(channels=2 * teacher.out_channels).to(device).train()
    ae      = AutoEncoder(in_channels=teacher.out_channels).to(device).train()
    opt = optim.Adam(list(student.parameters()) + list(ae.parameters()),
                     lr=args.lr, weight_decay=1e-5)

    ds = NormalOnlyDataset(comp_dir / "train" / "good", size=256, augment=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(args.epochs):
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                t = teacher(x)                        # [B, C, H, W]
            s = student(x)
            # Split student channels: first C predict teacher (local); last C used for hard-negative penalty
            s_match, s_hard = s.chunk(2, dim=1)
            ae_out = ae(t.detach())

            l_local = F.mse_loss(s_match, t.detach())
            l_global = F.mse_loss(s_hard, ae_out.detach())
            l_ae    = F.mse_loss(ae_out, t.detach())
            loss = l_local + l_global + l_ae

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if (epoch + 1) % 20 == 0:
            print(f"[{args.component}] epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    student.eval(); ae.eval()
    torch.save(student.state_dict(), out_dir / "student.pt")
    torch.save(ae.state_dict(),      out_dir / "autoencoder.pt")
    torch.save(teacher.state_dict(), out_dir / "teacher.pt")

    # Persist per-feature normalization stats for inference parity
    norm_stats = _compute_pdn_normalization(student, teacher, ae,
                                            comp_dir / "train" / "good", device)
    (out_dir / "pdn_normalization.json").write_text(json.dumps(norm_stats, indent=2))

    # Threshold calibration
    eval_ds = EvalDataset(comp_dir / "val", size=256)
    threshold_info = calibrate_threshold(
        scorer=_make_efficientad_scorer(student, teacher, ae, norm_stats, device),
        eval_dataset=eval_ds,
        target_recall=0.95,
        device=device,
    )
    (out_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2))

    (out_dir / "metadata.json").write_text(json.dumps({
        "taxonomy_version": "2.0.0",
        "preproc_version": "2.0.0",
        "model": "efficientad_pdn_s",
        "train_count": len(ds),
        "epochs": args.epochs,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))

    # ONNX export of the student (the heavy inference path)
    torch.onnx.export(
        student, torch.randn(1, 3, 256, 256, device=device),
        str(out_dir / "student.onnx"),
        input_names=["image"], output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )

if __name__ == "__main__":
    main()
```

### 5.3 Hyperparameters

| Hyperparameter | Default | Rationale |
|---|---|---|
| Image size | 256 | Matches EfficientAD paper (256 / 384 / 512) |
| Batch | 8 | Fits 4 GB VRAM with PDN-S × 2 + AE |
| Epochs | 200 | Empirically converges by ~150; 200 with cosine LR is conservative |
| Optimizer | Adam | β = (0.9, 0.999), wd = 1e-5 |
| LR | 1e-4 | Constant; small dataset, no scheduler needed |
| Teacher | PDN-S (distilled WideResNet-101) | Provided pretrained; fine-tuning teacher is forbidden |
| Loss weights | local 1.0, global 1.0, ae 1.0 | Equal weighting works across all 30 components |

### 5.4 Compute requirements

| Stage | Compute | Time per component (500 train crops) |
|---|---|---|
| Training (200 epochs, batch 8) | RTX 3060 | ~22 min |
| Threshold calibration | RTX 3060 | ~30 s |
| ONNX export | CPU | ~5 s |
| 30 components total | RTX 4090 (faster, parallel) | ~6 h |

---

## 6. Threshold calibration

Anomaly scores are continuous; the Phase 6 decision unit needs a binary threshold per (model × component). Default: **recall ≥ 95% on val/defect** (high recall preferred since false negatives = missed faults). Operators can switch to F1-optimal or FPR-target via config.

```python
# powervision/anomaly/calibrate.py
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_recall_curve

def calibrate_threshold(scorer, eval_dataset, target_recall=0.95, device="cuda:0") -> dict:
    scores, labels = [], []
    for sample in eval_dataset:
        s, _ = scorer(sample["image"].to(device))
        scores.append(s); labels.append(int(sample["label"]))
    scores, labels = np.asarray(scores), np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        # Cannot calibrate without both classes
        return {"score_threshold": float(scores.mean()), "method": "fallback_mean",
                "warning": "val set missing one class"}

    precision, recall, thr = precision_recall_curve(labels, scores)
    # precision_recall_curve returns thr with len(thr) = len(precision) - 1
    idxs = np.where(recall[:-1] >= target_recall)[0]
    if len(idxs):
        idx = idxs[-1]   # highest threshold still meeting recall target
        chosen = thr[idx]
        return {
            "score_threshold": float(chosen),
            "method": f"recall_at_least_{target_recall}",
            "precision_at_threshold": float(precision[idx]),
            "recall_at_threshold": float(recall[idx]),
            "f1_at_threshold": float(2 * precision[idx] * recall[idx] /
                                     (precision[idx] + recall[idx] + 1e-9)),
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }
    # Fallback: F1-optimal
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    idx = int(np.nanargmax(f1))
    return {
        "score_threshold": float(thr[idx]),
        "method": "f1_optimal_fallback",
        "precision_at_threshold": float(precision[idx]),
        "recall_at_threshold": float(recall[idx]),
        "f1_at_threshold": float(f1[idx]),
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
```

---

## 7. Evaluation

### 7.1 Metrics

| Metric | Definition | Pass threshold |
|---|---|---|
| Image AUROC | Area under ROC for image-level normal vs defect on val/test | ≥ 0.92 |
| Image AUPR  | Area under PR curve, defect = positive | ≥ 0.85 |
| Pixel AUROC (when masks available) | Pixel-level normal vs defect using anomaly mask | ≥ 0.90 |
| PRO score (per-region overlap, used in EfficientAD paper) | Per-region overlap up to FPR 0.30 | ≥ 0.85 |
| Recall at chosen threshold | TP / (TP + FN) | ≥ 0.95 |
| FPR at chosen threshold | FP / (FP + TN) | ≤ 0.15 (warn ≤ 0.25) |

### 7.2 Eval script: `scripts/anomaly/evaluate.py`

```python
"""Evaluate PatchCore + EfficientAD on a component's val/test split.

Outputs auroc_image.json, auroc_pixel.json, pr_curve.png, heatmap_examples/.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from powervision.anomaly.dataset import EvalDataset
from powervision.anomaly.patchcore   import PatchCore
from powervision.anomaly.efficientad import EfficientAD

def evaluate_model(scorer, eval_ds, out_dir: Path):
    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    for sample in eval_ds:
        s, hmap = scorer(sample["image"])
        image_scores.append(s); image_labels.append(int(sample["label"]))
        if sample.get("mask") is not None:
            pixel_scores.append(hmap.flatten())
            pixel_labels.append((sample["mask"].flatten() > 0).astype(int))
    image_scores = np.asarray(image_scores); image_labels = np.asarray(image_labels)
    auroc_img = roc_auc_score(image_labels, image_scores)
    aupr_img  = average_precision_score(image_labels, image_scores)
    (out_dir / "auroc_image.json").write_text(json.dumps({
        "auroc": float(auroc_img), "aupr": float(aupr_img),
        "n_normal": int((image_labels == 0).sum()), "n_defect": int(image_labels.sum()),
    }, indent=2))

    if pixel_scores:
        ps = np.concatenate(pixel_scores); pl = np.concatenate(pixel_labels)
        auroc_pix = roc_auc_score(pl, ps)
        (out_dir / "auroc_pixel.json").write_text(json.dumps(
            {"auroc": float(auroc_pix)}, indent=2))

    # PR curve image
    p, r, _ = precision_recall_curve(image_labels, image_scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r, p); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"AUPR={aupr_img:.3f}")
    fig.tight_layout(); fig.savefig(out_dir / "pr_curve.png", dpi=120)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True)
    ap.add_argument("--model", choices=["patchcore", "efficientad"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    eval_ds = EvalDataset(Path("dataset_v2/normals_only") / args.component / args.split, size=256)
    out = Path(f"runs/anomaly/{args.model}/{args.component}/evaluation")
    out.mkdir(parents=True, exist_ok=True)

    if args.model == "patchcore":
        m = PatchCore.from_run_dir(Path(f"runs/anomaly/patchcore/{args.component}"))
    else:
        m = EfficientAD.from_run_dir(Path(f"runs/anomaly/efficientad/{args.component}"))
    m.to(args.device)
    evaluate_model(m.score, eval_ds, out)

if __name__ == "__main__":
    main()
```

---

## 8. Inference adapters

### 8.1 `powervision/anomaly/patchcore.py`

```python
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

from powervision.anomaly.dinov2_backbone import DINOv2Patches
from powervision.preproc.pipeline import serve_preprocess_classifier

class PatchCore:
    def __init__(self, bank: torch.Tensor, neigh: int, threshold: float,
                 backbone: DINOv2Patches | None = None, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.bank = bank.float().to(self.device)        # [K, D]
        self.neigh = neigh
        self.threshold = threshold
        self.backbone = backbone or DINOv2Patches(model="dinov2_vitl14", layers=(2, 3),
                                                  device=self.device).eval()

    @classmethod
    def from_run_dir(cls, run_dir: Path, device: str = "cuda:0") -> "PatchCore":
        bank_blob = torch.load(run_dir / "memory_bank.pt", map_location="cpu")
        thr = json.loads((run_dir / "threshold.json").read_text())["score_threshold"]
        return cls(bank_blob["features"], bank_blob["neigh"], thr, device=device)

    def to(self, device: str): self.device = torch.device(device); self.bank = self.bank.to(self.device); return self

    @torch.inference_mode()
    def score(self, crop_bgr_or_tensor) -> tuple[float, np.ndarray]:
        x = self._to_chw(crop_bgr_or_tensor).unsqueeze(0).to(self.device)
        feats = self.backbone(x)
        feats = self._neigh_pool(feats)
        q = feats.reshape(-1, feats.shape[-1])
        d2 = torch.cdist(q, self.bank)
        patch_scores, _ = d2.min(dim=1)
        image_score = patch_scores.max().item()
        H = W = int(math.sqrt(q.shape[0]))
        heatmap = patch_scores.reshape(H, W).cpu().numpy()
        return image_score, heatmap

    def is_anomalous(self, crop) -> bool:
        s, _ = self.score(crop); return s >= self.threshold

    @staticmethod
    def _to_chw(x):
        if isinstance(x, torch.Tensor): return x
        chw_np = serve_preprocess_classifier(x)
        return torch.from_numpy(chw_np)

    def _neigh_pool(self, feats):
        B, N, D = feats.shape; H = W = int(math.sqrt(N))
        t = feats.transpose(1, 2).reshape(B, D, H, W)
        t = F.avg_pool2d(t, 2*self.neigh+1, stride=1, padding=self.neigh)
        return t.reshape(B, D, -1).transpose(1, 2)
```

### 8.2 `powervision/anomaly/efficientad.py`

```python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

from powervision.anomaly.pdn import PDN_S, AutoEncoder, TeacherKD
from powervision.preproc.pipeline import serve_preprocess_classifier

class EfficientAD:
    def __init__(self, student, teacher, ae, threshold: float, norm_stats: dict,
                 device: str = "cuda:0"):
        self.device = torch.device(device)
        self.student, self.teacher, self.ae = student.to(self.device).eval(), teacher.to(self.device).eval(), ae.to(self.device).eval()
        self.threshold = threshold
        self.norm = norm_stats

    @classmethod
    def from_run_dir(cls, run_dir: Path, device: str = "cuda:0") -> "EfficientAD":
        teacher = TeacherKD.load_pretrained(run_dir / "teacher.pt")
        student = PDN_S(channels=2 * teacher.out_channels)
        student.load_state_dict(torch.load(run_dir / "student.pt", map_location="cpu"))
        ae = AutoEncoder(in_channels=teacher.out_channels)
        ae.load_state_dict(torch.load(run_dir / "autoencoder.pt", map_location="cpu"))
        thr = json.loads((run_dir / "threshold.json").read_text())["score_threshold"]
        norm = json.loads((run_dir / "pdn_normalization.json").read_text())
        return cls(student, teacher, ae, thr, norm, device=device)

    def to(self, device): self.device = torch.device(device); return self

    @torch.inference_mode()
    def score(self, crop_bgr_or_tensor) -> tuple[float, np.ndarray]:
        x = self._to_chw(crop_bgr_or_tensor).unsqueeze(0).to(self.device)
        t = self.teacher(x)
        s = self.student(x); s_match, s_hard = s.chunk(2, dim=1)
        ae_out = self.ae(t)

        m_local  = ((s_match - t) ** 2).mean(dim=1, keepdim=True)
        m_global = ((s_hard  - ae_out) ** 2).mean(dim=1, keepdim=True)

        # Per-feature normalization with cached stats for inference parity
        m_local  = (m_local  - self.norm["local_mean"])  / (self.norm["local_std"]  + 1e-6)
        m_global = (m_global - self.norm["global_mean"]) / (self.norm["global_std"] + 1e-6)

        combined = torch.maximum(m_local, m_global).squeeze(0).squeeze(0)
        image_score = combined.max().item()
        heatmap = F.interpolate(combined[None, None], size=(256, 256),
                                  mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        return image_score, heatmap

    def is_anomalous(self, crop) -> bool:
        s, _ = self.score(crop); return s >= self.threshold

    @staticmethod
    def _to_chw(x):
        if isinstance(x, torch.Tensor): return x
        return torch.from_numpy(serve_preprocess_classifier(x))
```

### 8.3 Fusion (used by Phase 6)

```python
# powervision/anomaly/fusion.py
"""Per-component OR-of-experts gate. The Phase 6 decision unit calls this
to convert two model scores into a single calibrated anomaly probability."""
from dataclasses import dataclass

@dataclass
class AnomalyVerdict:
    is_anomalous: bool
    fused_score: float
    patchcore_score: float
    efficientad_score: float
    patchcore_threshold: float
    efficientad_threshold: float
    voted_by: list[str]

def fuse(patchcore_score: float, patchcore_thr: float,
          efficientad_score: float, efficientad_thr: float,
          weights: tuple[float, float] = (0.6, 0.4)) -> AnomalyVerdict:
    """Two independent gates with weighted fused score; OR fires the verdict.

    Empirically: PatchCore is more accurate but slower → weight 0.6.
    EfficientAD is faster + catches global anomalies → weight 0.4.
    Either model crossing its own threshold is sufficient to flag (high-recall posture)."""
    # Normalize each score to [0, 1] using its own threshold (z = s / (2*thr))
    n_pc = min(1.0, patchcore_score    / (2 * patchcore_thr   if patchcore_thr   > 0 else 1.0))
    n_ad = min(1.0, efficientad_score / (2 * efficientad_thr if efficientad_thr > 0 else 1.0))
    fused = weights[0] * n_pc + weights[1] * n_ad
    flagged = []
    if patchcore_score    >= patchcore_thr:    flagged.append("patchcore")
    if efficientad_score >= efficientad_thr: flagged.append("efficientad")
    return AnomalyVerdict(
        is_anomalous=bool(flagged),
        fused_score=float(fused),
        patchcore_score=float(patchcore_score),
        efficientad_score=float(efficientad_score),
        patchcore_threshold=float(patchcore_thr),
        efficientad_threshold=float(efficientad_thr),
        voted_by=flagged,
    )
```

---

## 9. Integration contracts

### 9.1 Consumed by Phase 6 (Fault decision unit)

For each detection from Phase 3:

```python
from powervision.detect.component_detector import ComponentDetector
from powervision.anomaly.patchcore   import PatchCore
from powervision.anomaly.efficientad import EfficientAD
from powervision.anomaly.fusion      import fuse

det = ComponentDetector("runs/component_phase_b/weights/best.pt")
patchcore = {c: PatchCore.from_run_dir(f"runs/anomaly/patchcore/{c}") for c in components_with_anomaly_models}
effad     = {c: EfficientAD.from_run_dir(f"runs/anomaly/efficientad/{c}") for c in components_with_anomaly_models}

for component_det, crop in det.crop_components(img_bgr):
    name = component_det.component_name
    if name not in patchcore:    # component without an anomaly model
        continue
    pc_score, pc_heat = patchcore[name].score(crop)
    ad_score, ad_heat = effad[name].score(crop)
    verdict = fuse(pc_score, patchcore[name].threshold,
                   ad_score, effad[name].threshold)
    # → handed to Phase 6 alongside Phase 5 classifier output
```

### 9.2 Heatmap contract

`heatmap` returned by both models is `np.ndarray` of shape `(H, W)` floating-point. Phase 6 normalizes (`(x - x.min()) / (x.max() - x.min() + 1e-9)`) and overlays on the crop using a `jet` colormap for the LLM report (Phase 7).

### 9.3 Component without an anomaly model

If a component fails the data floor (§3.4), no PatchCore / EfficientAD weights exist. Phase 6 falls back to:

1. Phase 5 fault classifier on the crop (still runs).
2. Phase 6 rule engine (still runs).
3. The crop is logged with `anomaly_models_available=false` so Phase 8 prioritizes data collection for this component.

---

## 10. Code structure

```text
scripts/anomaly/
├── build_anomaly_dataset.py
├── train_patchcore.py
├── train_efficientad.py
├── evaluate.py
├── train_all_components.py        # convenience: loops over component_taxonomy
└── recalibrate_thresholds.py      # threshold-only refresh after data drift

powervision/anomaly/
├── __init__.py
├── dinov2_backbone.py             # ViT-L/14 wrapper with layer hooks
├── pdn.py                         # PDN_S, AutoEncoder, TeacherKD
├── coreset.py                     # greedy k-center
├── dataset.py                     # NormalOnlyDataset, EvalDataset
├── calibrate.py                   # threshold calibration
├── patchcore.py
├── efficientad.py
├── fusion.py
└── aug.py

runs/anomaly/
├── patchcore/<component>/
└── efficientad/<component>/

models/
├── dinov2/dinov2_vitl14.pth       # shared backbone
└── efficientad/teacher_pdn_s.pth  # pretrained teacher (download once)

dataset_v2/normals_only/<component>/{train,val,test}/...
```

### 10.1 Extended `dvc.yaml` stages (templated per component)

```yaml
stages:
  build_anomaly_dataset:
    cmd: python scripts/anomaly/build_anomaly_dataset.py
    deps:
      - dataset_v2/manifest.parquet
      - dataset_v2/labels/components
      - dataset_v2/labels/faults
      - dataset_v2/images/rgb
      - scripts/anomaly/build_anomaly_dataset.py
    outs:
      - dataset_v2/normals_only

  patchcore@${component}:
    cmd: python scripts/anomaly/train_patchcore.py --component ${component}
    deps:
      - dataset_v2/normals_only/${component}
      - models/dinov2/dinov2_vitl14.pth
      - powervision/anomaly
      - scripts/anomaly/train_patchcore.py
    outs:
      - runs/anomaly/patchcore/${component}/memory_bank.pt
      - runs/anomaly/patchcore/${component}/threshold.json
      - runs/anomaly/patchcore/${component}/metadata.json

  efficientad@${component}:
    cmd: python scripts/anomaly/train_efficientad.py --component ${component}
    deps:
      - dataset_v2/normals_only/${component}
      - models/efficientad/teacher_pdn_s.pth
      - powervision/anomaly
      - scripts/anomaly/train_efficientad.py
    outs:
      - runs/anomaly/efficientad/${component}/student.pt
      - runs/anomaly/efficientad/${component}/autoencoder.pt
      - runs/anomaly/efficientad/${component}/threshold.json
      - runs/anomaly/efficientad/${component}/metadata.json
      - runs/anomaly/efficientad/${component}/student.onnx

  evaluate_anomaly@${component}:
    cmd: |
      python scripts/anomaly/evaluate.py --component ${component} --model patchcore --split test
      python scripts/anomaly/evaluate.py --component ${component} --model efficientad --split test
    deps:
      - runs/anomaly/patchcore/${component}/memory_bank.pt
      - runs/anomaly/efficientad/${component}/student.pt
      - dataset_v2/normals_only/${component}/test
    metrics:
      - runs/anomaly/patchcore/${component}/evaluation/auroc_image.json
      - runs/anomaly/efficientad/${component}/evaluation/auroc_image.json
```

A wrapper `scripts/anomaly/train_all_components.py` reads `component_taxonomy.yaml` and invokes `dvc repro` for every component in parallel (configurable `--max-workers`).

---

## 11. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Insufficient normal data (< 50 crops) | Memory bank shape too small; AUROC < 0.85 | Skip component, route ROIs to Phase 5 + rule engine only; flag for Phase 8 data collection |
| Contamination (a defective image leaked into `train/good`) | High FPR at deploy time; cluster of low-scoring "good" images on val | Hard filter via `build_anomaly_dataset.py` (fault-bbox IoU check); periodic re-audit using PatchCore's own scores to find suspect "good" crops |
| DINOv2 release version drift | Memory bank features incompatible across DINOv2 minor releases | `metadata.json` pins `backbone` and `dinov2_revision`; serving asserts match before scoring |
| Heatmap noise (single-patch false positives) | Spurious anomaly flags | `_locally_aware_pool` with `neigh=3`; raise to 5 if persistent |
| Threshold drift after lighting change at customer site | FPR spikes after site update | Auto-recalibration job (Phase 8) runs `recalibrate_thresholds.py` every 7 days using newly captured normal data |
| EfficientAD overfit on small `train/good` | Train loss → 0 but AUROC drops | Reduce epochs to 100; lower learning rate; enforce minimum 100 train crops |
| PatchCore inference too slow (> 200 ms) | Latency SLA breach | Reduce coreset to 512; FP16 inference; or fall back to EfficientAD-only for the affected component |
| Anomaly model disagrees with classifier (Phase 5) | Score fusion logs both flagged but disagree | Phase 6 routes to human review; both models' scores are logged for HITL adjudication |
| Concept drift (new vendor's capacitor looks "anomalous") | Sudden burst of anomaly flags | Monitoring alerts (Phase 8) trigger HITL labeling; new normal crops added; retrain memory bank |
| Cross-component leakage (a `cable_run_battery` ROI accidentally hits the `cable_run_input` memory bank) | Spurious "anomaly" because models are per-component | Phase 3's `Detection.component_id` is authoritative; never score with a different component's bank |
| Memory bank OOM at inference (many components × 1024 × 1024-d) | OOM at startup | Each bank is ~2 MB in FP16; 30 components ~60 MB — fits easily. Larger configs (K=4096): lazy-load per component crop. |

---

## 12. Phase 4 exit checklist

- [ ] `dataset_v2/normals_only/<component>/` populated for ≥ 70% of component classes, all passing data floors (§3.4).
- [ ] DINOv2 ViT-L/14 weights pinned at `models/dinov2/dinov2_vitl14.pth`; `metadata.json` records revision hash.
- [ ] EfficientAD teacher pinned at `models/efficientad/teacher_pdn_s.pth`.
- [ ] PatchCore memory banks built for every eligible component.
- [ ] EfficientAD student/AE trained for every eligible component; ONNX export passes parity.
- [ ] Per-model evaluation: image AUROC ≥ 0.92 on ≥ 80% of components; flagged components escalated to Phase 8.
- [ ] Threshold calibration done with `target_recall=0.95` (or operator-chosen alternative).
- [ ] Fusion adapter (`fuse()`) unit-tested with synthetic score pairs.
- [ ] MLflow registry entries created for every component's PatchCore + EfficientAD pair.
- [ ] Legacy demo (`api/main.py`, `api/streamlit_app.py`) inference unchanged.

Phase 4 is **frozen** when all boxes are checked. Phase 5 begins.
