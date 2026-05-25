# Phase 5 — Fault Classifier Training (100+ Classes)

> **Scope:** Backend + model training. Trains the supervised, fine-grained **fault classifier** that consumes Phase 3 component crops and assigns a specific fault label from the 120-class fault taxonomy (Phase 1). It complements — does **not** replace — the anomaly detectors (Phase 4): the classifier provides semantic labels when the fault type is known; PatchCore/EfficientAD cover novel faults. Together they feed the Phase 6 fault decision unit.

---

## 1. Phase objective

The Phase 3 component detector localizes where to look. The Phase 4 anomaly models say "something is wrong here." This phase says **"what** is wrong here": `capacitor_bulge`, `battery_terminal_corrosion`, `fuse_blown`, `ground_strap_missing`, etc.

| Aspect | Value |
|---|---|
| Model family | **EfficientNet-B4** (default) with **ViT-B/16** as upgrade path |
| Output | 120-way classification + `no_fault` class (id `120`, internal); softmax probability distribution |
| Input | Component ROI crop (BGR uint8, variable size; resized to 384²) |
| Training mode | Supervised on labeled fault crops + normal crops |
| Loss | Class-balanced cross-entropy with label smoothing + focal modulation for rare classes |
| Latency target | ≤ 25 ms / crop on RTX 3060 (FP16) |
| Eval target | Top-1 ≥ 0.85 macro-averaged across 120 classes; per-class recall ≥ 0.70 |

Deliverables:

| Deliverable | Artifact |
|---|---|
| Classifier crops dataset | `dataset_v2/crops/{train,val,test}/<fault_name>/*.jpg` (Phase 2 populates) |
| `no_fault` crops dataset | `dataset_v2/crops/{train,val,test}/no_fault/*.jpg` |
| Class index mapping | `runs/classifier/class_index.json` |
| Training script | `scripts/classifier/train_classifier.py` |
| Two-phase training configs | `configs/classifier/{phase_a.yaml, phase_b.yaml}` |
| Evaluation pack | `runs/classifier/<run_name>/evaluation/{confusion_matrix.png, per_class_metrics.csv, calibration_curve.png, evaluation_report.json}` |
| Inference adapter | `powervision/classify/fault_classifier.py` |
| Temperature-scaling calibration | `runs/classifier/<run_name>/temperature.json` |
| ONNX / TensorRT export | `runs/classifier/<run_name>/best.onnx`, `best.engine` |
| Per-component head routing config | `configs/classifier/component_fault_routing.yaml` |
| MLflow registry | `powervision-fault-classifier/v<N>` |

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Format |
|---|---|---|
| Fault crops | `dataset_v2/crops/<split>/<fault_name>/*.jpg` | JPEG, variable size ≥ 96² |
| Normal crops (`no_fault` class) | `dataset_v2/crops/<split>/no_fault/*.jpg` | JPEG |
| Component → applicable-fault map | `taxonomy/component_taxonomy.yaml` × `taxonomy/fault_taxonomy.yaml` | YAML |
| Fault taxonomy | `taxonomy/fault_taxonomy.yaml` | YAML |
| UPS-type fault whitelist | `taxonomy/ups_system_types.yaml` | YAML |
| Pretrained backbones | timm hub (`tf_efficientnetv2_b4`, `vit_base_patch16_224`) | PyTorch |

### 2.2 Outputs

```text
runs/classifier/
└── eff_b4_v1/
    ├── weights/{best.pt, last.pt, best.onnx, best.engine}
    ├── config_resolved.yaml
    ├── class_index.json
    ├── temperature.json
    ├── training_log.csv
    ├── tensorboard/
    └── evaluation/
        ├── per_class_metrics.csv
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── top_k_accuracy.json
        ├── calibration_curve.png
        ├── reliability_diagram.png
        ├── error_examples/<fault_name>/
        ├── slice_metrics.json
        └── evaluation_report.json
```

### 2.3 Format contracts

- **`class_index.json`**: ordered list of `(class_index, fault_id, fault_name)` triples. The classifier's logit at index `i` corresponds to `fault_id = class_index[i].fault_id`. **Index order is locked when training starts** and persists across re-trainings; new fault IDs are appended.
- **`temperature.json`**: `{"temperature": 1.28, "method": "logit_temperature", "ece_before": 0.094, "ece_after": 0.023}`. Applied at serve time: `probs = softmax(logits / temperature)`.
- **`component_fault_routing.yaml`**: defines, per component, which fault classes are physically possible. Inference uses this to mask out impossible classes from the softmax (e.g., `cable_run_input` can never produce `battery_terminal_corrosion`).

---

## 3. End-to-end dataset creation pipeline

### 3.1 Crop generation (recap from Phase 2 §7.2)

```bash
python scripts/preproc/build_classifier_crops.py \
  --manifest   dataset_v2/manifest.parquet \
  --labels     dataset_v2/labels/faults \
  --images     dataset_v2/images/rgb \
  --out        dataset_v2/crops \
  --pad-ratio  0.15 \
  --min-side-px 96
```

For Phase 5, additionally generate **normal crops** to learn a `no_fault` class:

```bash
python scripts/classifier/build_no_fault_crops.py \
  --manifest dataset_v2/manifest.parquet \
  --labels   dataset_v2/labels/components \
  --images   dataset_v2/images/rgb \
  --out      dataset_v2/crops \
  --per-component 200          # sample 200 normal crops per component class
  --pad-ratio 0.15
```

This script:

1. Joins on `manifest.parquet` where `is_normal == True`.
2. For each component bbox in such images, crops with `pad_ratio`.
3. Stratifies the output across `component_id` so the `no_fault` class doesn't get dominated by `screw_terminal` (which has many instances per image).
4. Writes to `dataset_v2/crops/<split>/no_fault/<image_id>__<idx>.jpg`.

### 3.2 Crop dataset distribution (target)

| Fault group | # crops train | # crops val | # crops test |
|---|---:|---:|---:|
| Demo legacy faults (ids 0–8) | 5 000–15 000 | 1 000–3 000 | 1 000–3 000 |
| Rectifier (9–24) | 3 000–8 000 | 600–1 600 | 600–1 600 |
| Inverter (25–44) | 3 500–9 000 | 700–1 800 | 700–1 800 |
| Battery (45–74) | 6 000–18 000 | 1 200–3 600 | 1 200–3 600 |
| Bypass (75–84) | 800–2 000 | 160–400 | 160–400 |
| Control PCB (85–99) | 1 500–4 000 | 300–800 | 300–800 |
| Cooling (100–109) | 1 200–3 000 | 240–600 | 240–600 |
| Enclosure (110–119) | 800–2 000 | 160–400 | 160–400 |
| `no_fault` | 20 000 (capped) | 4 000 | 4 000 |

The total target: **~50 000 train crops** across 121 classes (120 faults + `no_fault`). The dataset is **severely imbalanced** — minority class fixes are non-negotiable (§3.5).

### 3.3 Class balancing strategy

Three layers, applied together:

| Layer | Mechanism | When |
|---|---|---|
| 1. Resampling | `WeightedRandomSampler` with weight `1 / sqrt(class_count)` | Always |
| 2. Loss weighting | Class-balanced cross-entropy ("effective number of samples", Cui et al. 2019) | Always |
| 3. Loss shaping | Focal cross-entropy modulation (`γ = 2`) on top of class weights | Always |
| 4. Targeted augmentation | More aug per crop for classes < 100 train crops | Always |

Effective sample weighting:

```python
# powervision/classify/sampler.py
import numpy as np
def class_balanced_weights(class_counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Cui et al. 2019: w_c = (1 - beta) / (1 - beta**n_c)."""
    eff_num = 1.0 - np.power(beta, class_counts)
    w = (1.0 - beta) / np.where(eff_num > 0, eff_num, 1.0)
    return w / w.sum() * len(class_counts)
```

### 3.4 Rare-class handling

For any class with `train_count < 30`:

| Action | Setting |
|---|---|
| Reject from `test` (placed in `excluded.txt`) | Yes |
| Generate 8× Albumentations augmentations per crop | Yes |
| Boost class weight to 5× ceiling | Yes |
| Flag for active-learning collection (Phase 8) | Yes |
| Fall back to anomaly detection (Phase 4) at serve time | Yes — Phase 6 OR's the verdict |

A class with `train_count < 10` is **excluded from training** (logged to `runs/classifier/excluded_classes.json`); it relies on Phase 4 only.

### 3.5 Targeted augmentation

```python
# powervision/classify/aug.py
import albumentations as A
import cv2

CLASSIFIER_TRAIN = A.Compose([
    A.LongestMaxSize(max_size=420),                    # ensure crop > target before random crop
    A.PadIfNeeded(420, 420, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
    A.RandomResizedCrop(height=384, width=384, scale=(0.75, 1.0), ratio=(0.85, 1.18), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 25.0)),
    ], p=0.25),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4, p=0.15),
    A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.20),
])

CLASSIFIER_TRAIN_RARE = A.Compose([
    A.LongestMaxSize(max_size=420),
    A.PadIfNeeded(420, 420, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
    A.RandomResizedCrop(384, 384, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(0.30, 0.30, p=0.9),
    A.HueSaturationValue(12, 30, 20, p=0.7),
    A.Affine(rotate=(-25, 25), translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), p=0.7),
    A.Perspective(scale=(0.03, 0.08), p=0.3),
    A.OneOf([A.MotionBlur(5), A.GaussianBlur(5), A.GaussNoise(var_limit=(10, 40))], p=0.4),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=0.4),
    A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
])

CLASSIFIER_EVAL = A.Compose([
    A.LongestMaxSize(max_size=420),
    A.PadIfNeeded(420, 420, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
    A.CenterCrop(384, 384),
])
```

`augmentation_profiles.yaml` (Phase 2) names which pipeline a class uses; the dataset class (`scripts/classifier/data.py`) routes by `class_index`.

### 3.6 Train / val / test splits

Inherited from Phase 1 §9. Phase 5 reads `dataset_v2/splits/<split>.txt` and crops are pre-partitioned by `build_classifier_crops.py` according to the source image's split. **Same image's crops never appear in two splits.**

### 3.7 Dataset versioning

`dataset_v2/crops/` is DVC-tracked. Crop-set version is **derived** from (`dataset_v2/manifest.parquet hash` + `build_classifier_crops.py git hash`), recorded in `dataset_v2/crops/_version.json`. Classifier training pins this version in `class_index.json`'s metadata.

---

## 4. Model training pipeline

### 4.1 Model selection rationale

| Model | Params | Acc on ImageNet-1k | Latency RTX 3060 FP16 (384²) | Notes |
|---|---|---|---|---|
| ResNet-50 | 25 M | 76.1 | 8 ms | Classic baseline; weaker fine-grained recall |
| EfficientNet-B0 | 5.3 M | 77.7 | 5 ms | Edge variant |
| **EfficientNetV2-B4** | 19 M | 83.1 | 18 ms | **Default** — strong fine-grained perf, manageable footprint |
| ConvNeXt-Tiny | 28 M | 82.1 | 14 ms | Good upgrade path |
| ViT-B/16 (DeiT III) | 86 M | 85.7 | 22 ms | Upgrade if EffNet plateaus |
| Swin-T | 28 M | 81.3 | 16 ms | Alternative ViT |

**Default: EfficientNetV2-B4** (timm: `tf_efficientnetv2_b4.in21k_ft_in1k`). Rationale:

1. Convolutional inductive bias gives it a head start on small dataset sizes (50k crops).
2. Pretrained on ImageNet-21k → fine-tuned on ImageNet-1k provides strong transfer to industrial textures.
3. Native 384² training is a good match for ROI crop sizes (most fault evidence is texture-level).

**Upgrade trigger:** if Phase B EfficientNet val top-1 plateaus < 0.80 macro, switch to ViT-B/16. Same training script handles both via `--model` flag.

### 4.2 Two-phase fine-tuning (parallels the YOLO pattern)

| Phase | What | Why |
|---|---|---|
| **A** | Freeze backbone, train classifier head + `BatchNorm` stats for 10 epochs | Adapt head to 121 classes without disturbing pretrained features |
| **B** | Unfreeze all, full fine-tune with discriminative LR (head LR > backbone LR) | Refine backbone for industrial textures |

The pattern matches the demo's `train_phase_a.py` → `train_phase_b.py` design.

### 4.3 Hyperparameter configurations

#### `configs/classifier/phase_a.yaml`

```yaml
model:
  name: tf_efficientnetv2_b4.in21k_ft_in1k
  num_classes: 121          # 120 faults + no_fault
  drop_path_rate: 0.2
  pretrained: true

data:
  crops_root: dataset_v2/crops
  num_workers: 8
  image_size: 384
  augment_profile: standard  # CLASSIFIER_TRAIN
  rare_aug_profile: rare     # CLASSIFIER_TRAIN_RARE
  rare_class_threshold: 100  # classes with train_count < 100 use rare profile

loader:
  batch_size: 64
  use_weighted_sampler: true
  sampler_beta: 0.9999       # for class-balanced weights

optim:
  optimizer: adamw
  lr_head: 1.0e-3
  lr_backbone: 0.0           # frozen
  weight_decay: 0.05
  betas: [0.9, 0.999]

schedule:
  scheduler: cosine
  warmup_epochs: 1
  epochs: 10
  min_lr_ratio: 0.01

loss:
  type: focal_class_balanced
  gamma: 2.0
  label_smoothing: 0.05
  class_balanced_beta: 0.9999

freeze:
  backbone: true
  batchnorm: false   # let BN stats adapt to industrial color distribution
  head: false

augment:
  mixup: 0.0
  cutmix: 0.0

train:
  amp: true
  grad_clip: 1.0
  log_every: 50
  eval_every_epoch: 1
  save_best: macro_f1
  seed: 42
  deterministic: true

output:
  project: runs/classifier
  name: eff_b4_phase_a
```

#### `configs/classifier/phase_b.yaml`

```yaml
model:
  name: tf_efficientnetv2_b4.in21k_ft_in1k
  num_classes: 121
  drop_path_rate: 0.3
  pretrained: false                     # weights loaded from Phase A best.pt
  init_from: runs/classifier/eff_b4_phase_a/weights/best.pt

data:
  crops_root: dataset_v2/crops
  num_workers: 8
  image_size: 384
  augment_profile: standard
  rare_aug_profile: rare
  rare_class_threshold: 100

loader:
  batch_size: 32
  use_weighted_sampler: true
  sampler_beta: 0.9999

optim:
  optimizer: adamw
  lr_head:     3.0e-4
  lr_backbone: 3.0e-5                    # 10x lower than head
  weight_decay: 0.05
  betas: [0.9, 0.999]

schedule:
  scheduler: cosine
  warmup_epochs: 2
  epochs: 40
  min_lr_ratio: 0.01

loss:
  type: focal_class_balanced
  gamma: 2.0
  label_smoothing: 0.1
  class_balanced_beta: 0.9999

freeze:
  backbone: false
  batchnorm: false
  head: false

augment:
  mixup: 0.2
  cutmix: 0.2
  mix_prob: 0.5

train:
  amp: true
  grad_clip: 1.0
  early_stop:
    monitor: macro_f1
    patience: 8
    min_delta: 0.002
  save_best: macro_f1
  seed: 42
  deterministic: true

output:
  project: runs/classifier
  name: eff_b4_phase_b
```

### 4.4 Training script: `scripts/classifier/train_classifier.py`

```python
"""Two-phase fault classifier training. Supports A (frozen) and B (full).

Usage:
  python scripts/classifier/train_classifier.py --config configs/classifier/phase_a.yaml
  python scripts/classifier/train_classifier.py --config configs/classifier/phase_b.yaml
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from datetime import datetime
import numpy as np, yaml, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
from timm.scheduler import CosineLRScheduler

from powervision.classify.data import CropDataset, build_class_index
from powervision.classify.sampler import class_balanced_weights, build_weighted_sampler
from powervision.classify.loss import FocalClassBalancedLoss
from powervision.classify.mixup import MixupCutmix
from powervision.classify.eval import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output"]["project"]) / cfg["output"]["name"]
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    (out_dir).joinpath("config_resolved.yaml").write_text(yaml.safe_dump(cfg))

    device = torch.device(args.device)
    torch.manual_seed(cfg["train"]["seed"]); np.random.seed(cfg["train"]["seed"])
    torch.backends.cudnn.deterministic = cfg["train"]["deterministic"]

    # ---- Dataset ----
    class_index, ignored = build_class_index(
        crops_root=Path(cfg["data"]["crops_root"]),
        taxonomy_path=Path("taxonomy/fault_taxonomy.yaml"),
        min_train_count=10,   # < 10: ignored entirely (logged)
    )
    (out_dir / "class_index.json").write_text(json.dumps(class_index, indent=2))
    (out_dir / "ignored_classes.json").write_text(json.dumps(ignored, indent=2))
    num_classes = len(class_index)

    train_ds = CropDataset(cfg["data"], split="train", class_index=class_index,
                            rare_class_threshold=cfg["data"]["rare_class_threshold"])
    val_ds   = CropDataset(cfg["data"], split="val",   class_index=class_index)
    counts = train_ds.class_counts()

    sampler = (build_weighted_sampler(train_ds.labels, beta=cfg["loader"]["sampler_beta"])
                if cfg["loader"]["use_weighted_sampler"] else None)
    train_dl = DataLoader(train_ds, batch_size=cfg["loader"]["batch_size"],
                          sampler=sampler, shuffle=sampler is None,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg["loader"]["batch_size"],
                          shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # ---- Model ----
    model = timm.create_model(cfg["model"]["name"],
                              num_classes=num_classes,
                              pretrained=cfg["model"]["pretrained"],
                              drop_path_rate=cfg["model"]["drop_path_rate"]).to(device)
    if init_from := cfg["model"].get("init_from"):
        sd = torch.load(init_from, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=False)

    if cfg["freeze"]["backbone"]:
        for n, p in model.named_parameters():
            if not n.startswith("classifier") and not n.startswith("head"):
                p.requires_grad = False

    # ---- Loss ----
    cb_w = class_balanced_weights(counts, beta=cfg["loss"]["class_balanced_beta"])
    loss_fn = FocalClassBalancedLoss(
        class_weights=torch.tensor(cb_w, dtype=torch.float32, device=device),
        gamma=cfg["loss"]["gamma"],
        label_smoothing=cfg["loss"]["label_smoothing"],
    )

    # ---- Optimizer with discriminative LR ----
    head_keys = ["classifier", "head", "fc"]
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head_params if any(k in n for k in head_keys) else bb_params).append(p)
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": cfg["optim"]["lr_head"]},
        {"params": bb_params,   "lr": cfg["optim"]["lr_backbone"]},
    ], weight_decay=cfg["optim"]["weight_decay"], betas=tuple(cfg["optim"]["betas"]))

    sched = CosineLRScheduler(
        opt,
        t_initial=cfg["schedule"]["epochs"],
        warmup_t=cfg["schedule"]["warmup_epochs"],
        warmup_lr_init=cfg["optim"]["lr_head"] * 0.01,
        lr_min=min(cfg["optim"]["lr_head"], cfg["optim"]["lr_backbone"]) * cfg["schedule"]["min_lr_ratio"],
    )

    mixer = MixupCutmix(num_classes=num_classes,
                         mixup_alpha=cfg["augment"].get("mixup", 0.0),
                         cutmix_alpha=cfg["augment"].get("cutmix", 0.0),
                         prob=cfg["augment"].get("mix_prob", 0.5))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    tb = SummaryWriter(out_dir / "tensorboard")

    best_score = -1.0; patience_left = cfg["train"].get("early_stop", {}).get("patience", 10**6)
    for epoch in range(cfg["schedule"]["epochs"]):
        model.train()
        for step, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            x, y_soft = mixer(x, y)
            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                logits = model(x)
                loss = loss_fn(logits, y_soft)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt); scaler.update()
            if step % cfg["train"]["log_every"] == 0:
                tb.add_scalar("train/loss", loss.item(), epoch * len(train_dl) + step)
        sched.step(epoch + 1)

        # ---- Eval ----
        metrics = evaluate(model, val_dl, device, num_classes)
        for k, v in metrics.items():
            tb.add_scalar(f"val/{k}", v, epoch)
        score = metrics["macro_f1"]

        ckpt = dict(model=model.state_dict(), opt=opt.state_dict(),
                    epoch=epoch, metrics=metrics, cfg=cfg)
        torch.save(ckpt, out_dir / "weights" / "last.pt")
        if score > best_score:
            best_score = score; patience_left = cfg["train"].get("early_stop", {}).get("patience", 10**6)
            torch.save(ckpt, out_dir / "weights" / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best macro_f1={best_score:.4f}")
                break

    # ---- Persist metadata ----
    (out_dir / "metadata.json").write_text(json.dumps({
        "taxonomy_version": "2.0.0",
        "preproc_version": "2.0.0",
        "model": cfg["model"]["name"],
        "num_classes": num_classes,
        "best_macro_f1": float(best_score),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))

if __name__ == "__main__":
    main()
```

### 4.5 Loss function

```python
# powervision/classify/loss.py
import torch, torch.nn.functional as F

class FocalClassBalancedLoss(torch.nn.Module):
    """Cross-entropy with:
      • class-balanced weights (Cui 2019)
      • focal modulation (γ)
      • label smoothing ε
      • supports soft targets (mixup/cutmix one-hot mixtures).
    """
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("w", class_weights)
        self.gamma = gamma; self.eps = label_smoothing

    def forward(self, logits: torch.Tensor, targets) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        if targets.dim() == 1:                            # hard labels
            tgt = F.one_hot(targets, logits.shape[-1]).float()
        else:                                              # soft labels (mixup)
            tgt = targets.float()
        # Label smoothing
        if self.eps > 0:
            n = logits.shape[-1]
            tgt = tgt * (1.0 - self.eps) + self.eps / n
        # Focal modulation on confidence of the *true* class
        p = log_p.exp()
        focal = ((1.0 - (tgt * p).sum(dim=-1, keepdim=True)) ** self.gamma)
        # Weighted CE
        loss_per = -(self.w * tgt * log_p).sum(dim=-1)
        return (focal.squeeze(-1) * loss_per).mean()
```

### 4.6 Mixup / CutMix

```python
# powervision/classify/mixup.py
import torch, numpy as np
class MixupCutmix:
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5):
        self.nc = num_classes; self.m = mixup_alpha; self.c = cutmix_alpha; self.p = prob

    def __call__(self, x, y):
        if np.random.rand() > self.p or (self.m == 0 and self.c == 0):
            return x, torch.nn.functional.one_hot(y, self.nc).float()
        idx = torch.randperm(x.size(0), device=x.device)
        if np.random.rand() < 0.5 and self.m > 0:
            lam = float(np.random.beta(self.m, self.m))
            x = lam * x + (1 - lam) * x[idx]
        else:
            lam = float(np.random.beta(self.c, self.c))
            H, W = x.shape[-2:]
            rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
            cy, cx = np.random.randint(H), np.random.randint(W)
            y1, y2 = max(0, cy - rh//2), min(H, cy + rh//2)
            x1, x2 = max(0, cx - rw//2), min(W, cx + rw//2)
            x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)
        y1h = torch.nn.functional.one_hot(y,      self.nc).float()
        y2h = torch.nn.functional.one_hot(y[idx], self.nc).float()
        return x, lam * y1h + (1 - lam) * y2h
```

### 4.7 GPU / compute requirements

| Stage | Compute | Time |
|---|---|---|
| Phase A (10 epochs, 50k crops) | RTX 4070 (12 GB), batch 64 | ~45 min |
| Phase B (40 epochs, 50k crops) | RTX 4090 (24 GB), batch 32 | ~5–6 h |
| Fallback | RTX 3060 (8 GB), batch 16 (Phase A), batch 8 (Phase B) | ~14–18 h Phase B |
| Distributed | 4× A100, DDP | ~75 min Phase B |

OOM fallback:

| Backbone | Image size | Min VRAM (batch 8) |
|---|---|---|
| EfficientNetV2-B4 | 384 | ~6 GB |
| EfficientNetV2-B4 | 320 | ~5 GB |
| ViT-B/16 | 224 | ~7 GB |

### 4.8 Checkpoint management

| File | When | Purpose |
|---|---|---|
| `weights/last.pt` | Every epoch | Resume |
| `weights/best.pt` | When val macro-F1 improves | Serving + Phase B init |
| `weights/epoch{N}.pt` | Every `save_period` (config) | Time-travel |
| `class_index.json` | At Phase A start | Fixed for life of model series |
| `temperature.json` | After Phase B + calibration script | Serving |
| `metadata.json` | After every phase | Versioning |

Resume:

```bash
python scripts/classifier/train_classifier.py \
  --config configs/classifier/phase_b.yaml --resume
```

---

## 5. Temperature-scaling calibration

After Phase B, the raw softmax is over-confident (typical of deep classifiers). Run logit-temperature calibration on val:

```python
# scripts/classifier/calibrate_temperature.py
"""Optimize temperature T to minimize NLL on val (Guo et al. 2017)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from powervision.classify.data import CropDataset

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = nn.Parameter(torch.ones(1) * 1.3)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    ckpt = torch.load(run / "weights" / "best.pt", map_location="cpu")
    cfg = ckpt["cfg"]; ci = json.loads((run / "class_index.json").read_text())
    import timm
    model = timm.create_model(cfg["model"]["name"], num_classes=len(ci), pretrained=False).cuda().eval()
    model.load_state_dict(ckpt["model"])

    val_ds = CropDataset(cfg["data"], split="val", class_index=ci)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    all_logits, all_y = [], []
    with torch.inference_mode():
        for x, y in val_dl:
            all_logits.append(model(x.cuda()).cpu()); all_y.append(y)
    logits = torch.cat(all_logits); y = torch.cat(all_y)

    T = fit_temperature(logits, y)
    # ECE
    from powervision.classify.eval import expected_calibration_error
    ece_before = expected_calibration_error(F.softmax(logits, dim=-1).numpy(), y.numpy())
    ece_after  = expected_calibration_error(F.softmax(logits / T, dim=-1).numpy(), y.numpy())
    (run / "temperature.json").write_text(json.dumps({
        "temperature": T, "method": "logit_temperature",
        "ece_before": float(ece_before), "ece_after": float(ece_after),
    }, indent=2))
    print(f"Temperature={T:.3f}, ECE {ece_before:.4f} → {ece_after:.4f}")

if __name__ == "__main__":
    main()
```

Apply at serve time:

```python
probs = torch.softmax(model(x) / T, dim=-1)
```

Calibration matters: Phase 6's confidence gate uses these probabilities as-is. Without temperature scaling, every fault crosses the gate at p > 0.95 and no escalation ever happens.

---

## 6. Evaluation

### 6.1 Metrics

| Metric | Definition | PASS | WARN | FAIL |
|---|---|---|---|---|
| Top-1 accuracy (overall) | argmax matches label | ≥ 0.88 | 0.80–0.88 | < 0.80 |
| Macro-F1 | unweighted mean F1 across 121 classes | ≥ 0.80 | 0.70–0.80 | < 0.70 |
| Macro-recall | unweighted mean recall | ≥ 0.78 | 0.68–0.78 | < 0.68 |
| Per-class recall (each of 120 faults) | TP/(TP+FN) | each ≥ 0.70 | each ≥ 0.55 | any < 0.55 |
| Top-3 accuracy | true label in top-3 logits | ≥ 0.96 | 0.92–0.96 | < 0.92 |
| Expected calibration error (ECE) post temp scaling | binned | ≤ 0.05 | 0.05–0.10 | > 0.10 |
| Critical-fault recall (severity = critical) | recall on severity = critical faults only | ≥ 0.90 | 0.80–0.90 | < 0.80 |
| Inference latency p95 (RTX 3060 FP16, batch 8) | per-crop ms | ≤ 25 | 25–40 | > 40 |

The **critical-fault recall** check is the hardest gate; missing a critical fault is operationally worse than mis-labeling a low-severity one.

### 6.2 Evaluation script

```python
# scripts/classifier/evaluate.py
"""Full eval pack on test split. Writes per_class_metrics.csv, confusion matrix,
top-k accuracy, calibration curve, error examples, slice metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import (confusion_matrix, classification_report,
                              top_k_accuracy_score, f1_score, recall_score)
import matplotlib.pyplot as plt

from powervision.classify.data import CropDataset
from powervision.classify.eval import expected_calibration_error
import timm, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run = Path(args.run_dir); out = run / "evaluation"; out.mkdir(exist_ok=True, parents=True)

    ckpt = torch.load(run / "weights" / "best.pt", map_location="cpu")
    cfg = ckpt["cfg"]; ci = json.loads((run / "class_index.json").read_text())
    T = json.loads((run / "temperature.json").read_text())["temperature"]

    model = timm.create_model(cfg["model"]["name"], num_classes=len(ci), pretrained=False).cuda().eval()
    model.load_state_dict(ckpt["model"])

    ds = CropDataset(cfg["data"], split=args.split, class_index=ci)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    logits, labels = [], []
    with torch.inference_mode():
        for x, y in dl:
            logits.append((model(x.cuda()) / T).cpu()); labels.append(y)
    logits = torch.cat(logits); labels = torch.cat(labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    rep = classification_report(labels, preds, target_names=[c["name"] for c in ci],
                                  output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).transpose().reset_index().rename(columns={"index": "name"})
    df.to_csv(out / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(labels, preds, labels=list(range(len(ci))))
    np.save(out / "confusion_matrix.npy", cm)
    _plot_cm(cm, [c["name"] for c in ci], out / "confusion_matrix.png",
              normalize=False)
    _plot_cm(cm, [c["name"] for c in ci], out / "confusion_matrix_normalized.png",
              normalize=True)

    tops = {f"top_{k}_acc": float(top_k_accuracy_score(labels, probs, k=k, labels=list(range(len(ci)))))
            for k in (1, 3, 5)}
    (out / "top_k_accuracy.json").write_text(json.dumps(tops, indent=2))
    ece = expected_calibration_error(probs, labels)
    macro_f1     = float(f1_score(labels, preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(labels, preds, average="macro", zero_division=0))

    # Verdict
    worst_recall = df[df["name"].isin([c["name"] for c in ci])]["recall"].min()
    if tops["top_1_acc"] >= 0.88 and macro_f1 >= 0.80 and worst_recall >= 0.70:
        verdict = "PASS"
    elif tops["top_1_acc"] >= 0.80 and macro_f1 >= 0.70 and worst_recall >= 0.55:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    (out / "evaluation_report.json").write_text(json.dumps({
        **tops, "macro_f1": macro_f1, "macro_recall": macro_recall,
        "ece": float(ece), "verdict": verdict, "worst_class_recall": float(worst_recall),
    }, indent=2))

def _plot_cm(cm, names, path, normalize=False):
    cm = cm.astype(np.float32)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, aspect="auto")
    fig.colorbar(im); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

if __name__ == "__main__":
    main()
```

### 6.3 Slice analysis

`evaluate.py` also writes `slice_metrics.json` with per-`ups_type_id` and per-`severity` macro-F1 to detect cohort regressions:

```json
{
  "by_ups_type": {
    "ups_type_001": {"macro_f1": 0.84, "n": 6210},
    "ups_type_002": {"macro_f1": 0.78, "n": 1145},
    "ups_type_003": {"macro_f1": 0.71, "n": 240}
  },
  "by_severity": {
    "info": {"macro_f1": 0.83},
    "low":  {"macro_f1": 0.81},
    "medium": {"macro_f1": 0.80},
    "high": {"macro_f1": 0.85},
    "critical": {"macro_f1": 0.91, "recall": 0.94}
  }
}
```

### 6.4 Error analysis output

`runs/.../evaluation/error_examples/<true_fault_name>/<predicted_fault_name>/` collects up to 25 misclassified crops per (true, predicted) pair for SME inspection. Top-3 confused pairs are listed in `evaluation_report.json` under `most_confused_pairs`.

---

## 7. Inference adapter

```python
# powervision/classify/fault_classifier.py
"""Production-side classifier adapter. Applies preprocessing, temperature
scaling, per-component fault masking, and returns calibrated probabilities."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import numpy as np, torch, torch.nn.functional as F
import timm
import yaml

from powervision.preproc.pipeline import serve_preprocess_classifier, PreprocConfig

class FaultClassifier:
    def __init__(self, run_dir: str | Path, device: str = "cuda:0"):
        run_dir = Path(run_dir)
        self.run_dir = run_dir
        self.device = torch.device(device)
        ckpt = torch.load(run_dir / "weights" / "best.pt", map_location="cpu")
        self.cfg = ckpt["cfg"]
        self.class_index = json.loads((run_dir / "class_index.json").read_text())
        self.T = float(json.loads((run_dir / "temperature.json").read_text())["temperature"])
        self.metadata = json.loads((run_dir / "metadata.json").read_text())

        self.model = timm.create_model(self.cfg["model"]["name"],
                                       num_classes=len(self.class_index), pretrained=False).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        if device.startswith("cuda"):
            self.model = self.model.half()

        # Per-component fault mask (logit-space mask: -inf for impossible classes)
        routing = yaml.safe_load(
            (Path("configs/classifier/component_fault_routing.yaml")).read_text())
        self.component_to_idx_set: dict[str, set[int]] = {}
        name_to_idx = {c["name"]: i for i, c in enumerate(self.class_index)}
        for comp_name, fault_names in routing["components"].items():
            self.component_to_idx_set[comp_name] = {
                name_to_idx[n] for n in fault_names if n in name_to_idx
            } | {name_to_idx.get("no_fault", -1)} - {-1}

    @torch.inference_mode()
    def predict(self, crop_bgr: np.ndarray, component_name: str | None = None
                ) -> dict:
        x = torch.from_numpy(serve_preprocess_classifier(crop_bgr, PreprocConfig(target_size=384)))
        x = x.unsqueeze(0).to(self.device).half() if self.device.type == "cuda" else x.unsqueeze(0)
        logits = self.model(x)[0] / self.T

        # Mask out classes impossible for this component
        if component_name and component_name in self.component_to_idx_set:
            allowed = self.component_to_idx_set[component_name]
            mask = torch.full_like(logits, float("-inf"))
            for i in allowed: mask[i] = 0.0
            logits = logits + mask

        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        top3 = probs.argsort()[-3:][::-1]
        return {
            "top1_index": int(top3[0]),
            "top1_fault_id": int(self.class_index[top3[0]]["fault_id"]),
            "top1_fault_name": self.class_index[top3[0]]["name"],
            "top1_prob": float(probs[top3[0]]),
            "top3": [
                {"index": int(i),
                 "fault_id": int(self.class_index[i]["fault_id"]),
                 "fault_name": self.class_index[i]["name"],
                 "prob": float(probs[i])}
                for i in top3
            ],
            "logits_raw": None,        # set verbose=True to populate
        }

    def predict_batch(self, crops_with_components: Iterable[tuple[np.ndarray, str | None]]
                       ) -> list[dict]:
        return [self.predict(c, comp) for c, comp in crops_with_components]
```

### 7.1 `configs/classifier/component_fault_routing.yaml`

Auto-generated by `scripts/classifier/build_routing.py` which joins `component_taxonomy.yaml` and `fault_taxonomy.yaml`:

```yaml
# DO NOT EDIT BY HAND — regenerate via scripts/classifier/build_routing.py
version: "2.0.0"
components:
  input_terminal_block:
    - input_cable_fault
    - loose_connection
    - red_white_mismatch
    - ferrule_mismatch
  output_terminal_block:
    - output_cable_fault
    - loose_connection
    - red_white_mismatch
  battery_terminal_post:
    - battery_terminal_corrosion
    - battery_terminal_loose
    - battery_terminal_grease_missing
    - battery_polarity_reversed
  electrolytic_capacitor:
    - rectifier_capacitor_bulge
    - rectifier_capacitor_leakage
    - dc_link_cap_bulge
  # ... ~30 entries ...
```

Routing dramatically reduces inter-class confusion (a `battery_terminal_post` crop can't be misclassified as `fan_blade_broken`).

---

## 8. Integration contracts

### 8.1 Consumed by Phase 6 (Fault decision unit)

Per detection from Phase 3:

```python
from powervision.detect.component_detector import ComponentDetector
from powervision.classify.fault_classifier import FaultClassifier

det = ComponentDetector("runs/component_phase_b/weights/best.pt")
clf = FaultClassifier("runs/classifier/eff_b4_phase_b")

for component_det, crop in det.crop_components(img_bgr):
    pred = clf.predict(crop, component_name=component_det.component_name)
    # → pred dict joins fault_id back to taxonomy; Phase 6 fuses with anomaly verdict
```

### 8.2 Component metadata pinning

`metadata.json` records `taxonomy_version`. Phase 6 refuses to load the classifier if its `taxonomy_version` mismatches the loaded component detector's version. Cross-phase contracts:

| Field | Used by | Validation |
|---|---|---|
| `taxonomy_version` | Phases 3, 4, 5, 6 | Hard match |
| `preproc_version` | Phases 2, 3, 4, 5 | Hard match |
| `temperature` | Phase 6 (confidence gate) | Required |

### 8.3 Per-fault confidence thresholds (overlay on temperature scaling)

Phase 6's confidence gate uses class-specific thresholds when the global gate is too coarse. Optional file:

```yaml
# configs/classifier/per_class_thresholds.yaml
default: 0.60
overrides:
  battery_thermal_runaway_mark: 0.50   # high recall posture for critical
  rectifier_capacitor_bulge:    0.55
  enclosure_door_open:          0.75   # tighter to avoid over-paging
```

---

## 9. Code structure

```text
scripts/classifier/
├── build_no_fault_crops.py
├── build_routing.py
├── train_classifier.py
├── calibrate_temperature.py
├── evaluate.py
├── export_onnx.py
└── tune_per_class_thresholds.py

configs/classifier/
├── phase_a.yaml
├── phase_b.yaml
├── component_fault_routing.yaml
└── per_class_thresholds.yaml

powervision/classify/
├── __init__.py
├── data.py                 # CropDataset, build_class_index
├── sampler.py              # class_balanced_weights, weighted sampler
├── loss.py                 # FocalClassBalancedLoss
├── mixup.py
├── eval.py                 # ECE, per-class recall, etc.
└── fault_classifier.py     # inference adapter

runs/classifier/eff_b4_phase_a/
runs/classifier/eff_b4_phase_b/

dataset_v2/crops/{train,val,test}/<fault_name>/
```

### 9.1 Extended `dvc.yaml` stages

```yaml
stages:
  build_no_fault_crops:
    cmd: python scripts/classifier/build_no_fault_crops.py
    deps:
      - dataset_v2/manifest.parquet
      - dataset_v2/labels/components
      - dataset_v2/images/rgb
      - scripts/classifier/build_no_fault_crops.py
    outs:
      - dataset_v2/crops/train/no_fault
      - dataset_v2/crops/val/no_fault
      - dataset_v2/crops/test/no_fault

  build_routing:
    cmd: python scripts/classifier/build_routing.py
    deps:
      - taxonomy/fault_taxonomy.yaml
      - taxonomy/component_taxonomy.yaml
      - scripts/classifier/build_routing.py
    outs:
      - configs/classifier/component_fault_routing.yaml

  classifier_phase_a:
    cmd: python scripts/classifier/train_classifier.py
         --config configs/classifier/phase_a.yaml
    deps:
      - dataset_v2/crops
      - configs/classifier/phase_a.yaml
      - powervision/classify
      - scripts/classifier/train_classifier.py
    outs:
      - runs/classifier/eff_b4_phase_a/weights/best.pt
      - runs/classifier/eff_b4_phase_a/class_index.json

  classifier_phase_b:
    cmd: python scripts/classifier/train_classifier.py
         --config configs/classifier/phase_b.yaml
    deps:
      - runs/classifier/eff_b4_phase_a/weights/best.pt
      - dataset_v2/crops
      - configs/classifier/phase_b.yaml
    outs:
      - runs/classifier/eff_b4_phase_b/weights/best.pt
      - runs/classifier/eff_b4_phase_b/metadata.json

  classifier_calibrate:
    cmd: python scripts/classifier/calibrate_temperature.py
         --run-dir runs/classifier/eff_b4_phase_b
    deps:
      - runs/classifier/eff_b4_phase_b/weights/best.pt
    outs:
      - runs/classifier/eff_b4_phase_b/temperature.json

  classifier_evaluate:
    cmd: python scripts/classifier/evaluate.py
         --run-dir runs/classifier/eff_b4_phase_b --split test
    deps:
      - runs/classifier/eff_b4_phase_b/weights/best.pt
      - runs/classifier/eff_b4_phase_b/temperature.json
      - dataset_v2/crops/test
    metrics:
      - runs/classifier/eff_b4_phase_b/evaluation/evaluation_report.json

  classifier_export:
    cmd: python scripts/classifier/export_onnx.py
         --run-dir runs/classifier/eff_b4_phase_b
    deps:
      - runs/classifier/eff_b4_phase_b/weights/best.pt
    outs:
      - runs/classifier/eff_b4_phase_b/weights/best.onnx
      - runs/classifier/eff_b4_phase_b/weights/best.engine
```

---

## 10. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Severe class imbalance crushes macro-F1 | Tail classes recall ≈ 0 | Class-balanced loss + weighted sampler + rare-class aug + per-class threshold tuning + Phase 4 fallback for excluded classes |
| Confused pairs (e.g. `capacitor_bulge` ↔ `capacitor_leakage`) | Confusion matrix shows symmetric off-diagonal | Annotator review (Phase 1 §7.2); SME-defined "merged" parent class as fallback option; consider adding a binary auxiliary head trained on the confused pair |
| Calibration drift over time | ECE creeps up; confidence gate misfires | Re-run `calibrate_temperature.py` monthly on freshly labeled val crops |
| Domain shift across UPS types (good on `ups_type_001`, bad on `ups_type_003`) | Slice F1 gap > 10 pts | Stratified resampling; per-`ups_type_id` LoRA adapters (Phase 8 §6); demand more data from underrepresented types |
| Backbone overfits on textures specific to one site | Generalization drop on new sites | Heavier augmentation (CoarseDropout, ImageCompression, RandomShadow); Phase 8 active-learning seeds new sites first |
| Crop quality drops because Phase 3 mAP regressed | Classifier accuracy correlates with detector mAP | Pin both phases via taxonomy_version; gate promotion of detector on classifier downstream eval |
| `no_fault` class dominates predictions | Classifier becomes "no fault detector"; misses real faults | Cap `no_fault` crops to 1.5× the largest fault class; raise class-balanced weight on minority fault classes |
| Mixup at extreme severity classes erodes critical-fault recall | Critical recall < 0.90 | Disable mixup/cutmix for samples whose label has `severity = critical` (controlled by `mix_skip_severity_critical: true` in cfg) |
| Per-class threshold tuned to test split (leakage) | False sense of safety | `tune_per_class_thresholds.py` only reads `val/`, never `test/` |
| Routing config out of date with new fault IDs | Inference masks out a newly valid (component, fault) pair | `build_routing.py` is a DVC stage; CI fails if routing not regenerated after taxonomy bump |
| FP16 inference mismatch with FP32 training (subtle accuracy loss) | Test top-1 drops 1–2 pts in production | Validate ONNX/TRT parity via `export_onnx.py` parity test before promotion |

---

## 11. Phase 5 exit checklist

- [ ] `dataset_v2/crops/{train,val,test}/<fault_name>/` populated for all eligible classes (train_count ≥ 10).
- [ ] `no_fault` crops generated and balanced.
- [ ] `class_index.json` written and version-locked at Phase A start.
- [ ] `configs/classifier/component_fault_routing.yaml` auto-generated and committed.
- [ ] Phase A best.pt produced; macro-F1 monotonically improved.
- [ ] Phase B best.pt produced; macro-F1 ≥ 0.80 on val.
- [ ] `temperature.json` calibrated (ECE post-scaling ≤ 0.05).
- [ ] `evaluate.py --split test` returns `verdict: PASS`.
- [ ] Critical-fault recall ≥ 0.90.
- [ ] Per-class recall ≥ 0.70 for all classes (excluded classes documented).
- [ ] ONNX + TensorRT export passes parity test.
- [ ] `FaultClassifier` adapter unit-tested.
- [ ] MLflow registry entry `powervision-fault-classifier/v1` created (Stage = `Staging`).
- [ ] Demo (`api/main.py`, `api/streamlit_app.py`) inference unchanged.

Phase 5 is **frozen** when all boxes are checked. Phase 6 begins.
