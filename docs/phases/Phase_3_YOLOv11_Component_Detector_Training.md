# Phase 3 — YOLOv11 Component Detector Training

> **Scope:** Backend + model training only. This phase trains the **component-level** YOLOv11 detector that localizes 30 UPS components (cables, terminals, screws, connectors, relays, capacitors, …). It does **not** detect faults directly — fault decisions are made downstream by Phases 4–6 using the cropped ROIs produced here. The two-phase training pattern from the demo (`train_phase_a.py` → `train_phase_b.py`) is preserved and extended.

---

## 1. Phase objective

The demo model is a 9-class **fault** detector. The production component detector is functionally different:

| Aspect | Demo (legacy) | Phase 3 (production) |
|---|---|---|
| Classes | 9 fault classes | 30 component classes (see `taxonomy/component_taxonomy.yaml`) |
| Output use | Final fault localization | ROI source for Phases 4–6 |
| Confidence threshold | 0.25 | 0.30 (tunable per-class in `configs/yolo_component/serve.yaml`) |
| Model | `yolo11m.pt` two-phase | `yolo11m.pt` two-phase + optional `yolo11l.pt` upgrade path |
| Dataset | `dataset.yaml` (9 classes) | `dataset_v2/dataset_v2.yaml` (30 components) |
| Weights output | `runs/phase_b/weights/best.pt` | `runs/component_phase_b/weights/best.pt` |

The legacy 9-class fault detector remains available as a **fallback / regression baseline** (see Phase 8).

Deliverables:

| Deliverable | Artifact |
|---|---|
| Component dataset (YOLO format) | `dataset_v2/labels/components/{train,val,test}/*.txt` (Phase 2 populates) |
| Component dataset descriptor | `dataset_v2/dataset_v2.yaml` (Phase 1 generates) |
| Training scripts | `scripts/component_detector/train_phase_a.py`, `train_phase_b.py` |
| Hyperparameter configs | `configs/yolo_component/{phase_a.yaml, phase_b.yaml, serve.yaml}` |
| Evaluation pack | `runs/component_phase_b/evaluation/{confusion_matrix.png, per_class_metrics.csv, evaluation_report.json}` |
| Inference adapter (used by Phase 4/5/6) | `powervision/detect/component_detector.py` |
| ONNX / TensorRT export | `runs/component_phase_b/weights/best.onnx`, `best.engine` |
| Registered model entry | MLflow registry `powervision-component-detector/v<N>` |

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Format |
|---|---|---|
| Curated images | `dataset_v2/images/rgb/<split>/` | JPEG, 1024 px max side |
| Component YOLO labels | `dataset_v2/labels/components/<split>/` | `class_id cx cy w h` normalized |
| Dataset descriptor | `dataset_v2/dataset_v2.yaml` | Ultralytics YAML |
| Pretrained weights | `yolo11m.pt` (Ultralytics hub) | PyTorch checkpoint |
| Preprocessing module | `powervision.preproc.pipeline` | Python |

### 2.2 Outputs

```text
runs/component_phase_a/
├── weights/{best.pt, last.pt}
├── results.csv
├── results.png
└── args.yaml

runs/component_phase_b/
├── weights/{best.pt, last.pt, best.onnx, best.engine}
├── results.csv
├── results.png
├── train_batch*.jpg
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── PR_curve.png
├── args.yaml
└── evaluation/
    ├── per_class_metrics.csv
    ├── per_class_metrics.png
    ├── confusion_matrix.png
    ├── evaluation_report.json
    └── slice_metrics.json        # per-UPS-type, per-modality, per-lighting slice
```

`runs/component_phase_b/weights/best.pt` is the **serving artifact** consumed by Phase 4, 5, and 6.

---

## 3. End-to-end dataset creation pipeline

### 3.1 Why a separate component dataset (vs reusing fault labels)?

The fault classifier (Phase 5) needs tight crops of *parts* of a UPS regardless of whether they are faulty. The component detector therefore must be trained on a **fault-agnostic** labeling pass:

- Every visible component is boxed, whether faulty or not.
- A faulty capacitor and a healthy capacitor are both labeled `electrolytic_capacitor` (id 12 in `component_taxonomy.yaml`).
- The `dataset_v2/labels/components/` files are independent of `dataset_v2/labels/faults/`.

This separation is the only way to scale to 100+ faults: the component detector keeps its class count small (30) and reaches high mAP fast, while the fault classifier learns fine-grained semantics on cropped patches.

### 3.2 Labeling strategy (extends Phase 1 §7)

| Aspect | Setting |
|---|---|
| Tool | CVAT (Phase 1 §7.1) |
| Annotation type | Axis-aligned bbox |
| Multi-instance | Required (e.g. multiple `screw_terminal` per image) |
| Granularity | One bbox per visually distinct component; overlapping components allowed |
| Occlusion attribute | `0 / 0.25 / 0.5 / 0.75 / 1` recorded per box |
| Truncation attribute | `0 / 0.25 / 0.5 / 0.75 / 1` |
| Inter-annotator IoU acceptance | ≥ 0.7 |
| Min bbox size | 12 px on either side at native capture resolution; smaller boxes dropped |
| Hint pre-labeling | After v1 trained, use it to pre-label new batches (Phase 8 loop). |

### 3.3 Class balancing

Component classes are inherently imbalanced (a UPS has many screws but only one display). We do **not** rebalance the dataset — instead we let the YOLO loss handle it and slice-evaluate to confirm minority classes still meet mAP thresholds.

| Component | Typical instances / image | Strategy |
|---|---|---|
| screw_terminal (10) | 8–40 | Natural |
| cable_run_input (5) | 1–3 | Natural |
| display_panel (26) | 0–1 | Per-class augmentation in `augmentation_profiles.yaml` |
| transformer_core (21) | 0–1 | Per-class augmentation; image-level oversampling via `WeightedRandomSampler` if class mAP < 0.6 |

If oversampling is required, configure in `configs/yolo_component/phase_b.yaml`:

```yaml
class_weights: auto       # custom: inverse-frequency, capped at 5x
sampler:
  type: weighted
  enabled: false          # turn on only if val per-class mAP < 0.60 for any class
```

### 3.4 Augmentation strategy

Recap from Phase 2 §6.6: light photometric + geometric aug at training time (Ultralytics' built-in pipeline) **plus** offline Albumentations augmentation for scarce components only.

| Pipeline | Where | When |
|---|---|---|
| Ultralytics built-in (mosaic, mixup, HSV, Affine) | In `configs/yolo_component/phase_b.yaml` | Always at training |
| Albumentations `RGB_TRAIN` (Phase 2 §6.6) | `scripts/component_detector/augment_scarce.py` | Once, before training, for under-represented components |
| `additional_targets` thermal pipeline | Only if `modality: fused` is enabled | Optional Phase 3.5 |

Demo's convention is preserved: scarce-class augmentation produces additional copies under `dataset_v2/images/rgb/train/` with `__augN` suffix and matching labels under `dataset_v2/labels/components/train/`.

```bash
python scripts/component_detector/augment_scarce.py \
  --components display_panel transformer_core fuse_holder mov_block \
  --aug-per-image 5
```

### 3.5 Train / val / test splits

Inherited verbatim from Phase 1 §9. Phase 3 only re-asserts:

| Constraint | Effect for component detector |
|---|---|
| Device-disjoint | Different UPS units in train and val |
| Time-aware test | Most recent 15% by `captured_at` |
| Stratified per component | Floor of 2 instances in val and test per class |
| Slice analysis | Compute per-`ups_type_id`, per-`modality`, per-`lighting bucket` mAP in eval |

### 3.6 Dataset verification before training

```bash
python scripts/component_detector/verify_component_dataset.py
```

This wraps the existing `verify_dataset.py` logic plus:

- Every YOLO label class ID is in `component_taxonomy.yaml`.
- Every `images/rgb/<split>/<stem>.jpg` has a matching `labels/components/<split>/<stem>.txt` (empty allowed).
- Every bbox is within `[0, 1]` and has positive width/height.
- Per-class instance counts dumped to `runs/component_phase_b/evaluation/dataset_distribution.json`.
- Hard fail if any active component class has 0 instances in `train`.

---

## 4. Model training pipeline

### 4.1 Model selection rationale

| Candidate | Params | Latency (RTX 3060, 640²) | When to use |
|---|---|---|---|
| `yolo11n.pt` | 2.6 M | 4 ms | Edge devices, on-engineer-laptop demos |
| `yolo11s.pt` | 9.4 M | 6 ms | Default for prototypes |
| **`yolo11m.pt`** | 20.1 M | 11 ms | **Default for production** (matches demo) |
| `yolo11l.pt` | 25.3 M | 18 ms | Upgrade path if `yolo11m` mAP plateaus < 0.85 |
| `yolo11x.pt` | 56.9 M | 35 ms | Last resort; usually datasets are the bottleneck |

Choice: **`yolo11m`** matches the existing demo training infrastructure exactly. Server-side inference budget is generous (single-image latency target ≤ 80 ms p95 end-to-end including preprocessing).

### 4.2 Two-phase fine-tuning (preserves demo pattern)

The demo project pattern (frozen backbone → full fine-tune) generalizes well to a 30-class dataset of similar visual character. The phase A/B split provides:

- **Phase A** stabilizes the new heads (head_count differs from pretrained model) without disturbing pretrained backbone features.
- **Phase B** unfreezes everything at a lower LR so the backbone refines without forgetting.

#### Phase A (frozen backbone)

```yaml
# configs/yolo_component/phase_a.yaml
optimizer: MuSGD
momentum: 0.937
weight_decay: 0.0005
lr0: 0.001
lrf: 0.01
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
cos_lr: true

box: 7.5
cls: 0.5
dfl: 0.0       # YOLO26 / 11 detection variant; keep matching configs/yolo26s_finetune.yaml

freeze: 10     # freeze first 10 layers (backbone)
epochs: 20
batch: 16
imgsz: 640
patience: 8

hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 5.0
translate: 0.05
scale: 0.3
mosaic: 1.0
close_mosaic: 5
mixup: 0.05
copy_paste: 0.0

seed: 42
deterministic: true
save_period: 5
project: runs
name: component_phase_a
```

#### Phase B (full fine-tune)

```yaml
# configs/yolo_component/phase_b.yaml
optimizer: MuSGD
momentum: 0.937
weight_decay: 0.0005
lr0: 0.0003          # lower than Phase A
lrf: 0.01
warmup_epochs: 1.0
cos_lr: true

box: 7.5
cls: 0.5
dfl: 0.0

freeze: 0
epochs: 100
batch: 8
imgsz: 640
patience: 15

hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
mosaic: 1.0
close_mosaic: 15
mixup: 0.15
copy_paste: 0.0

seed: 42
deterministic: true
save_period: 10
project: runs
name: component_phase_b
```

### 4.3 Training scripts

#### `scripts/component_detector/train_phase_a.py`

```python
"""Phase A: train head only on frozen yolo11m backbone."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     default=str(ROOT / "dataset_v2" / "dataset_v2.yaml"))
    ap.add_argument("--cfg",      default=str(ROOT / "configs" / "yolo_component" / "phase_a.yaml"))
    ap.add_argument("--weights",  default="yolo11m.pt")
    ap.add_argument("--device",   default="0")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    model = YOLO(args.weights)

    train_kwargs = dict(
        data=args.data,
        device=args.device,
        cfg=args.cfg,           # Ultralytics merges cfg into defaults
        exist_ok=True,
    )
    if args.dry_run:
        train_kwargs.update(epochs=1, batch=2)

    model.train(**train_kwargs)

if __name__ == "__main__":
    sys.exit(main())
```

#### `scripts/component_detector/train_phase_b.py`

```python
"""Phase B: full fine-tune from Phase A best.pt."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PHASE_A_DEFAULT = ROOT / "runs" / "component_phase_a" / "weights" / "best.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",            default=str(ROOT / "dataset_v2" / "dataset_v2.yaml"))
    ap.add_argument("--cfg",             default=str(ROOT / "configs" / "yolo_component" / "phase_b.yaml"))
    ap.add_argument("--phase-a-weights", default=str(PHASE_A_DEFAULT))
    ap.add_argument("--device",          default="0")
    ap.add_argument("--resume",          action="store_true")
    args = ap.parse_args()

    weights = args.phase_a_weights
    if args.resume:
        last = ROOT / "runs" / "component_phase_b" / "weights" / "last.pt"
        if last.exists():
            weights = str(last)

    if not Path(weights).exists():
        sys.exit(f"Phase A weights not found at {weights}")

    model = YOLO(weights)
    model.train(
        data=args.data,
        device=args.device,
        cfg=args.cfg,
        resume=args.resume,
        exist_ok=True,
    )

    # Write a tiny pointer the inference layer trusts
    (ROOT / "runs" / "component_phase_b" / "weights" / "metadata.json").write_text(json.dumps({
        "taxonomy_version": "2.0.0",
        "preproc_version": "2.0.0",
        "modality": "rgb",
        "imgsz": 640,
    }, indent=2))

if __name__ == "__main__":
    sys.exit(main())
```

### 4.4 Loss function

Ultralytics ≥ 8.3 ships its own composite loss for `task: detect`:

- Box loss: CIoU (default)
- Classification loss: BCE (multi-label-style — appropriate for our 30 classes since multi-instance is common)
- DFL: disabled (`dfl: 0.0`) for YOLO11/26 detect variant; matches the demo config

No custom loss is needed. If a specific component class persistently under-recalls after Phase B, increase `cls_pw` (positive weight) in a config override rather than implementing a focal loss — empirically more stable on small industrial datasets.

### 4.5 Class weights and rare-class boosting

Three escalating fixes if Phase B per-class mAP@0.5 < 0.60 for any class:

| Step | Action |
|---|---|
| 1 | Re-run Phase B with `cls_pw` tuned upward (e.g. `cls_pw: 1.5`) and `single_cls: false` |
| 2 | Apply offline scarce-class augmentation (§3.4) and retrain Phase B |
| 3 | Enable `WeightedRandomSampler` via `sampler.enabled: true` in `phase_b.yaml`; Ultralytics integration via custom callback `scripts/component_detector/callbacks/weighted_sampler.py` |

### 4.6 Checkpoint management

| Artifact | When written | Purpose |
|---|---|---|
| `last.pt` | Every epoch | Resume |
| `best.pt` | When validation fitness improves | Serving + Phase B init |
| `epoch{N}.pt` | Every `save_period` epochs | Time-travel debugging |
| `metadata.json` | After Phase B completes | Pin taxonomy + preproc versions to the weights |

Resume:

```bash
python scripts/component_detector/train_phase_b.py --resume
```

The script finds `runs/component_phase_b/weights/last.pt` automatically.

### 4.7 GPU / compute requirements

| Stage | Recommended | Minimum |
|---|---|---|
| Phase A (20 epochs, 5k images) | 1× RTX 4070 (12 GB), ~45 min | 1× RTX 3060 (6 GB), ~2 h with `batch=8` |
| Phase B (100 epochs, 5k images) | 1× RTX 4090 (24 GB), ~6 h | 1× RTX 3060, ~22 h with `batch=4` |
| Distributed scale-out | DDP across 4× A100 → ~75 min Phase B on 20k images | — |

OOM fallback (preserves demo guidance):

| Batch | Phase | Approx VRAM |
|---|---|---|
| 16 | Phase A | ~7 GB |
| 8  | Phase B | ~5 GB |
| 4  | Fallback | ~3 GB |

### 4.8 Colab parity

The demo workflow runs Phase A/B in Colab via `notebooks/Final_Faults.ipynb`. A parallel notebook lives at `notebooks/Component_Detector.ipynb` and follows the exact same pattern (Drive mount → DVC pull → `python scripts/component_detector/train_phase_a.py` → `train_phase_b.py`).

---

## 5. Evaluation

### 5.1 `scripts/component_detector/validate_model.py`

```python
"""Reproducible eval pack — mirrors demo validate_model.py output schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_W = ROOT / "runs" / "component_phase_b" / "weights" / "best.pt"
DEFAULT_OUT = ROOT / "runs" / "component_phase_b" / "evaluation"

def evaluate(weights: Path, split: str, out: Path):
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(ROOT / "dataset_v2" / "dataset_v2.yaml"),
        split=split,
        imgsz=640, conf=0.001, iou=0.6, save_json=True,
        project=str(out.parent), name=out.name, exist_ok=True,
    )

    # Build per-class CSV
    rows = []
    for i, name in metrics.names.items():
        rows.append(dict(
            class_id=i, name=name,
            precision=float(metrics.box.p[i]),
            recall=float(metrics.box.r[i]),
            mAP50=float(metrics.box.ap50[i]),
            mAP50_95=float(metrics.box.ap[i]),
            instances=int(metrics.box.nc[i]) if hasattr(metrics.box, "nc") else None,
        ))
    df = pd.DataFrame(rows).sort_values("mAP50")
    df.to_csv(out / "per_class_metrics.csv", index=False)

    # Verdict — promoted from demo
    worst = df["mAP50"].min()
    if   worst >= 0.70: verdict = "PASS"
    elif worst >= 0.60: verdict = "WARN"
    else:               verdict = "FAIL"

    report = dict(
        weights=str(weights), split=split,
        overall_mAP50=float(metrics.box.map50),
        overall_mAP50_95=float(metrics.box.map),
        verdict=verdict,
        per_class=df.to_dict("records"),
        worst_class=df.iloc[0]["name"],
    )
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    # Per-class bar
    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(df))))
    ax.barh(df["name"], df["mAP50"])
    ax.axvline(0.7, color="g", linestyle="--", label="PASS≥0.70")
    ax.axvline(0.6, color="orange", linestyle="--", label="WARN≥0.60")
    ax.set_xlim(0, 1); ax.set_xlabel("mAP@0.5"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "per_class_metrics.png", dpi=120)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--split",   default="test", choices=["val", "test"])
    ap.add_argument("--out",     default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    evaluate(Path(args.weights), args.split, out)

if __name__ == "__main__":
    main()
```

### 5.2 Metrics and pass/fail thresholds (extends demo verdict)

| Metric | PASS | WARN | FAIL |
|---|---|---|---|
| Overall mAP@0.5 | ≥ 0.85 | 0.75–0.85 | < 0.75 |
| Overall mAP@0.5:0.95 | ≥ 0.65 | 0.55–0.65 | < 0.55 |
| Worst per-class mAP@0.5 | ≥ 0.70 | 0.60–0.70 | < 0.60 |
| Recall on `loose_connection`-adjacent components (cable, terminal) | ≥ 0.90 | 0.80–0.90 | < 0.80 |
| Inference latency p95 (RTX 3060, 640²) | ≤ 15 ms | 15–25 ms | > 25 ms |

A `FAIL` on *any* row blocks promotion to MLflow Production (Phase 8).

### 5.3 Slice analysis (`slice_metrics.json`)

`evaluate()` is extended to compute mAP@0.5 sliced by:

- `ups_type_id` (from `manifest.parquet`)
- `modality` (RGB vs fused)
- `lighting bucket` (derived from `mean_luminance` quantiles)
- `capture_view` (`front_open`, `side_left`, ...)

Slice values where mAP@0.5 drops > 10% below the overall metric are flagged in the report — these are root-cause hints for data collection priorities (Phase 8 active learning targets).

---

## 6. Inference adapter (used by Phases 4–6)

```python
# powervision/detect/component_detector.py
"""Production-side wrapper for the component detector. Centralizes
preprocessing, batching, confidence gating, and metadata propagation
so downstream consumers do not duplicate logic."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import numpy as np
import cv2
from ultralytics import YOLO

from powervision.preproc.pipeline import serve_preprocess_yolo, PreprocConfig

@dataclass
class Detection:
    component_id: int
    component_name: str
    bbox_xyxy: tuple[float, float, float, float]  # in input-image pixels
    confidence: float

class ComponentDetector:
    def __init__(self, weights: str | Path,
                 conf: float = 0.30, iou: float = 0.5, max_det: int = 100,
                 device: str = "auto"):
        weights = Path(weights)
        self.weights_path = weights
        self.model = YOLO(str(weights))
        # Per-class threshold overrides loaded from sibling file (optional).
        thr_file = weights.parent / "class_thresholds.json"
        self.class_thr: dict[int, float] = json.loads(thr_file.read_text()) if thr_file.exists() else {}
        # Metadata pin (set in Phase 3 §4.3 train_phase_b script).
        meta_file = weights.parent / "metadata.json"
        self.metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.conf, self.iou, self.max_det, self.device = conf, iou, max_det, device

    def predict(self, img_bgr: np.ndarray) -> list[Detection]:
        prepped, _ = serve_preprocess_yolo(img_bgr, PreprocConfig(target_size=640))
        res = self.model.predict(
            source=prepped, conf=self.conf, iou=self.iou, max_det=self.max_det,
            imgsz=640, verbose=False, device=self.device,
        )[0]
        out: list[Detection] = []
        # Map back from letterboxed coords to original.
        h0, w0 = img_bgr.shape[:2]
        s = 640 / max(h0, w0); nh, nw = int(h0*s), int(w0*s)
        pad_y, pad_x = (640 - nh) // 2, (640 - nw) // 2
        for box, conf_, cls_ in zip(res.boxes.xyxy.cpu().numpy(),
                                     res.boxes.conf.cpu().numpy(),
                                     res.boxes.cls.cpu().numpy()):
            cid = int(cls_)
            thr = self.class_thr.get(cid, self.conf)
            if conf_ < thr: continue
            x1, y1, x2, y2 = box
            x1 = max(0, (x1 - pad_x) / s); x2 = min(w0, (x2 - pad_x) / s)
            y1 = max(0, (y1 - pad_y) / s); y2 = min(h0, (y2 - pad_y) / s)
            out.append(Detection(cid, res.names[cid], (float(x1), float(y1), float(x2), float(y2)), float(conf_)))
        return out

    def predict_batch(self, imgs: Iterable[np.ndarray]) -> list[list[Detection]]:
        return [self.predict(i) for i in imgs]

    def crop_components(self, img_bgr: np.ndarray, pad_ratio: float = 0.15
                        ) -> list[tuple[Detection, np.ndarray]]:
        """Return list of (Detection, crop_bgr) ready for Phase 4 / Phase 5."""
        h, w = img_bgr.shape[:2]
        crops: list[tuple[Detection, np.ndarray]] = []
        for det in self.predict(img_bgr):
            x1, y1, x2, y2 = det.bbox_xyxy
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - bw * pad_ratio)); x2 = min(w, int(x2 + bw * pad_ratio))
            y1 = max(0, int(y1 - bh * pad_ratio)); y2 = min(h, int(y2 + bh * pad_ratio))
            crop = img_bgr[y1:y2, x1:x2].copy()
            if crop.size:
                crops.append((det, crop))
        return crops
```

### 6.1 Per-class threshold tuning

After Phase B training, run a threshold sweep on val:

```bash
python scripts/component_detector/tune_class_thresholds.py \
  --weights runs/component_phase_b/weights/best.pt \
  --split val \
  --metric f1 \
  --out runs/component_phase_b/weights/class_thresholds.json
```

This writes a JSON `{class_id: optimal_conf_threshold}` adjacent to `best.pt`. The inference adapter (§6) automatically loads it.

### 6.2 ONNX / TensorRT export

```bash
python -c "from ultralytics import YOLO; YOLO('runs/component_phase_b/weights/best.pt').export(format='onnx', dynamic=True, simplify=True, imgsz=640)"
python -c "from ultralytics import YOLO; YOLO('runs/component_phase_b/weights/best.pt').export(format='engine', half=True, imgsz=640, device=0)"
```

Validate parity:

```bash
python scripts/component_detector/validate_export_parity.py \
  --pt runs/component_phase_b/weights/best.pt \
  --onnx runs/component_phase_b/weights/best.onnx \
  --engine runs/component_phase_b/weights/best.engine \
  --fixture tests/fixtures/component_parity/
```

Parity criterion: ≥ 99% of detections match within IoU ≥ 0.95 and confidence delta ≤ 0.02.

---

## 7. Integration contracts

### 7.1 Consumed by Phase 4 (PatchCore / EfficientAD)

```python
from powervision.detect.component_detector import ComponentDetector
det = ComponentDetector("runs/component_phase_b/weights/best.pt")
for component_det, crop in det.crop_components(img_bgr, pad_ratio=0.15):
    # crop_bgr → anomaly model for component_det.component_name
    ...
```

Phase 4 keys per-component anomaly models by `component_name` from `taxonomy/component_taxonomy.yaml`. If `crop_components()` returns no detections **for components the rule engine expects**, the image is flagged `misframed` (Phase 6 decides whether to escalate to human review).

### 7.2 Consumed by Phase 5 (Fault classifier)

Same `crop_components()` output. The classifier receives the BGR crop **and** the `Detection` (so it knows which component it is — different classifier heads may apply).

### 7.3 Consumed by Phase 6 (Fault decision unit)

Phase 6 receives the **full list of `Detection`** objects (not just crops) to evaluate structural rules ("there must be 4 input cables", "screw count between 6 and 10", etc.).

### 7.4 Versioning contract

The deployed component detector pin must match the deployed fault classifier and rule engine taxonomy version. Phase 6 refuses to run if:

```python
component_detector.metadata["taxonomy_version"] != fault_classifier.metadata["taxonomy_version"]
```

---

## 8. Code structure

```text
scripts/component_detector/
├── verify_component_dataset.py
├── augment_scarce.py
├── train_phase_a.py
├── train_phase_b.py
├── validate_model.py
├── tune_class_thresholds.py
├── validate_export_parity.py
└── callbacks/
    └── weighted_sampler.py

configs/yolo_component/
├── phase_a.yaml
├── phase_b.yaml
└── serve.yaml

powervision/detect/
├── __init__.py
└── component_detector.py

runs/
├── component_phase_a/
└── component_phase_b/

dataset_v2/                  # already defined in Phase 1/2
└── dataset_v2.yaml          # 30-class component descriptor
```

### 8.1 Extended `dvc.yaml` stages for Phase 3

```yaml
stages:
  component_phase_a:
    cmd: python scripts/component_detector/train_phase_a.py
    deps:
      - dataset_v2/dataset_v2.yaml
      - dataset_v2/images/rgb/train
      - dataset_v2/images/rgb/val
      - dataset_v2/labels/components
      - configs/yolo_component/phase_a.yaml
      - scripts/component_detector/train_phase_a.py
    outs:
      - runs/component_phase_a/weights/best.pt
      - runs/component_phase_a/results.csv

  component_phase_b:
    cmd: python scripts/component_detector/train_phase_b.py
    deps:
      - runs/component_phase_a/weights/best.pt
      - dataset_v2/dataset_v2.yaml
      - configs/yolo_component/phase_b.yaml
      - scripts/component_detector/train_phase_b.py
    outs:
      - runs/component_phase_b/weights/best.pt
      - runs/component_phase_b/results.csv

  component_validate:
    cmd: python scripts/component_detector/validate_model.py --split test
    deps:
      - runs/component_phase_b/weights/best.pt
      - dataset_v2/dataset_v2.yaml
      - dataset_v2/labels/components
    metrics:
      - runs/component_phase_b/evaluation/evaluation_report.json

  component_tune_thresholds:
    cmd: python scripts/component_detector/tune_class_thresholds.py
         --weights runs/component_phase_b/weights/best.pt
         --split val
         --out    runs/component_phase_b/weights/class_thresholds.json
    deps:
      - runs/component_phase_b/weights/best.pt
    outs:
      - runs/component_phase_b/weights/class_thresholds.json
```

---

## 9. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Per-class mAP < 0.6 on one component | Verdict FAIL | Escalation ladder §4.5; if still failing, demote the class to "candidate" status in `component_taxonomy.yaml` and route those ROIs to the fault classifier via image-level (not component-level) inference. |
| Overfitting (train mAP >> val mAP) | val curve diverges | Reduce Phase B epochs; raise `weight_decay` to 0.001; enable `mixup: 0.20`; collect more data via Phase 8 |
| Domain shift across UPS types | Slice mAP large gap | Stratified resampling; per-`ups_type_id` fine-tune ("LoRA-like" approach: train a per-type adapter — see Phase 8 §6) |
| Lighting variance crashes recall | Recall drops at night-mode captures | Re-tune CLAHE in Phase 2; add `RandomShadow` aug at higher probability; collect dim-light examples |
| Annotation noise (overlapping screws, etc.) | Confusion matrix shows screw↔terminal swap | CVAT QA double-blind (Phase 1 §7.2); raise `min_area` in Albumentations; consider keypoint detection instead of bbox for screws (future work) |
| OOM in Phase B at `batch=8` | CUDA error | Drop to `batch=4`; use `amp: true` (default); `accumulate: 2` to maintain effective batch |
| Resume mismatch (taxonomy bumped mid-training) | Trainer crashes with `nc` mismatch | `metadata.json` carries `taxonomy_version`; resume script refuses if taxonomy bumped; force retrain from Phase A |
| Inference latency regression after `yolo11l` upgrade | p95 > 25 ms | TensorRT export with FP16; downgrade to `yolo11m` if mAP gap < 1.0 pt |
| Export parity broken | ONNX/TRT detections drift | `validate_export_parity.py` blocks deployment; pin Ultralytics version in `requirements.txt` |
| Pre-labeling drift (Phase 8 self-loop) | Component recall slowly degrades | Always require human review on pre-labeled batches before they hit `dataset_v2/labels/components/`; track `labeling_session_id` in manifest to detect runs of low-quality labels |

---

## 10. Phase 3 exit checklist

- [ ] `dataset_v2/labels/components/{train,val,test}/` populated; `verify_component_dataset.py` PASS.
- [ ] `configs/yolo_component/{phase_a,phase_b,serve}.yaml` reviewed and committed.
- [ ] `runs/component_phase_a/weights/best.pt` produced (Phase A completes).
- [ ] `runs/component_phase_b/weights/best.pt` produced; `metadata.json` written.
- [ ] `validate_model.py --split test` returns `verdict: PASS`.
- [ ] `tune_class_thresholds.py` writes `class_thresholds.json`.
- [ ] ONNX + TensorRT export passes parity check.
- [ ] `ComponentDetector` adapter unit-tested.
- [ ] MLflow registry entry `powervision-component-detector/v1` created (Stage = `Staging`).
- [ ] Demo (`api/main.py`) inference continues to work with `runs/phase_b/weights/best.pt` unchanged.

Phase 3 is **frozen** when all boxes are checked. Phase 4 begins.
