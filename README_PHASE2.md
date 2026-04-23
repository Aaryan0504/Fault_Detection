# Phase 2 — YOLO26 model configuration and two-phase training

This phase trains an oriented bounding box (OBB) detector for the six Elletromil fault classes on top of the Phase 1 dataset under `data/augmented/`.

## Prerequisites

- **`data/augmented/` present locally** — Either run Phase 1 (`python scripts/augment.py` after `data/raw/` is populated) or download/extract the shared archive; see **“Augmented dataset — not stored in git”** in `README.md` for the download link placeholder.
- Layout: `data/augmented/images/{train,val,test}/` and matching `labels/` trees, plus root `dataset.yaml`.
- Python environment with dependencies from `requirements.txt` installed (`pip install -r requirements.txt`).
- A CUDA-capable GPU is strongly recommended; CPU training is possible but slow.

## Layout (created or used in Phase 2)

- `configs/yolo26s_finetune.yaml` — hyperparameter overrides (MuSGD, cosine LR, light aug, `dfl: 0.0`, `task: obb`).
- `scripts/train_phase_a.py` — freeze backbone (10 layers), train head ~20 epochs.
- `scripts/train_phase_b.py` — unfreeze all, full fine-tune up to 100 epochs with early stopping.
- `scripts/monitor_training.py` — live plots from `results.csv`.
- `scripts/validate_model.py` — test/val/train evaluation pack under `runs/evaluation/`.
- `runs/phase_a/`, `runs/phase_b/` — training outputs (weights, CSV, plots). Scripts set `save_dir` so runs land at these paths (not under an extra task subfolder).

## Execution order

From the **project root** (`ups_yolo/`):

```bash
python scripts/train_phase_a.py
python scripts/train_phase_a.py --dry-run
```

Optional: start the monitor in another terminal (or background on Unix):

```bash
python scripts/monitor_training.py
```

Phase B (requires `runs/phase_a/weights/best.pt`):

```bash
python scripts/train_phase_b.py
```

Evaluation on the test split:

```bash
python scripts/validate_model.py --split test
```

Override Phase A weights or checkpoint path when needed:

```bash
python scripts/train_phase_b.py --phase-a-weights runs/phase_a/weights/best.pt
python scripts/validate_model.py --weights runs/phase_b/weights/best.pt --split val
```

## Why two-phase training?

On a **small, specialized dataset**, training all layers from the start often **overfits** or **destroys** useful COCO features in the backbone. **Phase A** freezes most of the network and adapts the **detection/OBB head** with a moderate learning rate so class boundaries stabilize. **Phase B** then unfreezes the full model with a **lower LR** so the backbone can refine features without wiping out the head. This is a standard transfer-learning pattern for limited industrial data.

## Interpreting evaluation outputs (`runs/evaluation/`)

| Artifact | Meaning |
|----------|---------|
| `confusion_matrix.png` | Count heatmap: **rows = ground-truth class**, **columns = predicted class** (includes a background bucket for unmatched boxes). |
| `per_class_metrics.csv` | Precision, recall, mAP@0.5, mAP@0.5:0.95 per class. |
| `per_class_metrics.png` | Grouped bars comparing precision, recall, and mAP50 per class. |
| `evaluation_report.json` | Overall mAP, per-class APs, timing (ms per image per stage), and a **verdict** string. |

**Verdict rules (from `validate_model.py`):**

- **PASS** — every class has mAP50 ≥ 0.70.
- **WARN** — all classes ≥ 0.60 but at least one class below 0.70.
- **FAIL** — any class has mAP50 &lt; 0.60.

## Resuming after interruption

- **Training:** Ultralytics supports `resume=True` on the **same run directory** and `last.pt`. For the scripts here, the simplest recovery is to re-run the same phase script; to resume in-place, open the generated `runs/phase_a/args.yaml` or `runs/phase_b/args.yaml`, confirm paths, then use the Ultralytics CLI or add `resume=True` to `model.train()` for that run (same `save_dir`).
- **Phase A “Resume?” prompt:** If `runs/phase_a/weights/best.pt` already exists, `train_phase_a.py` asks whether to **skip training** and only refresh metrics/summary.

## GPU memory (guidance)

Approximate VRAM for YOLO26-S OBB at 640×640:

| Batch | Phase | Approx. VRAM |
|-------|--------|----------------|
| 16 | Phase A | ~6 GB |
| 8 | Phase B | ~4 GB |
| 4 | Fallback | ~3 GB |

If you hit OOM, edit the `batch=` argument in the corresponding script (or pass overrides if you extend the CLI).

## When a class fails (mAP50 &lt; 0.6)

1. **Collect more real images** for that fault mode (lighting, angles, wiring variants).
2. **Audit labels** for that class (OBB points, class id, missed objects).
3. **Increase class-aware augmentation** in Phase 1 or bump relevant HSV/geometry gains in `configs/yolo26s_finetune.yaml` cautiously.

## YOLO26-specific notes

- **NMS-free / end-to-end inference** options exist in YOLO26; validation still reports standard OBB metrics.
- **`dfl: 0.0`** matches the YOLO26 setup where the distribution focal loss term is not used; box loss is handled by the updated head.
- **`task: obb`** selects rotated boxes and `OBBMetrics` (e.g. `metrics/mAP50(B)` in `results.csv`).
- **`optimizer: MuSGD`** is the momentum SGD variant recommended in the Ultralytics stack for these models.

## Monitoring training

`monitor_training.py` watches `results.csv` in the run folder (default `runs/phase_b`), refreshes every 10 seconds, saves `training_monitor.png` in that folder, and prints a one-line Rich status. It waits quietly until the CSV appears. Stop with **Ctrl+C**.
