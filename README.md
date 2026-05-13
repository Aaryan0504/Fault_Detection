# Fault Detection (YOLO)

This repository prepares a **YOLO detect** dataset (axis-aligned bounding boxes) and trains an **Ultralytics YOLO** model on transformer terminal fault classes. Raw photos are **manually labeled**; the **first six fault types** (class IDs **0–5** in `dataset.yaml`) use **offline augmentation** because they had limited data—variants are generated, stored on **Google Drive**, and the model is trained in **Google Colab**. The **last three fault types** (class IDs **6, 7, 8**) already had **enough images**, so those are **not augmented** and are used **directly from Drive** alongside the augmented set for a single nine-class training run.

---

## Dataset strategy (Drive + Colab)

| Group | Class IDs (`dataset.yaml`) | Data volume | Augmentation | Storage / training |
|--------|----------------------------|--------------|--------------|---------------------|
| **First six faults** | `0`–`5` | Limited | Yes — train split is expanded with Albumentations (see Phase 1 scripts); val/test copied as-is | Augmented images and labels are uploaded to **Google Drive**; **model training runs in Google Colab** (GPU). |
| **Last three faults** | `6`, `7`, `8` | Sufficient | **No** — originals used as-is | Placed on **Google Drive** without script-based augmentation; Colab (or local training) reads them from the same dataset layout as the other classes. |

The committed root file **`dataset.yaml`** defines **nine** classes (`nc: 9`) and points at the combined dataset root (typically `data/augmented` after you merge augmented 0–5 data with direct 6–8 data under the same `images/` / `labels/` tree). If you train only from Drive in Colab, mount Drive and set `path` in a copy of `dataset.yaml` to the mounted folder so `train`, `val`, and `test` resolve correctly.

---

## Fault classes (`dataset.yaml`)

| ID | Name | Notes |
|----|------|--------|
| 0 | `input_cable_fault` | In the “first six” group; typically augmented when data is scarce. |
| 1 | `loose_connection` | Same. |
| 2 | `output_cable_fault` | Same. |
| 3 | `ri_cable_mismatch` | Same. |
| 4 | `screw_faults` | Same. |
| 5 | `signal_cable_mismatch` | Same. |
| 6 | `J14_cable_mismatch` | **Enough data** — usually **no** Albumentations pipeline; use from Drive. |
| 7 | `red_white_mismatch` | Same. |
| 8 | `ferrule_mismatch` | Same. |

**Physical meaning (examples for early classes):**

- **0 — `input_cable_fault`:** Input cable phase order reversed (e.g. 8→9→10).
- **1 — `loose_connection`:** e.g. blue wire at top-left input terminal unseated.
- **2 — `output_cable_fault`:** Black/white output cables mismatched at output terminal.
- **3 — `ri_cable_mismatch`:** R/I cable mismatch.
- **4 — `screw_faults`:** e.g. one input terminal screw with wrong thread count.
- **5 — `signal_cable_mismatch`:** Signal cable mismatch.

Classes **6–8** follow the same YOLO label format; definitions match your field labeling convention.

**“No fault” images:** Treated as **background** (no object), not a class. Use a matching label file that is **empty**.

---

## Prerequisites

- **`data/augmented/`** (and optionally `data/raw/`) are **not stored in git** — see below for how to obtain or build them.
- Layout: `data/augmented/images/{train,val,test}/` and matching `labels/` trees, plus root `dataset.yaml`.
- Python 3 with dependencies: `pip install -r requirements.txt`.
- **Local training:** CUDA GPU strongly recommended; CPU is slow.
- **Colab training:** Use the Drive-mounted dataset path in `dataset.yaml`.

---

## Setup

From the **project root** (`Fault_Detection_Working/`):

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Dataset preparation (raw → augmented for classes 0–5)

### Placing raw images and labels (manual)

```text
data/raw/
  Images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label line (non-empty): `class_id cx cy w h` with all values normalized to `[0, 1]` relative to image width/height.

### Augmented dataset (`data/augmented/`) — not in git

Folders **`data/augmented/`** and **`data/raw/`** are **gitignored** so the repo stays small.

**Ways to get data:**

1. **Build augmented data locally** — After filling `data/raw/...`, run `python scripts/augment.py` (optional: `--aug-per-image K`). That writes augmented **train** images under `data/augmented/`, copies val/test unchanged, updates `dataset.yaml`. Use this especially for the **first six** fault types; merge in **un-augmented** images/labels for classes **6–8** into the same split folders if they live elsewhere.
2. **Download from shared storage** — e.g. a zip on **Google Drive**; extract at project root so paths match `dataset.yaml`.

**Collaborator download link (placeholder):** *`https://YOUR_LINK_TO_ZIP_OR_FOLDER_HERE`*

### Running the Phase 1 pipeline (from project root)

```bash
python scripts/augment.py
python scripts/verify_dataset.py
python scripts/preview_augmentation.py
```

| Script | Role |
|--------|------|
| **`augment.py`** | Augments **train** only (`--aug-per-image K` variants per train image); copies val/test unchanged; writes PNG + `.txt` under `data/augmented/`; aligns `dataset.yaml` with `path: data/augmented`. |
| **`verify_dataset.py`** | Layout, 1:1 image/label pairing, numeric ranges, readability; writes `verification_report.json`, prints PASS/FAIL. |
| **`preview_augmentation.py`** | Sample raw images, one augmentation pass, draws boxes → `preview_augmentation.png`. |

### Layout after `augment.py` (and merging class 6–8 data)

```text
data/augmented/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

`dataset.yaml` points `train`, `val`, `test` at `images/<split>`; labels live in parallel `labels/<split>/` with the same file stems.

### Label format (YOLO detect)

Each non-empty line: `class_id cx cy w h` with `class_id` in `0`–`8` for the nine classes in the committed `dataset.yaml`.

---

## Phase 2 — YOLO training configuration and two-phase fine-tuning

Phase 2 trains a **YOLO detect** model on the **combined** nine-class dataset (augmented **0–5** + direct **6–8** from Drive when you follow the workflow above). You can run training **locally** with the scripts below or **in Google Colab** against the dataset on Drive.

### Layout (Phase 2)

- `configs/yolo26s_finetune.yaml` — hyperparameter overrides (MuSGD, cosine LR, light aug, `dfl: 0.0`, `task: detect`).
- `scripts/train_phase_a.py` — freeze backbone (10 layers), train head ~20 epochs.
- `scripts/train_phase_b.py` — unfreeze all, full fine-tune up to 100 epochs with early stopping.
- `scripts/monitor_training.py` — live plots from `results.csv`.
- `scripts/validate_model.py` — evaluation pack under `runs/evaluation/`.
- `runs/phase_a/`, `runs/phase_b/` — training outputs (weights, CSV, plots).

### Execution order (local, from project root)

```bash
python scripts/train_phase_a.py
python scripts/train_phase_a.py --dry-run
```

Optional monitor (second terminal):

```bash
python scripts/monitor_training.py
```

Phase B (needs `runs/phase_a/weights/best.pt`):

```bash
python scripts/train_phase_b.py
```

Test split evaluation:

```bash
python scripts/validate_model.py --split test
```

Overrides:

```bash
python scripts/train_phase_b.py --phase-a-weights runs/phase_a/weights/best.pt
python scripts/validate_model.py --weights runs/phase_b/weights/best.pt --split val
```

### Why two-phase training?

On a **small or imbalanced** industrial dataset, training all layers from scratch often **overfits** or erases useful pretrained backbone features. **Phase A** freezes most of the network and adapts the detection head. **Phase B** unfreezes with a **lower** learning rate so the backbone refines without destroying the head.

### Evaluation outputs (`runs/evaluation/`)

| Artifact | Meaning |
|----------|---------|
| `confusion_matrix.png` | Rows = ground truth, columns = prediction (includes background for unmatched boxes). |
| `per_class_metrics.csv` | Precision, recall, mAP@0.5, mAP@0.5:0.95 per class. |
| `per_class_metrics.png` | Bar chart of precision, recall, mAP50. |
| `evaluation_report.json` | Overall mAP, per-class APs, timing, **verdict**. |

**Verdict (`validate_model.py`):**

- **PASS** — every class mAP50 ≥ 0.70  
- **WARN** — all ≥ 0.60 but some &lt; 0.70  
- **FAIL** — any class mAP50 &lt; 0.60  

### Resuming after interruption

- **Training:** Ultralytics can `resume=True` on the same run dir with `last.pt`. Re-run the same phase script, or edit `runs/phase_a/args.yaml` / `runs/phase_b/args.yaml` and use the CLI / `resume=True` in `model.train()` with the same `save_dir`.
- **Phase A prompt:** If `runs/phase_a/weights/best.pt` exists, `train_phase_a.py` may ask to skip training and refresh metrics only.

### GPU memory (local, guidance)

| Batch | Phase | Approx. VRAM |
|-------|--------|----------------|
| 16 | Phase A | ~6 GB |
| 8 | Phase B | ~4 GB |
| 4 | Fallback | ~3 GB |

Lower `batch` in the script if you hit OOM.

### If a class underperforms (mAP50 &lt; 0.6)

1. Collect more **real** images for that fault.  
2. **Audit labels** (class id, missed boxes, tight bboxes).  
3. For **0–5**, consider stronger or class-aware augmentation; tune `configs/yolo26s_finetune.yaml` cautiously.

### Notes

- Phase A starts from **`yolo11m.pt`** (see `scripts/train_phase_a.py`).
- `task: detect` uses axis-aligned boxes and standard box metrics.
- **`monitor_training.py`** watches `results.csv` (default `runs/phase_b`), refreshes every 10 s, saves `training_monitor.png`, Rich status line; stop with Ctrl+C.

---

## Inference and demo (optional)

### Streamlit UI (`api/streamlit_app.py`)

Lightweight **browser demo** for presentations: upload a faulty terminal image, see the **predicted fault name**, the **tagged** detection image, and an **ideal reference** image when paths are configured.

**Run** (from project root, after `pip install -r requirements.txt` and with `runs/phase_b/weights/best.pt` present, or override weights in the YAML below):

```bash
streamlit run api/streamlit_app.py
```

**Configuration — `api/ideal_images.yaml`**

| Key | Used for class IDs |
|-----|---------------------|
| `group_01245` | `0`, `1`, `2`, `4`, `5` (one shared reference image) |
| `class_3` | `3` |
| `class_6` | `6` |
| `class_7` | `7` |
| `class_8` | `8` |

Optional: set `weights` to an absolute path if `best.pt` is not at `runs/phase_b/weights/best.pt`. To point at another YAML file, set environment variable `FAULT_IDEAL_YAML` to that path.

**Windows paths in YAML:** do not use double quotes around `C:\...` (backslash is an escape in YAML). Use **single quotes** (`'C:\Users\...'`) or **forward slashes** (`C:/Users/...`).

Inference writes the upload **verbatim** to a temp file and calls `model.predict(source=...)` (same idea as the FastAPI route), so pixels are not preprocessed through PIL before the model.

**Ultralytics version:** YOLO11 checkpoints need **Ultralytics ≥ 8.3** (see `requirements.txt`). If loading `best.pt` fails with `C3k2` / attribute errors, upgrade: `pip install -U "ultralytics>=8.3.100"`.

If the app still looks dark, use the **☰ menu → Settings → Theme → Light** so Streamlit’s widgets match the app’s light styling.

### FastAPI (`api/main.py`)

REST inference: run with `python api/main.py` (or `uvicorn api.main:app`) and `POST /predict` with the image file. See `api/main.py` for host, port, and response shape.

---

## General notes

- Run scripts as `python scripts/<script>.py` from the **project root** so relative paths resolve.
- Missing label file → image skipped with warning. Empty label file → background sample.
- Augmentation uses Albumentations bbox transforms; labels are the **transformed** boxes, not guessed.

---

## Repository root (reference)

Key paths under `Fault_Detection_Working/`:

- `dataset.yaml` — nine classes, dataset root path, splits.  
- `data/raw/`, `data/augmented/` — local data (gitignored).  
- `scripts/` — augment, verify, preview, train Phase A/B, monitor, validate.  
- `configs/` — YOLO training overrides.  
- `runs/` — training and evaluation outputs (often gitignored or large).  
- `api/` — FastAPI server (`main.py`), Streamlit demo (`streamlit_app.py`), ideal-image paths (`ideal_images.yaml`).  

**`README_PHASE2.md`** is a one-line pointer to this file so older links still resolve.
