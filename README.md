# ups_yolo — Phase 1: Dataset preparation (YOLO OBB)

This repository prepares an oriented bounding-box (OBB) dataset for training a **YOLO26s** model on six Electromil three-phase transformer terminal faults. Raw photos are augmented with Albumentations, split into train/validation/test sets with stratified sampling, and exported in YOLO OBB label format for Ultralytics.

## Setup

Create a virtual environment (recommended), then install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Placing raw images and labels

For each fault class, put images under `data/raw/<class_folder>/` and matching YOLO OBB labels under `data/raw/<class_folder>/labels/`:

- `data/raw/no_fault/*.jpg|*.jpeg|*.png` with `data/raw/no_fault/labels/<same_stem>.txt`
- The same pattern for: `input_cable_fault`, `loose_connection`, `output_cable_fault`, `signal_cable_fault`, `screw_fault`

Each label file contains one line per object:

`class_id cx cy w h angle`

All of `cx`, `cy`, `w`, and `h` are normalized to the image width/height. `angle` is in degrees (OpenCV `minAreaRect`-style convention) and should lie in `[-180, 180]`. An empty `.txt` file denotes a background image with no objects.

## Augmented dataset (`data/augmented/`) — not stored in git

The folders **`data/augmented/`** and **`data/raw/`** are **gitignored** so the GitHub repo stays small (no multi‑hundred‑MB JPEG pushes).

**Get the augmented data in one of these ways:**

1. **Build it yourself** — After placing files under `data/raw/...` on your machine (not in git), run `python scripts/augment.py` (optional: `--target-per-class N`). That writes `data/augmented/` and updates `dataset.yaml`.
2. **Download a shared archive** — Upload your zip to Google Drive, a GitHub Release, cloud storage, etc., then put the **public or shared download link** here for collaborators:
   - **Augmented dataset download:** *`https://YOUR_LINK_TO_ZIP_OR_FOLDER_HERE`*

If you use a zip, extract it at the **project root** so you have `data/augmented/images/{train,val,test}/` and `data/augmented/labels/{train,val,test}/`. The committed `dataset.yaml` uses `path: data/augmented` (relative to the repo root).

## Running the pipeline (from project root)

Run in this order:

```bash
python scripts/augment.py
python scripts/verify_dataset.py
python scripts/preview_augmentation.py
```

1. **`augment.py`** — Reads every class folder, applies the augmentation pipeline up to **`--target-per-class`** samples per class (default **150**), performs a **70% / 15% / 15%** stratified split across all classes using `random_state=42`, writes PNG images and `.txt` labels, and **rewrites `dataset.yaml`** with a **relative** `path: data/augmented` for portable clones.
2. **`verify_dataset.py`** — Validates directory layout, 1:1 image/label pairing, label numeric ranges, and image readability (Pillow). Writes `verification_report.json` in the project root and prints **PASS/FAIL** per check.
3. **`preview_augmentation.py`** — Picks one raw image per class, runs the **same** augmentation pipeline once, draws OBBs on the original and augmented views, and saves `preview_augmentation.png` (3×2 grid of side-by-side comparisons).

## Expected layout after `augment.py`

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

`dataset.yaml` will point `train`, `val`, and `test` at `images/<split>` under the resolved dataset root; labels live in the parallel `labels/<split>` folders with identical file stems.

## Label format (YOLO OBB)

Each non-empty label line has exactly six values:

| Field | Meaning                                      |
|-----------|-----------------------------------------------|
| `class_id` | Integer class index `0`–`5`                  |
| `cx`, `cy` | Normalized center of the oriented rectangle |
| `w`, `h`   | Normalized side lengths of the rectangle     |
| `angle`    | Rotation in degrees (see OpenCV `boxPoints`) |

This matches Ultralytics-style **OBB** training when `task: obb` is set in `dataset.yaml`.

## Fault classes (physical meaning)

| ID | Name | Intended physical fault |
|----|----------------------|-------------------------|
| 0  | `no_fault`           | Reference / ideal wiring and hardware |
| 1  | `input_cable_fault`  | Input cable phase order reversed (e.g. 8→9→10) |
| 2  | `loose_connection`   | Blue wire at top-left input terminal unseated |
| 3  | `output_cable_fault` | Black/white output cables mismatched at output terminal |
| 4  | `signal_cable_fault` | Two blue signal wires at top-left input incorrect |
| 5  | `screw_fault`        | One input terminal screw with wrong thread count |

## Notes

- Scripts assume they are run as `python scripts/<script>.py` from the **project root** so relative paths resolve correctly.
- If a raw image has **no** label file, it is skipped with a warning. An **empty** label file produces a background sample (empty exported label).
- The augmentation pipeline uses **axis-aligned YOLO boxes as a proxy** plus **four corner keypoints** so oriented boxes stay geometrically consistent under flips, rotation, and crop.
