# ups_yolo — Phase 1: Dataset preparation (YOLO detect)

This repository prepares a **YOLO detect** dataset (axis-aligned bounding boxes) for training an Ultralytics YOLO model (Phase 2 uses **`yolo11m.pt`**) on transformer terminal fault classes. Raw photos are **manually labeled**, then the training split is augmented with Albumentations and exported in standard YOLO label format.

## Setup

Create a virtual environment (recommended), then install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Placing raw images and labels (manual)

This repo expects a fixed split layout:

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

Each label file contains one line per object:

`class_id cx cy w h`

All of `cx`, `cy`, `w`, and `h` are normalized to the image width/height (values in `[0, 1]`).

### “No fault” images

“No fault” is treated as **background** (no object), not a class. For a “no fault” image, create the matching label file but leave it **empty**.

## Augmented dataset (`data/augmented/`) — not stored in git

The folders **`data/augmented/`** and **`data/raw/`** are **gitignored** so the GitHub repo stays small (no multi‑hundred‑MB JPEG pushes).

**Get the augmented data in one of these ways:**

1. **Build it yourself** — After placing files under `data/raw/...` on your machine (not in git), run `python scripts/augment.py` (optional: `--aug-per-image K`). That writes `data/augmented/` and updates `dataset.yaml`.
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

1. **`augment.py`** — Uses your existing `data/raw/Images/{train,val,test}` split. It augments **only the train split** (`--aug-per-image K` variants per train image), copies val/test unchanged, writes PNG images and `.txt` labels under `data/augmented/`, and rewrites `dataset.yaml` with a **relative** `path: data/augmented`.
2. **`verify_dataset.py`** — Validates directory layout, 1:1 image/label pairing, label numeric ranges, and image readability (Pillow). Writes `verification_report.json` in the project root and prints **PASS/FAIL** per check.
3. **`preview_augmentation.py`** — Picks sample raw images, runs the same augmentation once, draws **detect boxes** on the original and augmented views, and saves `preview_augmentation.png`.

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

## Label format (YOLO detect)

Each non-empty label line has exactly five values:

| Field | Meaning                                      |
|-----------|-----------------------------------------------|
| `class_id` | Integer class index `0`–`5`                  |
| `cx`, `cy` | Normalized center of the rectangle |
| `w`, `h`   | Normalized width/height of the rectangle     |

This matches Ultralytics **detect** training when `task: detect` is set in `dataset.yaml`.

## Fault classes (physical meaning)

| ID | Name | Intended physical fault |
|----|----------------------|-------------------------|
| 0  | `input_cable_fault`  | Input cable phase order reversed (e.g. 8→9→10) |
| 1  | `loose_connection`   | Blue wire at top-left input terminal unseated |
| 2  | `output_cable_fault` | Black/white output cables mismatched at output terminal |
| 3  | `ri_cable_mismatch`  | R/I cable mismatch |
| 4  | `signal_cable_mismatch` | Signal cable mismatch |
| 5  | `screw_fault`        | One input terminal screw with wrong thread count |

## Notes

- Scripts assume they are run as `python scripts/<script>.py` from the **project root** so relative paths resolve correctly.
- If a raw image has **no** label file, it is skipped with a warning. An **empty** label file produces a background sample (empty exported label).
- The augmentation pipeline uses Albumentations bbox transforms, so augmented labels are the **transformed version** of your manual labels (no guessing).
