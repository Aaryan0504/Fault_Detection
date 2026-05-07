"""Create empty YOLO detect label files for raw images (placeholders).

Use this if you want to quickly create one empty `.txt` per image before manual labeling.
It does **not** assign classes or boxes.

Run from project root: ``python scripts/bootstrap_raw_labels.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
RAW_IMAGES_ROOT = RAW_ROOT / "Images"
RAW_LABELS_ROOT = RAW_ROOT / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def write_empty_labels() -> int:
    """Create one empty label file per image in each split.

    Returns:
        Number of label files created (skips existing files).
    """
    if not RAW_IMAGES_ROOT.is_dir():
        raise SystemExit(f"Missing raw images folder: {RAW_IMAGES_ROOT}")
    RAW_LABELS_ROOT.mkdir(parents=True, exist_ok=True)

    created = 0
    for split in ("train", "val", "test"):
        img_dir = RAW_IMAGES_ROOT / split
        if not img_dir.is_dir():
            continue
        lbl_dir = RAW_LABELS_ROOT / split
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(img_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.name.startswith("."):
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = lbl_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                continue
            label_path.write_text("", encoding="utf-8")
            created += 1

    return created


def main() -> None:
    """Write empty label placeholders under ``data/raw/labels/<split>``."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    n = write_empty_labels()
    print(f"Created {n} empty label file(s).")


if __name__ == "__main__":
    main()
