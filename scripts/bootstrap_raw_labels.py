"""Create YOLO OBB label files for raw images from folder layout and naming.

By default:
 - ``no_fault`` images get an **empty** label (background).
  - Other classes get a **tight placeholder** OBB in a plausible region of the frame
    (refine with a real labeler before serious training).

Use ``--legacy-full-frame`` to restore the old near-full-image box for every file.

Run from project root: ``python scripts/bootstrap_raw_labels.py``.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import cv2

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Folder name -> YOLO class id (must match dataset.yaml).
FOLDER_TO_CLASS: dict[str, int] = {
    "no_fault": 0,
    "input_cable_fault": 1,
    "loose_connection": 2,
    "output_cable_fault": 3,
    "signal_cable_fault": 4,
    "screw_fault": 5,
}

# Expected filename stem prefix (before _<number>) for sanity checks.
FOLDER_TO_NAME_PREFIX: dict[str, str] = {
    "no_fault": "No_Fault",
    "input_cable_fault": "Input_Cable_Fault",
    "loose_connection": "Loose_Connection",
    "output_cable_fault": "Output_Cable_Fault",
    "signal_cable_fault": "Signal_Cable_Fault",
    "screw_fault": "Screw_Fault",
}

# Normalized (cx, cy, w, h, angle_deg) placeholders — smaller than full frame.
# Corners via OpenCV minAreaRect convention; refine per image in a labeling tool.
PLACEHOLDER_OBB_BY_CLASS: dict[int, tuple[float, float, float, float, float]] = {
    1: (0.38, 0.30, 0.22, 0.20, 0.0),
    2: (0.24, 0.32, 0.16, 0.14, 0.0),
    3: (0.52, 0.68, 0.24, 0.22, 0.0),
    4: (0.26, 0.30, 0.14, 0.14, 0.0),
    5: (0.42, 0.38, 0.12, 0.12, 0.0),
}


def obb_line_normalized(cls: int, cx: float, cy: float, w: float, h: float, angle_deg: float) -> str:
    """One Ultralytics OBB label line (class + 8 normalized corner coordinates)."""

    pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
    flat = pts.flatten()
    return f"{cls} " + " ".join(f"{float(x):.6f}" for x in flat)


def legacy_full_frame_obb_line(cls: int) -> str:
    """Near-full-frame box (historical bootstrap behavior)."""

    return obb_line_normalized(cls, 0.5, 0.5, 0.95, 0.95, 0.0)


_STEM_PATTERN = re.compile(r"^(.+)_(\d+)$", re.IGNORECASE)


def stem_matches_folder(stem: str, folder_name: str) -> bool:
    """Return True if the image stem matches the expected naming pattern.

    Args:
        stem: Filename without extension (e.g. ``No_Fault_12``).
        folder_name: Raw subfolder name (e.g. ``no_fault``).

    Returns:
        Whether ``stem`` starts with the expected prefix for ``folder_name``.
    """

    expected = FOLDER_TO_NAME_PREFIX.get(folder_name)
    if not expected:
        return True
    m = _STEM_PATTERN.match(stem)
    if not m:
        return stem.startswith(expected)
    prefix = m.group(1)
    return prefix.replace("_", "").lower() == expected.replace("_", "").lower()


def write_labels_for_raw_tree(raw_root: Path, *, legacy_full_frame: bool = False) -> tuple[int, int]:
    """Create ``labels/*.txt`` next to each class folder of raw images.

    Args:
        raw_root: Path to ``data/raw``.
        legacy_full_frame: If True, write one large box per image (old behavior).

    Returns:
        ``(num_written, num_warnings)`` for reporting.
    """

    written = 0
    warnings = 0
    for folder_name, cls_id in sorted(FOLDER_TO_CLASS.items(), key=lambda x: x[1]):
        class_dir = raw_root / folder_name
        if not class_dir.is_dir():
            continue
        labels_dir = class_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.name.startswith("."):
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            stem = image_path.stem
            if not stem_matches_folder(stem, folder_name):
                LOGGER.warning(
                    "Filename %s may not match folder %s (expected prefix %s). "
                    "Label still uses class %d from folder.",
                    image_path.name,
                    folder_name,
                    FOLDER_TO_NAME_PREFIX.get(folder_name, "?"),
                    cls_id,
                )
                warnings += 1

            label_path = labels_dir / f"{stem}.txt"
            if legacy_full_frame:
                text = legacy_full_frame_obb_line(cls_id) + "\n"
            elif cls_id == 0:
                text = ""
            else:
                cx, cy, bw, bh, ang = PLACEHOLDER_OBB_BY_CLASS[cls_id]
                text = obb_line_normalized(cls_id, cx, cy, bw, bh, ang) + "\n"
            label_path.write_text(text, encoding="utf-8")
            written += 1

    return written, warnings


def main() -> None:
    """Write bootstrap OBB labels for every raw image under ``data/raw``."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap YOLO OBB labels under data/raw.")
    parser.add_argument(
        "--legacy-full-frame",
        action="store_true",
        help="Use one ~full-frame box per image (including no_fault), as in older versions.",
    )
    args = parser.parse_args()
    n, w = write_labels_for_raw_tree(RAW_ROOT, legacy_full_frame=args.legacy_full_frame)
    print(f"Wrote {n} label file(s); naming warnings: {w}")


if __name__ == "__main__":
    main()
