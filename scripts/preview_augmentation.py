"""Preview augmentation by drawing OBB boxes before/after for one image per class."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_PATH = PROJECT_ROOT / "preview_augmentation.png"
SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import augment as aug  # noqa: E402 - import after sys.path injection


CLASS_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


def draw_yolo_obb_lines(image_bgr: np.ndarray, label_text: str, palette: list[tuple[int, int, int]]) -> np.ndarray:
    """Draw all YOLO OBB objects from label text onto a BGR image copy.

    Args:
        image_bgr: Source image in BGR format.
        label_text: Newline-separated YOLO OBB label lines (may be empty).
        palette: List of BGR colors indexed by class id.

    Returns:
        A new image array with drawn contours.
    """

    canvas = image_bgr.copy()
    if not label_text.strip():
        return canvas
    h, w = canvas.shape[:2]
    for raw_line in label_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(float(parts[0]))
        color = palette[cls_id % len(palette)]
        if len(parts) == 9:
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
            pts = np.stack(
                [coords[:, 0] * w, coords[:, 1] * h],
                axis=1,
            ).astype(np.int32)
            cv2.drawContours(canvas, [pts], 0, color, 2)
        elif len(parts) == 6:
            cx, cy, bw, bh, ang = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            c_x = cx * w
            c_y = cy * h
            rw = bw * w
            rh = bh * h
            box = ((float(c_x), float(c_y)), (float(rw), float(rh)), float(ang))
            pts = cv2.boxPoints(box).astype(np.int32)
            cv2.drawContours(canvas, [pts], 0, color, 2)
        else:
            continue
    return canvas


def pick_first_raw_image(class_dir: Path) -> Path | None:
    """Return the first sorted raw image path in a class directory, if any.

    Args:
        class_dir: Path such as ``data/raw/no_fault``.

    Returns:
        Path to an image or ``None`` if none exist.
    """

    images = aug.discover_raw_images(class_dir)
    return images[0] if images else None


def resize_to_height(image_bgr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize an image to a target height, preserving aspect ratio.

    Args:
        image_bgr: BGR image.
        target_h: Desired height in pixels.

    Returns:
        Resized image.
    """

    h, w = image_bgr.shape[:2]
    if h == 0:
        return image_bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def build_side_by_side(
    original_bgr: np.ndarray,
    original_labels: str,
    augmented_bgr: np.ndarray,
    augmented_label_lines: list[str],
    target_h: int,
) -> np.ndarray:
    """Create a horizontal concatenation of annotated original and augmented views.

    Args:
        original_bgr: Original BGR image.
        original_labels: Raw label file contents.
        augmented_bgr: Augmented BGR image.
        augmented_label_lines: Augmented label lines as strings.
        target_h: Resize height for each panel before concatenation.

    Returns:
        A single BGR image with both panels side-by-side.
    """

    left = draw_yolo_obb_lines(original_bgr, original_labels, CLASS_COLORS_BGR)
    right = draw_yolo_obb_lines(augmented_bgr, "\n".join(augmented_label_lines), CLASS_COLORS_BGR)
    left_r = resize_to_height(left, target_h)
    right_r = resize_to_height(right, target_h)
    max_w = max(left_r.shape[1], right_r.shape[1])
    def pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
        h_i, w_i = img.shape[:2]
        if w_i == width:
            return img
        out = np.zeros((h_i, width, 3), dtype=np.uint8)
        out[:, :w_i] = img
        return out

    left_p = pad_to_width(left_r, max_w)
    right_p = pad_to_width(right_r, max_w)
    return np.concatenate([left_p, right_p], axis=1)


def run_preview(target_panel_h: int = 360) -> Path:
    """Generate ``preview_augmentation.png`` for six classes.

    Args:
        target_panel_h: Panel height used when assembling the comparison grid.

    Returns:
        Path to the written preview image.
    """

    transform = aug.build_augmentation_pipeline()
    tiles: list[np.ndarray] = []

    for folder_name, fault_class in aug.CLASS_FOLDERS:
        class_dir = aug.RAW_ROOT / folder_name
        image_path = pick_first_raw_image(class_dir)
        if image_path is None:
            LOGGER.warning("No raw image for class folder %s; using blank tile.", folder_name)
            blank = np.zeros((target_panel_h, target_panel_h * 2, 3), dtype=np.uint8)
            tiles.append(blank)
            continue

        label_path = aug.resolve_label_path(image_path)
        if not label_path.is_file():
            LOGGER.warning("Missing label for %s; skipping class %s.", image_path, folder_name)
            blank = np.zeros((target_panel_h, target_panel_h * 2, 3), dtype=np.uint8)
            tiles.append(blank)
            continue

        label_text = label_path.read_text(encoding="utf-8", errors="replace")
        image_bgr = aug.read_image_bgr(image_path)
        if image_bgr is None:
            LOGGER.error("Failed to read %s; skipping class %s.", image_path, folder_name)
            blank = np.zeros((target_panel_h, target_panel_h * 2, 3), dtype=np.uint8)
            tiles.append(blank)
            continue

        obb_objects = [] if label_text.strip() == "" else aug.parse_obb_label_lines(label_text)
        seed = aug.derive_augmentation_seed(fault_class, image_path, 0)
        out = aug.augment_image_with_labels(image_bgr, obb_objects, transform, seed)
        if out is None:
            LOGGER.warning("Augmentation failed for %s; using original pair.", image_path)
            aug_bgr = image_bgr.copy()
            lines_out = [ln.strip() for ln in label_text.splitlines() if ln.strip()]
        else:
            aug_bgr, lines_out = out

        tile = build_side_by_side(image_bgr, label_text, aug_bgr, lines_out, target_panel_h)
        tiles.append(tile)

    # Arrange 3 columns x 2 rows.
    row0 = np.concatenate(tiles[0:3], axis=1)
    row1 = np.concatenate(tiles[3:6], axis=1)
    max_w = max(row0.shape[1], row1.shape[1])
    def pad_width(img: np.ndarray) -> np.ndarray:
        h_i, w_i = img.shape[:2]
        out = np.zeros((h_i, max_w, 3), dtype=np.uint8)
        out[:, :w_i] = img
        return out

    grid = np.concatenate([pad_width(row0), pad_width(row1)], axis=0)
    if not cv2.imwrite(str(PREVIEW_PATH), grid):
        raise RuntimeError(f"Failed to write preview image to {PREVIEW_PATH}")
    return PREVIEW_PATH


def main() -> None:
    """Entry point: write ``preview_augmentation.png`` in the project root."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    out = run_preview()
    print(f"Wrote preview grid to {out}")


if __name__ == "__main__":
    main()
