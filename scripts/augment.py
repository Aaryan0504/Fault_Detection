"""Augment raw transformer fault images and labels for YOLO detect training."""

from __future__ import annotations

import argparse
import gc
import inspect
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
AUG_ROOT = PROJECT_ROOT / "data" / "augmented"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

RAW_IMAGES_ROOT = RAW_ROOT / "Images"
RAW_LABELS_ROOT = RAW_ROOT / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Default augmentation multiplier for the *training* split.
# Total training images written will be approximately (1 + aug_per_image) * N_train_raw.
DEFAULT_AUG_PER_IMAGE = 5
RANDOM_STATE = 42

# Raw photos are often very high resolution; Albumentations allocates multiple
# full-size buffers per step (flip, rotate, warp). Cap the long edge before
# augmenting to avoid OpenCV (-4: Insufficient memory).
MAX_INPUT_LONG_EDGE = 2048


def get_project_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute path to the project root (parent of ``scripts/``).
    """

    return PROJECT_ROOT


def _build_gauss_noise(p: float) -> Any:
    """Build a GaussNoise transform compatible with Albumentations 1.x and 2.x.

    Args:
        p: Probability of applying the transform.

    Returns:
        An Albumentations GaussNoise transform instance.
    """

    sig = inspect.signature(A.GaussNoise.__init__)
    if "var_limit" in sig.parameters:
        return A.GaussNoise(var_limit=(10, 50), p=p)
    std_lo = math.sqrt(10.0) / 255.0
    std_hi = math.sqrt(50.0) / 255.0
    return A.GaussNoise(std_range=(std_lo, std_hi), mean_range=(0.0, 0.0), p=p)


def build_augmentation_pipeline() -> A.Compose:
    """Create the Albumentations pipeline specified for this project.

    Uses YOLO axis-aligned bboxes (``[cx, cy, w, h]`` normalized).

    Returns:
        A composed Albumentations transform.
    """

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE, p=0.7),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
                p=0.6,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1.0), p=0.4),
            _build_gauss_noise(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True,
            check_each_transform=False,
        ),
    )


def parse_label_lines(text: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse raw label text into YOLO detect bboxes.

    This project is now **detect-only**: raw labels must be in 5-field YOLO format:
    ``class cx cy w h`` (all normalized to [0, 1]).

    Any rotated-box label line (legacy 6-field or 9-field quadrilateral formats) is treated
    as a hard error to avoid silently generating incorrect training labels.

    Args:
        text: Full contents of a label ``.txt`` file.

    Returns:
        List of ``(class_id, (cx, cy, w, h))`` with values normalized to ``[0, 1]``.
    """

    objects: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 5:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            objects.append((cls_id, (cx, cy, w, h)))
        else:
            raise ValueError(
                "Unsupported label format: expected 5 values per line (cls cx cy w h). "
                f"Got {len(parts)} values: '{line}'."
            )
    return objects


def read_image_bgr(path: Path) -> np.ndarray | None:
    """Read a BGR image from disk using OpenCV.

    Args:
        path: Path to an image file.

    Returns:
        A ``uint8`` BGR image array, or ``None`` if reading failed.
    """

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def downscale_bgr_if_needed(image_bgr: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Uniformly resize so the longest side is at most ``max_long_edge`` pixels.

    Used to limit RAM during augmentation. YOLO labels are normalized, so they
    remain correct after proportional downscaling without editing coordinates.

    Args:
        image_bgr: Input image in BGR ``uint8`` format.
        max_long_edge: Maximum allowed length of the longer image side.

    Returns:
        Possibly resized image; returns the input unchanged if already smaller.
    """

    if max_long_edge <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image_bgr
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def derive_augmentation_seed(fault_class: int, image_path: Path, aug_index: int) -> int:
    """Derive a deterministic RNG seed for one augmentation call.

    Args:
        fault_class: Fault class id ``0..5``.
        image_path: Source image path.
        aug_index: Augmentation repetition index.

    Returns:
        A non-negative 32-bit integer seed.
    """

    name_hash = hash(image_path.name) % 1_000_003
    return int((RANDOM_STATE + fault_class * 1_000_003 + name_hash + aug_index * 17) % (2**31))


def augment_image_with_labels(
    image_bgr: np.ndarray,
    objects: list[tuple[int, tuple[float, float, float, float]]],
    transform: A.Compose,
    rng_seed: int,
) -> tuple[np.ndarray, list[str]] | None:
    """Run the augmentation pipeline and produce new YOLO detect label lines.

    Args:
        image_bgr: Source image in BGR format.
        objects: List of ``(class_id, (cx, cy, w, h))`` normalized; may be empty for background images.
        transform: Albumentations compose pipeline.
        rng_seed: Seed controlling stochastic transforms for this call.

    Returns:
        ``(augmented_bgr, label_lines)``, or ``None`` if all objects were dropped.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    bboxes: list[list[float]] = []
    class_labels: list[int] = []

    for cls_id, (cx, cy, w_n, h_n) in objects:
        bboxes.append([float(cx), float(cy), float(w_n), float(h_n)])
        class_labels.append(int(cls_id))

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        out = transform(
            image=image_rgb,
            bboxes=bboxes,
            class_labels=class_labels,
        )
    except Exception:
        LOGGER.exception("Augmentation failed; skipping sample.")
        return None

    aug_rgb = out["image"]
    aug_h, aug_w = aug_rgb.shape[:2]
    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

    if not objects:
        return aug_bgr, []

    bboxes_out = out.get("bboxes")
    labels_out = out.get("class_labels")
    if not bboxes_out or labels_out is None or len(labels_out) != len(bboxes_out):
        LOGGER.warning("Augmentation dropped all boxes or labels mismatched; skipping sample.")
        return None

    new_lines: list[str] = []
    for (cx, cy, w_n, h_n), cls_id in zip(bboxes_out, labels_out, strict=False):
        cx_f = float(np.clip(cx, 0.0, 1.0))
        cy_f = float(np.clip(cy, 0.0, 1.0))
        w_f = float(np.clip(w_n, 0.0, 1.0))
        h_f = float(np.clip(h_n, 0.0, 1.0))
        if w_f <= 0.0 or h_f <= 0.0:
            continue
        new_lines.append(f"{int(cls_id)} {cx_f:.6f} {cy_f:.6f} {w_f:.6f} {h_f:.6f}")

    if not new_lines:
        LOGGER.warning("All boxes invalid after augmentation; skipping sample.")
        return None

    return aug_bgr, new_lines


def discover_raw_images(class_dir: Path) -> list[Path]:
    """List all raw images for one class directory.

    Args:
        class_dir: Directory containing images.

    Returns:
        Sorted list of image paths.
    """

    paths: list[Path] = []
    if not class_dir.is_dir():
        return paths
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)


def resolve_label_path(image_path: Path) -> Path:
    """Resolve raw label path for a raw image under ``data/raw/Images/<split>``.

    Args:
        image_path: Path to a raw image inside a class folder.

    Returns:
        Path to the expected label file.
    """

    split = image_path.parent.name
    return RAW_LABELS_ROOT / split / f"{image_path.stem}.txt"


def prepare_output_dirs() -> None:
    """Create (or reset) augmented output directories under ``data/augmented``."""

    if AUG_ROOT.exists():
        shutil.rmtree(AUG_ROOT)
    (AUG_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (AUG_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (AUG_ROOT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (AUG_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (AUG_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (AUG_ROOT / "labels" / "test").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml() -> None:
    """Write ``dataset.yaml`` with a path relative to the repo root (portable for git clones)."""

    # Ultralytics resolves ``path`` relative to this YAML file; forward slashes work on Windows too.
    rel_aug = (Path("data") / "augmented").as_posix()
    cfg = {
        "path": rel_aug,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": "detect",
        "nc": 6,
        "names": {
            0: "input_cable_fault",
            1: "loose_connection",
            2: "output_cable_fault",
            3: "ri_cable_mismatch",
            4: "signal_cable_mismatch",
            5: "screw_fault",
        },
    }
    DATASET_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

def list_split_images(split: str) -> list[Path]:
    """List all raw images for a split under ``data/raw/Images/<split>``."""
    split_dir = RAW_IMAGES_ROOT / split
    if not split_dir.is_dir():
        return []
    paths: list[Path] = []
    for p in split_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)

def _write_sample(image_bgr: np.ndarray, label_lines: list[str], out_img: Path, out_lbl: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_img), image_bgr):
        raise RuntimeError(f"Failed to write image {out_img}")
    out_lbl.write_text(("\n".join(label_lines) + ("\n" if label_lines else "")), encoding="utf-8")


def copy_split(split: str) -> int:
    """Copy raw split images/labels to augmented output without augmentation."""
    images = list_split_images(split)
    out_img_dir = AUG_ROOT / "images" / split
    out_lbl_dir = AUG_ROOT / "labels" / split
    written = 0
    for image_path in tqdm(images, desc=f"Copy {split}", unit="img"):
        label_path = resolve_label_path(image_path)
        if not label_path.is_file():
            LOGGER.warning("Missing label file for %s; skipping.", image_path)
            continue
        label_text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            objects = [] if label_text == "" else parse_label_lines(label_text)
        except ValueError as exc:
            raise SystemExit(f"Label format error in {label_path}: {exc}") from None
        # Preserve empty labels (background) by writing an empty file.
        label_lines = (
            [f"{cid} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}" for cid, (cx, cy, w_n, h_n) in objects]
            if objects
            else []
        )

        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            LOGGER.error("Failed to read %s; skipping.", image_path)
            continue
        image_bgr = downscale_bgr_if_needed(image_bgr, MAX_INPUT_LONG_EDGE)
        out_img = out_img_dir / f"{image_path.stem}.png"
        out_lbl = out_lbl_dir / f"{image_path.stem}.txt"
        _write_sample(image_bgr, label_lines, out_img, out_lbl)
        written += 1
    return written

def augment_train_split(transform: A.Compose, aug_per_image: int) -> int:
    """Write train split: copy originals + write augmented variants (train only)."""
    images = list_split_images("train")
    out_img_dir = AUG_ROOT / "images" / "train"
    out_lbl_dir = AUG_ROOT / "labels" / "train"
    written = 0

    for image_path in tqdm(images, desc="Augment train", unit="img"):
        label_path = resolve_label_path(image_path)
        if not label_path.is_file():
            LOGGER.warning("Missing label file for %s; skipping.", image_path)
            continue
        label_text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            objects = [] if label_text == "" else parse_label_lines(label_text)
        except ValueError as exc:
            raise SystemExit(f"Label format error in {label_path}: {exc}") from None

        base_lines = (
            [f"{cid} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}" for cid, (cx, cy, w_n, h_n) in objects]
            if objects
            else []
        )

        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            LOGGER.error("Failed to read image %s; skipping.", image_path)
            continue
        image_bgr = downscale_bgr_if_needed(image_bgr, MAX_INPUT_LONG_EDGE)
        gc.collect()

        # 1) Always write the original (normalized to PNG, labels re-written).
        out_img = out_img_dir / f"{image_path.stem}.png"
        out_lbl = out_lbl_dir / f"{image_path.stem}.txt"
        _write_sample(image_bgr, base_lines, out_img, out_lbl)
        written += 1

        # 2) Write augmented variants.
        # Use class_id from the first object if present; else use 0 for seeding.
        seed_class = int(objects[0][0]) if objects else 0
        for aug_idx in range(int(aug_per_image)):
            seed = derive_augmentation_seed(seed_class, image_path, aug_idx)
            result = augment_image_with_labels(image_bgr, objects, transform, seed)
            if result is None:
                continue
            aug_bgr, lines = result
            aug_stem = f"{image_path.stem}_aug{aug_idx:02d}"
            out_img_a = out_img_dir / f"{aug_stem}.png"
            out_lbl_a = out_lbl_dir / f"{aug_stem}.txt"
            _write_sample(aug_bgr, lines, out_img_a, out_lbl_a)
            written += 1

    return written


def main() -> None:
    """Run dataset preparation for detect: augment train only, keep val/test real."""

    parser = argparse.ArgumentParser(
        description=(
            "Augment raw images into a YOLO detect dataset under data/augmented. "
            "Train split is augmented; val/test are copied through unchanged."
        ),
    )
    parser.add_argument(
        "--aug-per-image",
        type=int,
        default=DEFAULT_AUG_PER_IMAGE,
        metavar="K",
        help=(
            "Number of augmented variants to generate per *train* image (default: "
            f"{DEFAULT_AUG_PER_IMAGE}). Total train images will be roughly (1+K)*N."
        ),
    )
    args = parser.parse_args()
    if args.aug_per_image < 0:
        raise SystemExit("--aug-per-image must be >= 0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    prepare_output_dirs()
    transform = build_augmentation_pipeline()

    if not RAW_IMAGES_ROOT.is_dir():
        raise SystemExit(f"Missing raw images folder: {RAW_IMAGES_ROOT}")
    if not RAW_LABELS_ROOT.is_dir():
        raise SystemExit(f"Missing raw labels folder: {RAW_LABELS_ROOT}")

    n_train = augment_train_split(transform, aug_per_image=args.aug_per_image)
    n_val = copy_split("val")
    n_test = copy_split("test")

    write_dataset_yaml()
    print(f"Wrote augmented dataset to {AUG_ROOT}")
    print(f"train written: {n_train} (includes originals + augmented)")
    print(f"val copied:   {n_val} (no augmentation)")
    print(f"test copied:  {n_test} (no augmentation)")


if __name__ == "__main__":
    main()
