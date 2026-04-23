"""Augment raw transformer fault images and labels for YOLO OBB training."""

from __future__ import annotations

import argparse
import gc
import inspect
import logging
import math
import random
import shutil
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
AUG_ROOT = PROJECT_ROOT / "data" / "augmented"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

CLASS_FOLDERS: list[tuple[str, int]] = [
    ("no_fault", 0),
    ("input_cable_fault", 1),
    ("loose_connection", 2),
    ("output_cable_fault", 3),
    ("signal_cable_fault", 4),
    ("screw_fault", 5),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Lower default keeps Phase B training/validation feasible on modest hardware;
# override with ``python scripts/augment.py --target-per-class N``.
DEFAULT_TARGET_PER_CLASS = 150
TRAIN_FRAC = 0.70
RANDOM_STATE = 42

# Raw photos are often very high resolution; Albumentations allocates multiple
# full-size buffers per step (flip, rotate, warp). Cap the long edge before
# augmenting to avoid OpenCV (-4: Insufficient memory). Normalized YOLO OBB
# labels stay valid under uniform scaling.
MAX_INPUT_LONG_EDGE = 2048


@dataclass(frozen=True)
class AugmentedSample:
    """One augmented image/label pair ready for disk export.

    Attributes:
        fault_class: Integer fault class id (0-5) from the source folder.
        image_bgr: Augmented image in BGR uint8, shape (H, W, 3).
        label_lines: List of YOLO OBB lines as strings without newline.
        source_image_path: Original raw image path (for traceability).
        aug_index: Augmentation index for this source image.
    """

    fault_class: int
    image_bgr: np.ndarray
    label_lines: list[str]
    source_image_path: Path
    aug_index: int


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

    Oriented boxes are represented as four corner keypoints (pixel ``xy``)
    plus an axis-aligned YOLO ``[cx, cy, w, h]`` proxy derived from the
    oriented box, with ``BboxParams`` configured as required.

    Returns:
        A composed Albumentations transform.
    """

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # BORDER_REFLECT makes Albumentations 2.x replicate keypoints on a reflection
            # grid (~9× the input points), which breaks single-OBB labels. REPLICATE avoids
            # that while still filling rotated edges sensibly.
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
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["kp_class_labels"],
            remove_invisible=False,
            check_each_transform=False,
        ),
    )


def obb_to_corner_keypoints(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Convert a normalized YOLO OBB to four corner points in pixel coordinates.

    Args:
        cx: Normalized x center in ``[0, 1]``.
        cy: Normalized y center in ``[0, 1]``.
        w: Normalized width in ``[0, 1]``.
        h: Normalized height in ``[0, 1]``.
        angle_deg: Rotation angle in degrees (OpenCV ``minAreaRect`` convention).
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Float32 array with shape ``(4, 2)`` of corner ``(x, y)`` in pixels.
    """

    c_x = float(cx) * image_width
    c_y = float(cy) * image_height
    rect_w = float(w) * image_width
    rect_h = float(h) * image_height
    rect = ((c_x, c_y), (rect_w, rect_h), float(angle_deg))
    pts = cv2.boxPoints(rect)
    return pts.astype(np.float32)


def parse_obb_label_lines(text: str) -> list[tuple[int, np.ndarray]]:
    """Parse raw label text into per-object normalized quadrilaterals.

    Supports Ultralytics OBB format (``class x1 y1 x2 y2 x3 y3 x4 y4``,9 fields)
    and legacy center format (``class cx cy w h angle``, 6 fields).

    Args:
        text: Full contents of a label ``.txt`` file.

    Returns:
        List of ``(class_id, corners)`` where ``corners`` is float32 ``(4, 2)``
        with ``xy`` normalized to ``[0, 1]``.
    """

    objects: list[tuple[int, np.ndarray]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 9:
            cls_id = int(float(parts[0]))
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
            objects.append((cls_id, coords))
        elif len(parts) == 6:
            cls_id = int(float(parts[0]))
            cx, cy, w, h, ang = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            corners = obb_to_corner_keypoints(cx, cy, w, h, ang, 1, 1)
            objects.append((cls_id, corners.astype(np.float32)))
        else:
            LOGGER.warning(
                "Skipping malformed label line (expected 9 or 6 values, got %d): %s",
                len(parts),
                line,
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
    obb_objects: list[tuple[int, np.ndarray]],
    transform: A.Compose,
    rng_seed: int,
) -> tuple[np.ndarray, list[str]] | None:
    """Run the augmentation pipeline and produce new YOLO OBB label lines.

    Args:
        image_bgr: Source image in BGR format.
        obb_objects: List of ``(class_id, corners)`` with ``corners`` shape ``(4, 2)``
            in normalized ``xy``; may be empty for background images.
        transform: Albumentations compose pipeline.
        rng_seed: Seed controlling stochastic transforms for this call.

    Returns:
        ``(augmented_bgr, label_lines)``, or ``None`` if all objects were dropped.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    bboxes: list[list[float]] = []
    class_labels: list[int] = []
    keypoints: list[list[float]] = []
    kp_class_labels: list[int] = []

    for cls_id, corners_norm in obb_objects:
        corners_px = corners_norm.astype(np.float64).copy()
        corners_px[:, 0] *= float(width)
        corners_px[:, 1] *= float(height)
        xs = corners_px[:, 0]
        ys = corners_px[:, 1]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        bcx = ((x_min + x_max) * 0.5) / float(width)
        bcy = ((y_min + y_max) * 0.5) / float(height)
        bnw = (x_max - x_min) / float(width)
        bnh = (y_max - y_min) / float(height)
        bboxes.append([bcx, bcy, bnw, bnh])
        class_labels.append(int(cls_id))
        for pt in corners_px:
            keypoints.append([float(pt[0]), float(pt[1])])
            kp_class_labels.append(int(cls_id))

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        out = transform(
            image=image_rgb,
            bboxes=bboxes,
            class_labels=class_labels,
            keypoints=keypoints,
            kp_class_labels=kp_class_labels,
        )
    except Exception:
        LOGGER.exception("Augmentation failed; skipping sample.")
        return None

    aug_rgb = out["image"]
    aug_h, aug_w = aug_rgb.shape[:2]
    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

    if not obb_objects:
        return aug_bgr, []

    kp_out = out.get("keypoints")
    labels_out = out.get("kp_class_labels")
    if not kp_out or len(kp_out) % 4 != 0:
        LOGGER.warning("Augmentation dropped or mangled keypoints; skipping sample.")
        return None
    if not labels_out or len(labels_out) != len(kp_out):
        LOGGER.warning(
            "Augmentation keypoint labels missing or length mismatch "
            "(keypoints=%d, labels=%s); skipping sample.",
            len(kp_out),
            len(labels_out) if labels_out is not None else None,
        )
        return None

    new_lines: list[str] = []
    for i in range(0, len(kp_out), 4):
        quad = np.asarray(kp_out[i : i + 4], dtype=np.float32)
        cls_kp = int(labels_out[i])
        nx = np.clip(quad[:, 0] / float(aug_w), 0.0, 1.0)
        ny = np.clip(quad[:, 1] / float(aug_h), 0.0, 1.0)
        flat = np.stack([nx, ny], axis=1).reshape(-1)
        coord_str = " ".join(f"{float(v):.6f}" for v in flat)
        new_lines.append(f"{cls_kp} {coord_str}")

    if not new_lines:
        LOGGER.warning("All OBB boxes invalid after augmentation; skipping sample.")
        return None

    return aug_bgr, new_lines


def discover_raw_images(class_dir: Path) -> list[Path]:
    """List all raw images for one class directory.

    Args:
        class_dir: Path like ``data/raw/no_fault``.

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
    """Return the sibling label path ``labels/<stem>.txt`` for a raw image.

    Args:
        image_path: Path to a raw image inside a class folder.

    Returns:
        Path to the expected label file.
    """

    return image_path.parent / "labels" / f"{image_path.stem}.txt"


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
        "task": "obb",
        "nc": 6,
        "names": {
            0: "no_fault",
            1: "input_cable_fault",
            2: "loose_connection",
            3: "output_cable_fault",
            4: "signal_cable_fault",
            5: "screw_fault",
        },
    }
    DATASET_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def augment_class_folder(
    folder_name: str,
    fault_class: int,
    transform: A.Compose,
    target_per_class: int,
) -> tuple[int, int, list[AugmentedSample]]:
    """Augment all images for one fault-class folder.

    Args:
        folder_name: Subfolder name under ``data/raw``.
        fault_class: Integer class id for this folder.
        transform: Albumentations pipeline.
        target_per_class: Cap on augmented samples for this class.

    Returns:
        ``(raw_count, target_generated, samples)`` where ``samples`` length is
        exactly ``target_per_class`` when ``raw_count > 0`` and enough augments succeed.
    """

    class_dir = RAW_ROOT / folder_name
    images = discover_raw_images(class_dir)
    raw_count = len(images)
    if raw_count == 0:
        LOGGER.error("No raw images found in %s", class_dir)
        return 0, 0, []

    multiplier = int(ceil(target_per_class / float(raw_count)))
    samples: list[AugmentedSample] = []

    for image_path in tqdm(images, desc=f"Augment {folder_name}", unit="img"):
        label_path = resolve_label_path(image_path)
        if not label_path.is_file():
            LOGGER.warning("Missing label file for %s; skipping image.", image_path)
            continue

        label_text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        if label_text == "":
            obb_objects: list[tuple[int, float, float, float, float, float]] = []
        else:
            obb_objects = parse_obb_label_lines(label_text)

        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            LOGGER.error("Failed to read image %s; skipping.", image_path)
            continue

        image_bgr = downscale_bgr_if_needed(image_bgr, MAX_INPUT_LONG_EDGE)
        gc.collect()

        for aug_idx in range(multiplier):
            seed = derive_augmentation_seed(fault_class, image_path, aug_idx)
            result = augment_image_with_labels(image_bgr, obb_objects, transform, seed)
            if result is None:
                continue
            aug_bgr, lines = result
            samples.append(
                AugmentedSample(
                    fault_class=fault_class,
                    image_bgr=aug_bgr,
                    label_lines=lines,
                    source_image_path=image_path,
                    aug_index=aug_idx,
                ),
            )

    if len(samples) > target_per_class:
        rng = np.random.default_rng(RANDOM_STATE)
        pick = rng.choice(len(samples), size=target_per_class, replace=False)
        pick_sorted = sorted(int(x) for x in pick.tolist())
        samples = [samples[i] for i in pick_sorted]
    elif len(samples) < target_per_class:
        LOGGER.warning(
            "Class %s produced only %d samples (target %d).",
            folder_name,
            len(samples),
            target_per_class,
        )

    return raw_count, len(samples), samples


def split_and_save_samples(all_samples: list[AugmentedSample]) -> dict[str, dict[str, int]]:
    """Stratify-split combined samples and write PNG images and YOLO labels.

    Args:
        all_samples: Augmented samples from all classes.

    Returns:
        Nested counts ``counts[split][class_or_total]`` for summary printing.
    """

    if not all_samples:
        return {}

    indices = np.arange(len(all_samples), dtype=np.int64)
    y = np.asarray([s.fault_class for s in all_samples], dtype=np.int64)

    idx_train, idx_temp = train_test_split(
        indices,
        test_size=(1.0 - TRAIN_FRAC),
        stratify=y,
        random_state=RANDOM_STATE,
    )
    y_temp = y[idx_temp]
    val_fraction_of_temp = 0.5
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    split_to_indices = {
        "train": idx_train.tolist(),
        "val": idx_val.tolist(),
        "test": idx_test.tolist(),
    }

    counts: dict[str, dict[str, int]] = {
        "train": {"total": 0},
        "val": {"total": 0},
        "test": {"total": 0},
    }
    for split_name in counts:
        for _, cid in CLASS_FOLDERS:
            counts[split_name][str(cid)] = 0

    global_index = 0
    for split_name, idx_list in split_to_indices.items():
        img_dir = AUG_ROOT / "images" / split_name
        lbl_dir = AUG_ROOT / "labels" / split_name
        for i in idx_list:
            sample = all_samples[int(i)]
            stem = f"sample_{global_index:06d}"
            global_index += 1
            img_path = img_dir / f"{stem}.png"
            lbl_path = lbl_dir / f"{stem}.txt"
            if not cv2.imwrite(str(img_path), sample.image_bgr):
                LOGGER.error("Failed to write image %s", img_path)
                continue
            lbl_path.write_text(
                ("\n".join(sample.label_lines) + ("\n" if sample.label_lines else "")),
                encoding="utf-8",
            )
            counts[split_name]["total"] += 1
            counts[split_name][str(sample.fault_class)] += 1

    return counts


def print_summary_table(rows: list[tuple[str, int, int, int, int, int]]) -> None:
    """Print the augmentation summary table to stdout.

    Args:
        rows: Rows of
            ``(class_name, raw_count, augmented, train, val, test)``.
    """

    header = f"{'Class':<22} | {'Raw':>5} | {'Aug':>5} | {'Train':>5} | {'Val':>5} | {'Test':>5}"
    print(header)
    print("-" * len(header))
    for name, raw_c, aug_c, tr, va, te in rows:
        print(f"{name:<22} | {raw_c:5d} | {aug_c:5d} | {tr:5d} | {va:5d} | {te:5d}")


def main() -> None:
    """Run full dataset preparation: augment, split, save, and update ``dataset.yaml``."""

    parser = argparse.ArgumentParser(
        description="Augment raw Electromil images into a YOLO OBB dataset under data/augmented.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=DEFAULT_TARGET_PER_CLASS,
        metavar="N",
        help=(
            "Maximum augmented samples per fault class after stratified split inputs "
            f"(default: {DEFAULT_TARGET_PER_CLASS})."
        ),
    )
    args = parser.parse_args()
    if args.target_per_class < 1:
        raise SystemExit("--target-per-class must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    prepare_output_dirs()
    transform = build_augmentation_pipeline()

    per_class_samples: list[tuple[str, int, int, list[AugmentedSample]]] = []
    all_samples: list[AugmentedSample] = []

    for folder_name, fault_class in CLASS_FOLDERS:
        raw_c, aug_c, samples = augment_class_folder(
            folder_name,
            fault_class,
            transform,
            target_per_class=args.target_per_class,
        )
        per_class_samples.append((folder_name, raw_c, aug_c, samples))
        all_samples.extend(samples)

    split_counts = split_and_save_samples(all_samples)

    write_dataset_yaml()

    summary_rows: list[tuple[str, int, int, int, int, int]] = []
    for folder_name, raw_c, aug_c, samples in per_class_samples:
        fault_class = next(cid for fname, cid in CLASS_FOLDERS if fname == folder_name)
        tr = split_counts.get("train", {}).get(str(fault_class), 0) if split_counts else 0
        va = split_counts.get("val", {}).get(str(fault_class), 0) if split_counts else 0
        te = split_counts.get("test", {}).get(str(fault_class), 0) if split_counts else 0
        summary_rows.append((folder_name, raw_c, aug_c, tr, va, te))

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
