"""Run OBB inference and save images with per-detection boxes and class labels.

Expects a trained Ultralytics YOLO OBB model and ``dataset.yaml`` for class names.
Run from project root: ``python scripts/tag_images.py --source path/to/img_or_dir``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
DEFAULT_OUT = PROJECT_ROOT / "runs" / "tagged"

CLASS_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


def resolve_ultralytics_device() -> str:
    """Return a ``device=`` value Ultralytics accepts without assuming CUDA exists."""

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_class_names(dataset_yaml: Path) -> list[str]:
    """Load ordered class names from ``dataset.yaml``."""

    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=int)]
    return list(names)


def draw_obb_predictions(
    image_bgr: np.ndarray,
    xyxyxyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Draw oriented boxes and text labels on a BGR image copy."""

    out = image_bgr.copy()
    h, w = out.shape[:2]
    n = xyxyxyxy.shape[0]
    for i in range(n):
        pts = xyxyxyxy[i].reshape(4, 2).astype(np.int32)
        cls_i = int(class_ids[i])
        color = CLASS_COLORS_BGR[cls_i % len(CLASS_COLORS_BGR)]
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        name = class_names[cls_i] if 0 <= cls_i < len(class_names) else str(cls_i)
        label = f"{name} {float(confidences[i]):.2f}"
        anchor_x = int(np.clip(np.min(pts[:, 0]), 0, w - 1))
        anchor_y = int(np.clip(np.min(pts[:, 1]) - 6, 20, h - 1))
        cv2.putText(
            out,
            label,
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def run_tagging(
    weights: Path,
    source: Path,
    out_dir: Path,
    conf: float,
    imgsz: int,
    device: str | None,
) -> int:
    """Run prediction on ``source`` and write tagged images under ``out_dir``.

    Returns:
        Number of images written.
    """

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    class_names = load_class_names(DATASET_YAML)
    dev = device if device is not None else resolve_ultralytics_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source.resolve()),
        task="obb",
        imgsz=imgsz,
        conf=conf,
        device=dev,
        save=False,
        verbose=False,
    )

    written = 0
    for result in results:
        path = Path(result.path)
        im = result.orig_img
        if im is None:
            im = cv2.imread(str(path))
        if im is None:
            LOGGER.warning("Could not read image: %s", path)
            continue
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)

        obb = result.obb
        if obb is not None and len(obb) > 0:
            xy = obb.xyxyxyxy
            if hasattr(xy, "cpu"):
                xy = xy.cpu().numpy()
            else:
                xy = np.asarray(xy)
            conf_arr = obb.conf
            if hasattr(conf_arr, "cpu"):
                conf_arr = conf_arr.cpu().numpy()
            else:
                conf_arr = np.asarray(conf_arr)
            cls_arr = obb.cls
            if hasattr(cls_arr, "cpu"):
                cls_arr = cls_arr.cpu().numpy()
            else:
                cls_arr = np.asarray(cls_arr)
            im = draw_obb_predictions(im, xy, conf_arr, cls_arr, class_names)

        out_path = out_dir / f"tagged_{path.name}"
        if not cv2.imwrite(str(out_path), im):
            LOGGER.warning("Failed to write %s", out_path)
            continue
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag images with OBB fault detections.")
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Model weights (default: {DEFAULT_WEIGHTS})",
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or directory of images.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Force device, e.g. "cpu", "0", "mps" (default: auto).',
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    n = run_tagging(
        weights=args.weights,
        source=args.source,
        out_dir=args.out_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"Wrote {n} tagged image(s) to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
