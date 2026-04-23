"""Phase B training: full fine-tune of YOLO26-S OBB model for Elletromil fault detection."""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO
from ultralytics.utils.metrics import OBBMetrics

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_PHASE_A_WEIGHTS: Path = PROJECT_ROOT / "runs" / "phase_a" / "weights" / "best.pt"
PHASE_B_SAVE_DIR: Path = PROJECT_ROOT / "runs" / "phase_b"
RESULTS_CSV: Path = PHASE_B_SAVE_DIR / "results.csv"
SUMMARY_JSON: Path = PHASE_B_SAVE_DIR / "phase_b_summary.json"
DATASET_YAML: Path = PROJECT_ROOT / "dataset.yaml"

console = Console()
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure root logging for Phase B diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def load_class_names(dataset_yaml: Path) -> list[str]:
    """Load ordered class names from a YOLO ``dataset.yaml``.

    Args:
        dataset_yaml: Path to the dataset configuration file.

    Returns:
        List of class names sorted by ascending class index.

    Raises:
        FileNotFoundError: If ``dataset_yaml`` does not exist.
        KeyError: If the ``names`` mapping is missing or malformed.
    """
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    if isinstance(names, list):
        return list(names)
    raise KeyError("dataset.yaml must contain a 'names' dict or list.")


def get_device_description() -> str:
    """Return a human-readable device string for CUDA, MPS, or CPU.

    Returns:
        str: Device label suitable for logging.
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_ultralytics_device() -> str:
    """Return a ``device=`` value Ultralytics accepts (avoids ``auto`` when CUDA is missing).

    Returns:
        ``"0"`` for the first CUDA device, ``"mps"`` when available, else ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON by writing to a temporary file then replacing the target.

    Args:
        path: Destination JSON path.
        payload: Serializable dictionary to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_epochs_trained(results_csv: Path) -> int:
    """Infer completed epochs from ``results.csv``.

    Args:
        results_csv: Training results CSV produced by Ultralytics.

    Returns:
        Epoch count from the CSV body, or ``0`` if unavailable.
    """
    if not results_csv.is_file():
        return 0
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def build_per_class_ap(metrics: OBBMetrics, class_names: list[str]) -> dict[str, float]:
    """Map each class name to its AP@0.5 (single-class AP vector).

    Args:
        metrics: OBB metrics after validation or training.
        class_names: Ordered class names for indices ``0 .. nc-1``.

    Returns:
        Dictionary ``class_name -> AP50``.
    """
    ap50 = metrics.box.ap50
    if isinstance(ap50, list):
        ap50_arr = np.array(ap50, dtype=np.float64)
    else:
        ap50_arr = np.asarray(ap50, dtype=np.float64)
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        if i < len(ap50_arr):
            out[name] = float(ap50_arr[i])
    return out


def run_phase_b(phase_a_weights: Path, dry_run: bool) -> None:
    """Run Phase B full fine-tuning or dry-run the configuration.

    Args:
        phase_a_weights: Path to Phase A ``best.pt`` checkpoint.
        dry_run: If ``True``, log settings and exit without training.

    Raises:
        SystemExit: If weights are missing or training fails.
    """
    class_names = load_class_names(DATASET_YAML)
    device_hint = get_device_description()
    train_device = resolve_ultralytics_device()
    if train_device == "cpu":
        logger.info(
            "Using CPU for training (no CUDA in this PyTorch install or no GPU). "
            "Install a CUDA-enabled torch from https://pytorch.org/get-started/locally/ for GPU."
        )

    table = Table(title="Phase B — Training summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Loaded from", str(phase_a_weights.resolve()))
    table.add_row("Task", "OBB")
    table.add_row("Epochs", "100")
    table.add_row("Batch size", "8")
    table.add_row("LR", "1e-4 (cosine decay to 1e-6 via lrf=0.01)")
    table.add_row("Frozen layers", "none (all unfrozen)")
    table.add_row("Early stopping patience", "15")
    table.add_row("Device", f"{device_hint} (train device={train_device})")

    train_kwargs: dict[str, Any] = {
        "data": "dataset.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "freeze": 0,
        "task": "obb",
        "cfg": "configs/yolo26s_finetune.yaml",
        "lr0": 0.0001,
        "lrf": 0.01,
        "cos_lr": True,
        "project": "runs",
        "name": "phase_b",
        "exist_ok": True,
        "val": True,
        "plots": True,
        "save": True,
        "patience": 15,
        "device": train_device,
        "save_dir": str(PHASE_B_SAVE_DIR),
    }

    logger.info("Train kwargs: %s", train_kwargs)
    console.print(table)

    if dry_run:
        logger.info("Dry-run: exiting before training.")
        return

    if not phase_a_weights.is_file():
        logger.error("Phase A weights not found at %s. Run Phase A first.", phase_a_weights)
        raise SystemExit(1)

    model = YOLO(str(phase_a_weights))
    try:
        metrics = model.train(**train_kwargs)
    except Exception:
        logger.error("Phase B training failed.\n%s", traceback.format_exc())
        raise SystemExit(1) from None

    if not isinstance(metrics, OBBMetrics) or metrics.box.all_ap is None or len(metrics.box.all_ap) == 0:
        logger.warning("Final metrics missing; validating best Phase B weights.")
        best_b = PHASE_B_SAVE_DIR / "weights" / "best.pt"
        model_b = YOLO(str(best_b))
        try:
            model_b.val(
                data=str(DATASET_YAML.resolve()),
                task="obb",
                imgsz=640,
                batch=8,
                split="val",
                plots=False,
                save_dir=str((PHASE_B_SAVE_DIR / "val_fallback").resolve()),
            )
        except Exception:
            logger.error("Fallback validation failed.\n%s", traceback.format_exc())
            raise SystemExit(1) from None
        metrics = model_b.metrics
        if not isinstance(metrics, OBBMetrics):
            raise SystemExit("Expected OBBMetrics from fallback validation.")

    best_map50 = float(metrics.box.map50)
    best_map50_95 = float(metrics.box.map)
    console.print(f"[bold]Best mAP50:[/bold] {best_map50:.4f}  |  [bold]Best mAP50-95:[/bold] {best_map50_95:.4f}")

    per_class = build_per_class_ap(metrics, class_names)
    _print_per_class_ap_table(per_class)

    low = [c for c, ap in per_class.items() if ap < 0.6]
    if low:
        console.print(
            "[yellow]Recommendation:[/yellow] Classes with AP@0.5 < 0.6: "
            f"{', '.join(low)} — collect more images, audit labels, or add class-specific augmentation."
        )
    else:
        console.print("[green]All classes have AP@0.5 ≥ 0.6.[/green]")

    epochs = parse_epochs_trained(RESULTS_CSV)
    if epochs == 0 and hasattr(model, "trainer") and model.trainer is not None:
        epochs = int(getattr(model.trainer, "epoch", 0)) + 1

    best_weights = PHASE_B_SAVE_DIR / "weights" / "best.pt"
    summary: dict[str, Any] = {
        "best_weights_path": str(best_weights.resolve()) if best_weights.is_file() else "",
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "per_class_ap": per_class,
        "total_epochs_trained": epochs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(SUMMARY_JSON, summary)
    logger.info("Wrote summary to %s", SUMMARY_JSON)


def _print_per_class_ap_table(per_class: dict[str, float]) -> None:
    """Print per-class AP@0.5 in a Rich table.

    Args:
        per_class: Mapping of class name to AP50.
    """
    t = Table(title="Per-class AP@0.5")
    t.add_column("Class", style="cyan")
    t.add_column("AP50", style="white")
    for name, ap in per_class.items():
        t.add_row(name, f"{ap:.4f}")
    console.print(t)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase B.

    Returns:
        Namespace with ``phase_a_weights`` path and ``dry_run`` flag.
    """
    p = argparse.ArgumentParser(description="Phase B: full fine-tune of OBB model from Phase A weights.")
    p.add_argument(
        "--phase-a-weights",
        type=Path,
        default=DEFAULT_PHASE_A_WEIGHTS,
        help="Path to Phase A best.pt (default: runs/phase_a/weights/best.pt).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training.",
    )
    return p.parse_args()


def main() -> None:
    """CLI entrypoint for Phase B training."""
    _configure_logging()
    args = parse_args()
    run_phase_b(phase_a_weights=args.phase_a_weights.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
