"""Phase A training: frozen backbone, OBB head adaptation for Elletromil fault detection."""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO
from ultralytics.utils.metrics import OBBMetrics

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_YAML: Path = PROJECT_ROOT / "dataset.yaml"
PHASE_A_SAVE_DIR: Path = PROJECT_ROOT / "runs" / "phase_a"
BEST_WEIGHTS: Path = PHASE_A_SAVE_DIR / "weights" / "best.pt"
RESULTS_CSV: Path = PHASE_A_SAVE_DIR / "results.csv"
SUMMARY_JSON: Path = PHASE_A_SAVE_DIR / "phase_a_summary.json"

console = Console()
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure root logging for Phase A diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def get_device_description() -> str:
    """Return a human-readable device string for CUDA, MPS, or CPU.

    Returns:
        str: One of ``"cuda"``, ``"cuda:0"``, ``"mps"``, or ``"cpu"`` as appropriate.
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_ultralytics_device() -> str:
    """Return a ``device=`` value Ultralytics accepts (avoids ``auto`` when CUDA is missing).

    ``device=auto`` assumes a CUDA build and visible GPUs; CPU-only wheels (``torch+cpu``) raise
    ``ValueError`` instead of falling back.

    Returns:
        ``"0"`` for the first CUDA device, ``"mps"`` on Apple Silicon when available, else ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_class_names(dataset_yaml: Path) -> list[str]:
    """Load ordered class names from a YOLO ``dataset.yaml``.

    Args:
        dataset_yaml: Path to the dataset configuration file.

    Returns:
        List of class names sorted by ascending class index.

    Raises:
        FileNotFoundError: If ``dataset_yaml`` does not exist.
        KeyError: If the ``names`` mapping is missing from the YAML.
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
    """Infer the number of completed epochs from a YOLO ``results.csv`` if present.

    Args:
        results_csv: Path to the training results CSV.

    Returns:
        Number of data rows (epochs) in the CSV, or ``0`` if the file is missing or empty.
    """
    if not results_csv.is_file():
        return 0
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def run_phase_a_training(dry_run: bool) -> None:
    """Execute Phase A training or perform a dry-run configuration check.

    Args:
        dry_run: If ``True``, log configuration and exit without training.

    Raises:
        SystemExit: With code ``1`` on training failure.
    """
    class_names = load_class_names(DATASET_YAML)
    device_hint = get_device_description()
    train_device = resolve_ultralytics_device()
    if train_device == "cpu":
        logger.info(
            "Using CPU for training (no CUDA in this PyTorch install or no GPU). "
            "Install a CUDA-enabled torch build from https://pytorch.org/get-started/locally/ to use GPU."
        )

    table = Table(title="Phase A — Training summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Model variant", "yolo26s-obb (COCO OBB pretrained)")
    table.add_row("Task", "OBB (Oriented Bounding Box)")
    table.add_row("Classes", f"6 — {', '.join(class_names)}")
    table.add_row("Frozen layers", "first 10")
    table.add_row("Epochs", "20")
    table.add_row("Batch size", "16")
    table.add_row("Image size", "640")
    table.add_row("Optimizer", "MuSGD")
    table.add_row("Device", f"{device_hint} (train device={train_device})")

    train_kwargs: dict[str, Any] = {
        "data": "dataset.yaml",
        "epochs": 20,
        "imgsz": 640,
        "batch": 16,
        "freeze": 10,
        "task": "obb",
        "cfg": "configs/yolo26s_finetune.yaml",
        "project": "runs",
        "name": "phase_a",
        "exist_ok": True,
        "val": True,
        "plots": True,
        "save": True,
        "device": train_device,
        "save_dir": str(PHASE_A_SAVE_DIR),
    }

    logger.info("Train kwargs: %s", train_kwargs)
    console.print(table)

    if dry_run:
        logger.info("Dry-run: exiting before training.")
        return

    if BEST_WEIGHTS.is_file():
        ans = input("Phase A weights found. Resume? [y/n] ").strip().lower()
        if ans in {"y", "yes"}:
            logger.info("Skipping training; using existing weights at %s", BEST_WEIGHTS)
            metrics = _metrics_from_weights(BEST_WEIGHTS)
            epochs = parse_epochs_trained(RESULTS_CSV)
            _finalize_phase_a(metrics, epochs, skipped_training=True)
            return

    # yolo26s.pt is detect (xyxy); OBB data needs yolo26s-obb.pt or loss splits cls+4 vs 6-dim targets.
    model = YOLO("yolo26s-obb.pt")
    try:
        metrics = model.train(**train_kwargs)
    except Exception:
        logger.error("Phase A training failed.\n%s", traceback.format_exc())
        raise SystemExit(1) from None

    if not isinstance(metrics, OBBMetrics) or metrics.box.all_ap is None or len(metrics.box.all_ap) == 0:
        logger.warning("Final metrics missing; running validation on best weights.")
        metrics = _metrics_from_weights(BEST_WEIGHTS)

    epochs = parse_epochs_trained(RESULTS_CSV)
    if epochs == 0 and hasattr(model, "trainer") and model.trainer is not None:
        epochs = int(getattr(model.trainer, "epoch", 0)) + 1

    _finalize_phase_a(metrics, epochs, skipped_training=False)


def _metrics_from_weights(weights: Path) -> OBBMetrics:
    """Run ``model.val`` on given weights and return OBB metrics.

    Args:
        weights: Path to a ``.pt`` checkpoint.

    Returns:
        OBB metrics from validation.

    Raises:
        SystemExit: If validation fails.
    """
    m = YOLO(str(weights))
    try:
        m.val(
            data=str(DATASET_YAML.resolve()),
            task="obb",
            imgsz=640,
            batch=16,
            split="val",
            plots=False,
            save_dir=str((PHASE_A_SAVE_DIR / "val_resume").resolve()),
        )
    except Exception:
        logger.error("Validation on existing weights failed.\n%s", traceback.format_exc())
        raise SystemExit(1) from None
    out = m.metrics
    if not isinstance(out, OBBMetrics):
        raise SystemExit("Expected OBBMetrics from validation.")
    return out


def _finalize_phase_a(metrics: OBBMetrics, epochs_trained: int, skipped_training: bool) -> None:
    """Print Phase A outcomes, optional warning, and write ``phase_a_summary.json``.

    Args:
        metrics: Validation metrics object after Phase A (or from resumed weights).
        epochs_trained: Number of epochs recorded or inferred.
        skipped_training: Whether training was skipped due to user resume choice.
    """
    best_map50 = float(metrics.box.map50)
    best_map50_95 = float(metrics.box.map)
    best_path = BEST_WEIGHTS if BEST_WEIGHTS.is_file() else PHASE_A_SAVE_DIR / "weights" / "last.pt"

    console.print(f"[bold green]Best weights:[/bold green] {best_path}")
    console.print(f"[bold]Best mAP50:[/bold] {best_map50:.4f}  |  [bold]Best mAP50-95:[/bold] {best_map50_95:.4f}")

    if best_map50 < 0.5:
        console.print(
            "[yellow]Warning:[/yellow] mAP50 < 0.5 — verify label alignment and consider stronger augmentation."
        )

    summary: dict[str, Any] = {
        "best_weights_path": str(best_path.resolve()),
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "epochs_trained": epochs_trained,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skipped_training": skipped_training,
    }
    atomic_write_json(SUMMARY_JSON, summary)
    logger.info("Wrote summary to %s", SUMMARY_JSON)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase A.

    Returns:
        Parsed namespace with ``dry_run`` flag.
    """
    p = argparse.ArgumentParser(description="Phase A: train OBB head with frozen backbone (YOLO26-S).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training.",
    )
    return p.parse_args()


def main() -> None:
    """CLI entrypoint for Phase A training."""
    _configure_logging()
    args = parse_args()
    run_phase_a_training(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
