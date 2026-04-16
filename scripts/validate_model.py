"""Post-training validation: OBB metrics, confusion matrix, and pass/fail reporting."""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text
from ultralytics import YOLO
from ultralytics.utils.metrics import OBBMetrics

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
EVAL_DIR: Path = PROJECT_ROOT / "runs" / "evaluation"
DEFAULT_WEIGHTS: Path = PROJECT_ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DATASET_YAML: Path = PROJECT_ROOT / "dataset.yaml"

console = Console()
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def resolve_ultralytics_device() -> str:
    """Return a ``device=`` value Ultralytics accepts without assuming CUDA exists.

    Returns:
        ``"0"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_class_names(dataset_yaml: Path) -> list[str]:
    """Load ordered class names from ``dataset.yaml``.

    Args:
        dataset_yaml: Path to dataset YAML.

    Returns:
        Class names ordered by index.

    Raises:
        FileNotFoundError: If the file is missing.
        KeyError: If ``names`` is not present.
    """
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    if isinstance(names, list):
        return list(names)
    raise KeyError("dataset.yaml must define names.")


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via a temporary file.

    Args:
        path: Destination path.
        content: Full file contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically.

    Args:
        path: Destination ``.json`` path.
        payload: Serializable dictionary.
    """
    atomic_write_text(path, json.dumps(payload, indent=2))


def save_confusion_matrix_plot(
    metrics: OBBMetrics,
    class_names: list[str],
    out_path: Path,
) -> None:
    """Save a seaborn heatmap: rows = actual, columns = predicted (counts).

    Args:
        metrics: Metrics containing ``confusion_matrix`` from YOLO validation with ``plots=True``.
        class_names: List of class names (without background).
        out_path: Output PNG path.
    """
    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is None:
        logger.warning("No confusion matrix on metrics; skip confusion_matrix.png.")
        return
    raw = np.asarray(cm_obj.matrix, dtype=np.float64)
    # Ultralytics stores matrix[predicted, actual]; transpose for rows=actual, cols=predicted.
    display = raw.T
    labels = list(class_names) + ["background"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        display,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (counts)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fig.savefig(tmp, dpi=200)
    plt.close(fig)
    tmp.replace(out_path)


def per_class_rows_from_summary(metrics: OBBMetrics) -> list[dict[str, float | str]]:
    """Convert ``metrics.summary()`` rows to flat metric dicts.

    Args:
        metrics: OBB metrics object.

    Returns:
        List of per-class metric dictionaries.
    """
    rows_out: list[dict[str, float | str]] = []
    for row in metrics.summary(normalize=True, decimals=6):
        rows_out.append(
            {
                "class": str(row["Class"]),
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "mAP50": float(row["mAP50"]),
                "mAP50_95": float(row["mAP50-95"]),
            }
        )
    return rows_out


def save_per_class_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    """Save per-class metrics CSV atomically.

    Args:
        rows: Per-class records.
        path: Output CSV path.
    """
    df = pd.DataFrame(rows)
    buf = df.to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, buf)


def save_per_class_bar_chart(rows: list[dict[str, float | str]], path: Path) -> None:
    """Save grouped bar chart for precision, recall, and mAP50 per class.

    Args:
        rows: Per-class metric dicts.
        path: Output PNG path.
    """
    df = pd.DataFrame(rows)
    classes = df["class"].tolist()
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.8), 6))
    ax.bar(x - width, df["precision"], width, label="precision", color="#4C72B0")
    ax.bar(x, df["recall"], width, label="recall", color="#55A868")
    ax.bar(x + width, df["mAP50"], width, label="mAP50", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.set_title("Per-class precision, recall, and mAP50")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    tmp.replace(path)


def compute_verdict(rows: list[dict[str, float | str]]) -> str:
    """Compute PASS / WARN / FAIL from per-class mAP50.

    Args:
        rows: Per-class metrics including ``mAP50``.

    Returns:
        ``"PASS"``, ``"WARN"``, or ``"FAIL"``.
    """
    maps50 = [float(r["mAP50"]) for r in rows]
    if any(m < 0.60 for m in maps50):
        return "FAIL"
    if any(m < 0.70 for m in maps50):
        return "WARN"
    return "PASS"


def print_metrics_table(rows: list[dict[str, float | str]]) -> None:
    """Print a Rich table of per-class metrics; highlight low mAP50 in red.

    Args:
        rows: Per-class metric dicts.
    """
    t = Table(title="Per-class validation metrics")
    t.add_column("class", style="cyan")
    t.add_column("precision")
    t.add_column("recall")
    t.add_column("mAP50")
    t.add_column("mAP50-95")
    for r in rows:
        m50 = float(r["mAP50"])
        m5095 = float(r["mAP50_95"])
        t.add_row(
            str(r["class"]),
            f"{float(r['precision']):.4f}",
            f"{float(r['recall']):.4f}",
            Text(f"{m50:.4f}", style="red" if m50 < 0.6 else ""),
            f"{m5095:.4f}",
        )
    console.print(t)


def run_validation(weights: Path, split: str) -> None:
    """Run ``model.val`` and export evaluation artifacts.

    Args:
        weights: Checkpoint path.
        split: Dataset split name.

    Raises:
        SystemExit: On validation failure.
    """
    class_names = load_class_names(DATASET_YAML)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    try:
        model.val(
            data=str(DATASET_YAML.resolve()),
            task="obb",
            imgsz=640,
            batch=8,
            split=split,
            plots=True,
            save_dir=str(EVAL_DIR),
            device=resolve_ultralytics_device(),
        )
    except Exception:
        logger.error("Validation failed.\n%s", traceback.format_exc())
        raise SystemExit(1) from None

    metrics = model.metrics
    if not isinstance(metrics, OBBMetrics):
        logger.error("Expected OBBMetrics after validation.")
        raise SystemExit(1)

    rows = per_class_rows_from_summary(metrics)
    print_metrics_table(rows)

    save_confusion_matrix_plot(metrics, class_names, EVAL_DIR / "confusion_matrix.png")
    save_per_class_csv(rows, EVAL_DIR / "per_class_metrics.csv")
    save_per_class_bar_chart(rows, EVAL_DIR / "per_class_metrics.png")

    per_class_ap50 = {str(r["class"]): float(r["mAP50"]) for r in rows}
    per_class_ap5095 = {str(r["class"]): float(r["mAP50_95"]) for r in rows}

    speed = {k: float(v) for k, v in metrics.speed.items()}
    total_ms = sum(speed.values())

    report: dict[str, Any] = {
        "weights": str(weights.resolve()),
        "split": split,
        "overall_mAP50": float(metrics.box.map50),
        "overall_mAP50_95": float(metrics.box.map),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap5095,
        "per_class_precision": {str(r["class"]): float(r["precision"]) for r in rows},
        "per_class_recall": {str(r["class"]): float(r["recall"]) for r in rows},
        "speed_ms_per_image": speed,
        "total_ms_per_image": total_ms,
        "metrics_results_dict": {
            str(k): float(v) if hasattr(v, "__float__") else v for k, v in metrics.results_dict.items()
        },
        "verdict": compute_verdict(rows),
    }
    atomic_write_json(EVAL_DIR / "evaluation_report.json", report)

    verdict = report["verdict"]
    if verdict == "PASS":
        console.print("[bold green]Verdict: PASS[/bold green] — all classes mAP50 ≥ 0.70.")
    elif verdict == "WARN":
        console.print(
            "[bold yellow]Verdict: WARN[/bold yellow] — all classes mAP50 ≥ 0.60 but some below 0.70."
        )
    else:
        console.print(
            "[bold red]Verdict: FAIL[/bold red] — at least one class has mAP50 < 0.60."
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Namespace with ``weights`` and ``split``.
    """
    p = argparse.ArgumentParser(description="Validate YOLO OBB model and export evaluation pack.")
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to weights (default: runs/phase_b/weights/best.pt).",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entrypoint for validation."""
    _configure_logging()
    args = parse_args()
    run_validation(weights=args.weights.resolve(), split=args.split)


if __name__ == "__main__":
    main()
