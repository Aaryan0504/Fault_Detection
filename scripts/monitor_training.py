"""Live training monitor: plots YOLO ``results.csv`` metrics on a fixed interval."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging for the training monitor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def read_total_epochs(run_dir: Path) -> int | None:
    """Read planned epoch count from ``args.yaml`` in the run directory if available.

    Args:
        run_dir: Ultralytics run directory (contains ``args.yaml``).

    Returns:
        The ``epochs`` field from args, or ``None`` if missing.
    """
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        return None
    try:
        with args_yaml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        e = cfg.get("epochs")
        return int(e) if e is not None else None
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        logger.debug("Could not parse epochs from args.yaml", exc_info=True)
        return None


def load_results_csv(results_path: Path) -> pd.DataFrame | None:
    """Load ``results.csv`` if it exists and is non-empty.

    Args:
        results_path: Path to ``results.csv``.

    Returns:
        DataFrame or ``None`` if the file is absent or invalid.
    """
    if not results_path.is_file():
        return None
    try:
        df = pd.read_csv(results_path)
        if df.empty:
            return None
        return df
    except (OSError, pd.errors.ParserError):
        logger.warning("Failed to parse %s", results_path, exc_info=True)
        return None


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Return a column as float series if present.

    Args:
        df: Results dataframe.
        col: Column name.

    Returns:
        Series or ``None``.
    """
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def render_training_plot(
    df: pd.DataFrame,
    out_path: Path,
    total_epochs: int | None,
) -> tuple[int, float | None, float | None, float | None]:
    """Render the four-panel training monitor figure.

    Args:
        df: Training metrics dataframe.
        out_path: Path for ``training_monitor.png``.
        total_epochs: Planned epoch count for titles, if known.

    Returns:
        Tuple of (current_epoch, last_box_loss, last_cls_loss, last_map50).
    """
    epoch = df["epoch"].astype(int) if "epoch" in df.columns else pd.Series(range(1, len(df) + 1))
    cur_epoch = int(epoch.iloc[-1])
    y_cap = total_epochs if total_epochs is not None else cur_epoch

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training monitor (epoch {cur_epoch}/{y_cap})")

    tr_box = _safe_series(df, "train/box_loss")
    va_box = _safe_series(df, "val/box_loss")
    if tr_box is not None:
        axes[0, 0].plot(epoch, tr_box, label="train/box_loss")
    if va_box is not None:
        axes[0, 0].plot(epoch, va_box, label="val/box_loss")
    axes[0, 0].set_title("Box loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()

    tr_cls = _safe_series(df, "train/cls_loss")
    va_cls = _safe_series(df, "val/cls_loss")
    if tr_cls is not None:
        axes[0, 1].plot(epoch, tr_cls, label="train/cls_loss")
    if va_cls is not None:
        axes[0, 1].plot(epoch, va_cls, label="val/cls_loss")
    axes[0, 1].set_title("Cls loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()

    m50 = _safe_series(df, "metrics/mAP50(B)")
    m5095 = _safe_series(df, "metrics/mAP50-95(B)")
    if m50 is not None:
        axes[1, 0].plot(epoch, m50, label="mAP50(B)")
    if m5095 is not None:
        axes[1, 0].plot(epoch, m5095, label="mAP50-95(B)")
    axes[1, 0].set_title("mAP metrics")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()

    lr = None
    for key in ("lr/pg0", "lr/pg1"):
        lr = _safe_series(df, key)
        if lr is not None:
            axes[1, 1].plot(epoch, lr, label=key)
            break
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("epoch")
    if lr is not None:
        axes[1, 1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    tmp.replace(out_path)

    last_box = float(tr_box.iloc[-1]) if tr_box is not None and not tr_box.empty else None
    last_cls = float(tr_cls.iloc[-1]) if tr_cls is not None and not tr_cls.empty else None
    last_map = float(m50.iloc[-1]) if m50 is not None and not m50.empty else None
    return cur_epoch, last_box, last_cls, last_map


def monitor_loop(run_dir: Path, interval_s: float = 10.0) -> None:
    """Poll ``results.csv`` every ``interval_s`` seconds and refresh plots.

    Args:
        run_dir: Directory containing ``results.csv`` (YOLO run folder).
        interval_s: Seconds between refreshes.
    """
    results_csv = run_dir / "results.csv"
    plot_path = run_dir / "training_monitor.png"
    total_epochs = read_total_epochs(run_dir)

    logger.info("Watching %s (refresh every %ss). Ctrl+C to stop.", results_csv, interval_s)
    try:
        while True:
            df = load_results_csv(results_csv)
            if df is None:
                logger.info("Waiting for %s ...", results_csv)
                time.sleep(interval_s)
                continue
            if total_epochs is None:
                total_epochs = read_total_epochs(run_dir)
            cur, box_l, cls_l, m50 = render_training_plot(df, plot_path, total_epochs)
            y_cap = total_epochs if total_epochs is not None else "?"
            box_s = f"{box_l:.4f}" if box_l is not None else "n/a"
            cls_s = f"{cls_l:.4f}" if cls_l is not None else "n/a"
            map_s = f"{m50:.4f}" if m50 is not None else "n/a"
            console.print(
                f"Epoch {cur}/{y_cap} | box_loss: {box_s} | cls_loss: {cls_s} | mAP50: {map_s}"
            )
            time.sleep(interval_s)
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace including ``run_dir``.
    """
    p = argparse.ArgumentParser(description="Live plot YOLO results.csv during training.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/phase_b"),
        help="Run directory containing results.csv (default: runs/phase_b).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (default: 10).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entrypoint for the training monitor."""
    _configure_logging()
    args = parse_args()
    run_dir = args.run_dir.resolve()
    monitor_loop(run_dir, interval_s=args.interval)


if __name__ == "__main__":
    main()
