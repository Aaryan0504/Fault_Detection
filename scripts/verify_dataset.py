"""Verify augmented YOLO OBB dataset integrity and write a JSON report."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageFile

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
REPORT_PATH = PROJECT_ROOT / "verification_report.json"

ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass
class CheckResult:
    """Result of a single verification check.

    Attributes:
        name: Human-readable check name.
        passed: Whether the check succeeded.
        details: Structured details for reporting.
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def load_dataset_config(yaml_path: Path) -> dict[str, Any]:
    """Load ``dataset.yaml`` as a dictionary.

    Args:
        yaml_path: Path to ``dataset.yaml``.

    Returns:
        Parsed YAML mapping.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """

    if not yaml_path.is_file():
        raise FileNotFoundError(str(yaml_path))
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_dataset_root(cfg: dict[str, Any], yaml_path: Path) -> Path:
    """Resolve the on-disk dataset root using ``pathlib``.

    Args:
        cfg: Parsed ``dataset.yaml`` content.
        yaml_path: Path to the YAML file (anchor for relative paths).

    Returns:
        Absolute dataset root directory.
    """

    raw_path = Path(str(cfg["path"]))
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (yaml_path.parent / raw_path).resolve()


def train_to_labels_relative(train_relative: Path) -> Path:
    """Map ``images/<split>`` to ``labels/<split>``.

    Args:
        train_relative: Relative path such as ``images/train``.

    Returns:
        Relative labels path such as ``labels/train``.
    """

    parts = train_relative.parts
    if not parts or parts[0] != "images":
        raise ValueError(f"Expected train/val/test paths under images/, got {train_relative}")
    return Path("labels") / Path(*parts[1:])


def check_paths_exist(cfg: dict[str, Any], dataset_root: Path) -> CheckResult:
    """Verify dataset root and split directories exist on disk.

    Args:
        cfg: Parsed ``dataset.yaml``.
        dataset_root: Resolved dataset root.

    Returns:
        ``CheckResult`` describing success and any missing paths.
    """

    missing: list[str] = []
    if not dataset_root.is_dir():
        missing.append(str(dataset_root))
    for key in ("train", "val", "test"):
        rel = Path(str(cfg[key]))
        abs_dir = (dataset_root / rel).resolve()
        if not abs_dir.is_dir():
            missing.append(str(abs_dir))
        lbl_rel = train_to_labels_relative(rel)
        abs_lbl = (dataset_root / lbl_rel).resolve()
        if not abs_lbl.is_dir():
            missing.append(str(abs_lbl))
    passed = len(missing) == 0
    return CheckResult(
        name="paths_exist",
        passed=passed,
        details={"missing": missing, "dataset_root": str(dataset_root)},
    )


def list_split_files(dataset_root: Path, images_rel: Path) -> tuple[list[Path], Path]:
    """List image files for a split and return the labels directory path.

    Args:
        dataset_root: Dataset root directory.
        images_rel: Relative images directory for the split.

    Returns:
        ``(image_paths, labels_dir)`` where ``labels_dir`` is resolved.
    """

    img_dir = (dataset_root / images_rel).resolve()
    lbl_rel = train_to_labels_relative(images_rel)
    lbl_dir = (dataset_root / lbl_rel).resolve()
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if not img_dir.is_dir():
        return [], lbl_dir
    images = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    return images, lbl_dir


def check_counts_match(cfg: dict[str, Any], dataset_root: Path) -> CheckResult:
    """Confirm each split has a 1:1 pairing of images and label files.

    Args:
        cfg: Parsed ``dataset.yaml``.
        dataset_root: Resolved dataset root.

    Returns:
        ``CheckResult`` with per-split counts and mismatches.
    """

    per_split: dict[str, Any] = {}
    mismatches: list[str] = []
    for split_key in ("train", "val", "test"):
        images_rel = Path(str(cfg[split_key]))
        images, lbl_dir = list_split_files(dataset_root, images_rel)
        stems = {p.stem for p in images}
        label_files: dict[str, Path] = {}
        if lbl_dir.is_dir():
            label_files = {p.stem: p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
        missing_labels = sorted(stems - set(label_files.keys()))
        extra_labels = sorted(set(label_files.keys()) - stems)
        if missing_labels or extra_labels:
            mismatches.append(
                f"{split_key}: missing_labels={len(missing_labels)} extra_labels={len(extra_labels)}",
            )
        per_split[split_key] = {
            "images": len(images),
            "labels": len(label_files),
            "missing_label_stems": missing_labels[:50],
            "extra_label_stems": extra_labels[:50],
        }
    passed = len(mismatches) == 0
    return CheckResult(
        name="image_label_counts",
        passed=passed,
        details={"per_split": per_split, "mismatches": mismatches},
    )


def validate_label_line(
    line: str,
    line_no: int,
    path: Path,
) -> tuple[bool, list[str]]:
    """Validate one Ultralytics OBB line (``cls + 8 xy`` corners, normalized).

    Args:
        line: One line from a label file.
        line_no: 1-based line number for messages.
        path: Label file path for messages.

    Returns:
        ``(ok, issues)`` where ``issues`` contains human-readable problems.
    """

    issues: list[str] = []
    parts = line.strip().split()
    if len(parts) != 9:
        issues.append(f"{path}:{line_no}: expected 9 values (cls + 8 xy), got {len(parts)}")
        return False, issues
    try:
        cls_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        issues.append(f"{path}:{line_no}: non-numeric values")
        return False, issues
    if cls_id < 0 or cls_id > 5:
        issues.append(f"{path}:{line_no}: class_id {cls_id} not in [0, 5]")
    for j, val in enumerate(coords):
        if val < 0.0 or val > 1.0:
            issues.append(f"{path}:{line_no}: coord[{j}]={val} not in [0, 1]")
    return len(issues) == 0, issues


def check_label_syntax(cfg: dict[str, Any], dataset_root: Path) -> CheckResult:
    """Parse every label file and validate OBB constraints.

    Args:
        cfg: Parsed ``dataset.yaml``.
        dataset_root: Resolved dataset root.

    Returns:
        ``CheckResult`` with counts and example failures.
    """

    bad_lines = 0
    total_lines = 0
    examples: list[str] = []
    for split_key in ("train", "val", "test"):
        images_rel = Path(str(cfg[split_key]))
        _, lbl_dir = list_split_files(dataset_root, images_rel)
        if not lbl_dir.is_dir():
            continue
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            text = lbl_path.read_text(encoding="utf-8", errors="replace")
            if text.strip() == "":
                continue
            for i, raw_line in enumerate(text.splitlines(), start=1):
                if not raw_line.strip():
                    continue
                total_lines += 1
                ok, issues = validate_label_line(raw_line, i, lbl_path)
                if not ok:
                    bad_lines += 1
                    if len(examples) < 25:
                        examples.extend(issues)
    passed = bad_lines == 0
    return CheckResult(
        name="label_syntax",
        passed=passed,
        details={"total_lines": total_lines, "bad_lines": bad_lines, "examples": examples},
    )


def check_images_readable(cfg: dict[str, Any], dataset_root: Path) -> CheckResult:
    """Detect unreadable or corrupted images using Pillow.

    Args:
        cfg: Parsed ``dataset.yaml``.
        dataset_root: Resolved dataset root.

    Returns:
        ``CheckResult`` listing corrupted files.
    """

    corrupted: list[str] = []
    checked = 0
    for split_key in ("train", "val", "test"):
        images_rel = Path(str(cfg[split_key]))
        images, _ = list_split_files(dataset_root, images_rel)
        for img_path in images:
            checked += 1
            try:
                with Image.open(img_path) as im:
                    im.load()
                    _ = im.convert("RGB")
            except Exception as exc:  # noqa: BLE001 - collect any PIL/read errors
                LOGGER.warning("Unreadable image %s: %s", img_path, exc)
                corrupted.append(str(img_path))
    passed = len(corrupted) == 0
    return CheckResult(
        name="images_readable",
        passed=passed,
        details={"checked": checked, "corrupted": corrupted},
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the verification report JSON to disk.

    Args:
        path: Output JSON path (typically project root).
        payload: Serializable report dictionary.
    """

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    """Run all verification checks and emit console + JSON output."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    checks: list[CheckResult] = []

    try:
        cfg = load_dataset_config(DATASET_YAML)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        payload = {"overall_pass": False, "error": str(exc)}
        write_report(REPORT_PATH, payload)
        print("FAIL: dataset.yaml missing")
        return

    dataset_root = resolve_dataset_root(cfg, DATASET_YAML)

    c1 = check_paths_exist(cfg, dataset_root)
    checks.append(c1)
    print(
        f"{'PASS' if c1.passed else 'FAIL'}: {c1.name} "
        f"(missing_paths={len(c1.details.get('missing', []))})",
    )

    c2 = check_counts_match(cfg, dataset_root)
    checks.append(c2)
    mismatches = c2.details.get("mismatches", [])
    print(f"{'PASS' if c2.passed else 'FAIL'}: {c2.name} (mismatch_reports={len(mismatches)})")

    c3 = check_label_syntax(cfg, dataset_root)
    checks.append(c3)
    print(
        f"{'PASS' if c3.passed else 'FAIL'}: {c3.name} "
        f"(lines={c3.details.get('total_lines')} bad={c3.details.get('bad_lines')})",
    )

    c4 = check_images_readable(cfg, dataset_root)
    checks.append(c4)
    print(
        f"{'PASS' if c4.passed else 'FAIL'}: {c4.name} "
        f"(checked={c4.details.get('checked')} corrupted={len(c4.details.get('corrupted', []))})",
    )

    overall = all(c.passed for c in checks)
    payload = {
        "overall_pass": overall,
        "dataset_root": str(dataset_root),
        "checks": [
            {"name": c.name, "passed": c.passed, "details": c.details} for c in checks ],
    }
    write_report(REPORT_PATH, payload)
    print(f"Overall: {'PASS' if overall else 'FAIL'} (report: {REPORT_PATH})")


if __name__ == "__main__":
    main()
