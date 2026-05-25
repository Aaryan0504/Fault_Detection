# Phase 2 — Multi-modal Data Collection and Preprocessing Pipeline

> **Scope:** Backend-only. Defines how RGB, thermal/IR, and multi-angle captures enter the system, how they are aligned/fused/normalized, and what every downstream model receives. This phase consumes Phase 1's taxonomy and dataset contract and produces the curated `dataset_v2/` tree plus a runtime preprocessing service used at both training and inference time.

---

## 1. Phase objective

The demo project trains and infers on **single-frame RGB only**. The production system must:

1. Ingest **paired RGB + thermal/IR** captures with deterministic metadata (timestamp, device ID, capture view).
2. Support **multi-angle** captures of the same UPS unit in a single session.
3. Produce **calibrated, aligned, fused** images that all downstream models share a single source of truth for.
4. Reject blurry / misframed / over- or under-exposed captures **before** they reach the YOLO detector.
5. Be **identical between training and serving** — no preprocessing skew is allowed.

Deliverables:

| Deliverable | Artifact |
|---|---|
| Capture protocol spec | `docs/capture_protocol.md` (companion to this phase) |
| Thermal–RGB calibration toolkit | `scripts/preproc/calibrate_rgb_ir.py` + `calibration/<device_id>/*.json` |
| Ingestion CLI | `scripts/ingest/ingest_session.py` |
| Image quality gate (IQA) | `powervision/preproc/quality.py` |
| Preprocessing transform (training + serving) | `powervision/preproc/transforms.py` |
| Modal fusion | `powervision/preproc/fusion.py` |
| Albumentations augmentation pipelines per modality | `powervision/preproc/aug.py` |
| Curated dataset under `dataset_v2/` (populated) | DVC-tracked |
| Manifest enrichment | `dataset_v2/manifest.parquet` columns extended (Phase 1 schema) |
| FastAPI preprocessing microservice (optional) | `services/preproc/main.py` |

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Format |
|---|---|---|
| Raw RGB captures from field engineers | Smartphone / DSLR / industrial cam | JPEG/PNG, EXIF preserved |
| Raw thermal captures | FLIR, Seek Thermal, Hikmicro | Radiometric JPEG (`.jpg` with embedded RJPEG) **or** `.tiff` 16-bit + `.csv` temperature grid |
| Capture session manifest | Engineer's app upload | JSON (see §3.3) |
| Phase 1 taxonomy | `taxonomy/*.yaml` | YAML |
| Phase 1 manifest skeleton | `dataset_v2/manifest.parquet` | Parquet |

### 2.2 Outputs

```text
dataset_v2/
├── images/
│   ├── rgb/{train,val,test}/<image_id>.jpg          # 1024×1024 max, sRGB, EXIF-stripped
│   ├── thermal/{train,val,test}/<image_id>.png      # 16-bit single-channel, aligned to RGB
│   └── fused/{train,val,test}/<image_id>.png        # 4-channel PNG (RGB + thermal)
├── preview/{train,val,test}/<image_id>.jpg          # 512px JPEG for fast QA only
├── thermal_meta/{train,val,test}/<image_id>.json    # min/max/avg °C, palette used
└── manifest.parquet                                  # enriched

calibration/
└── <device_id>/
    ├── homography.json        # 3x3 H mapping thermal→RGB pixels
    ├── intrinsics_rgb.json
    ├── intrinsics_ir.json
    ├── reprojection_error.json
    └── checkerboard_samples/

logs/preproc/
└── <run_id>/
    ├── rejected.jsonl          # one row per rejected image
    └── stats.json
```

### 2.3 Format contracts

- **RGB on-disk**: JPEG quality 95, sRGB ICC, max side 1024 px (preserves training detail without inflating disk), EXIF stripped except for retained fields `{DateTimeOriginal, Make, Model, ISO, ExposureTime}` written into `manifest.parquet`.
- **Thermal on-disk**: 16-bit grayscale PNG; pixel value = `(temperature_celsius + 100) * 100` (offset to handle negative °C), recoverable via `temp_c = pixel / 100.0 - 100.0`. Calibration constants written to `thermal_meta/<image_id>.json`.
- **Fused on-disk**: 4-channel PNG (R, G, B, T) where T is the rescaled thermal channel (`uint8`, 0–255 normalized per Phase 2's calibration constants). Used by Phase 3 if and only if `model.modality == "fused"`; otherwise RGB-only consumers ignore it.

---

## 3. Capture protocol

Quality of every downstream model is bounded by data acquired here. Phase 2 ships an engineer-facing capture spec.

### 3.1 Hardware baseline

| Equipment | Minimum spec | Recommended |
|---|---|---|
| RGB camera | 12 MP, autofocus, manual exposure available | Industrial USB3 12 MP (e.g. Basler ace) or Samsung S22+ class |
| Thermal camera | Resolution ≥ 160×120, NETD ≤ 70 mK | FLIR E76 (320×240, ≤ 30 mK) or Hikmicro HM-TPK20 |
| Mount | Handheld OK for triage; tripod required for golden references | Tripod + level + ring light |
| Lighting | 800 lux minimum on subject | 1500 lux diffuse, 5500 K CRI > 90 |
| Calibration target | Heated checkerboard (7×9 inner corners, 30 mm square) | OEM heated target + flat-field plate |

### 3.2 Capture views per UPS

`taxonomy/ups_system_types.yaml` enumerates `image_capture_views` per UPS type. For each view, capture:

| Frame | Modality | Notes |
|---|---|---|
| Overview | RGB | Wide, covers entire panel; used for misframing detection only |
| Detail | RGB | Tight, parallel to panel, 30–60 cm |
| Thermal | IR | Same framing as Detail, captured within 5 s |
| Multi-angle pair | RGB | ±15° rotation around vertical axis |

Engineer's app enforces this through a capture wizard; the resulting session JSON (§3.3) is the only thing Phase 2 trusts.

### 3.3 Session manifest schema

```json
{
  "session_id": "S_20260524_0001",
  "device_id": "tenantA_devid_001",
  "ups_type_id": "ups_type_001",
  "captured_at": "2026-05-24T13:42:11+05:30",
  "engineer_id": "eng_017",
  "location": {"site": "BLR-DC-04", "rack": "R12"},
  "frames": [
    {
      "frame_id": "F001",
      "view": "front_open",
      "rgb_path": "S_20260524_0001/F001_rgb.jpg",
      "thermal_path": "S_20260524_0001/F001_ir.jpg",
      "rgb_meta": {"iso": 400, "exposure_us": 8000, "focal_mm": 4.5},
      "ir_meta": {"emissivity": 0.95, "reflected_temp_c": 25.0, "atmospheric_temp_c": 26.0}
    }
  ],
  "consent_for_training": true
}
```

The ingestion CLI (§5.1) refuses sessions that fail this schema.

---

## 4. RGB ↔ Thermal calibration

Thermal and RGB cameras have different focal lengths, principal points, and (often) parallax. Without per-device calibration, "modal fusion" is just noise.

### 4.1 One-time per-device calibration

1. Print or order a **heated checkerboard** (passive boards with reflective squares also work for thermal in many models, but heated is more robust).
2. Capture 20–30 paired RGB+IR images of the board across the lens FOV (corners, center, tilted, near, far).
3. Run:

   ```bash
   python scripts/preproc/calibrate_rgb_ir.py \
     --device-id tenantA_devid_001 \
     --pairs calibration_raw/tenantA_devid_001/ \
     --pattern 7x9 \
     --square-mm 30
   ```

4. Outputs `calibration/<device_id>/{homography.json, intrinsics_rgb.json, intrinsics_ir.json, reprojection_error.json}`.
5. Acceptance criterion: **median reprojection error ≤ 1.5 px in RGB space**; otherwise reshoot.

```python
# scripts/preproc/calibrate_rgb_ir.py (core)
import cv2, json, numpy as np
from pathlib import Path

def calibrate(rgb_imgs, ir_imgs, pattern=(7, 9), square_mm=30):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm

    rgb_pts, ir_pts, obj_pts = [], [], []
    for rp, ip in zip(rgb_imgs, ir_imgs):
        rgb = cv2.imread(str(rp)); ir = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        ok_r, c_r = cv2.findChessboardCornersSB(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), pattern)
        ok_i, c_i = cv2.findChessboardCornersSB(ir, pattern)
        if not (ok_r and ok_i):
            continue
        rgb_pts.append(c_r); ir_pts.append(c_i); obj_pts.append(objp)

    K_r, D_r, K_i, D_i = (np.zeros((3,3)) for _ in range(4))
    _, K_r, D_r, _, _ = cv2.calibrateCamera(obj_pts, rgb_pts, rgb.shape[1::-1], None, None)
    _, K_i, D_i, _, _ = cv2.calibrateCamera(obj_pts, ir_pts,  ir.shape[1::-1],  None, None)

    # Per-pair homography averaged via stereo rectification fallback.
    H_list = [cv2.findHomography(ip.reshape(-1, 2), rp.reshape(-1, 2), cv2.RANSAC)[0]
              for rp, ip in zip(rgb_pts, ir_pts)]
    H = np.median(np.stack(H_list), axis=0)

    # Reprojection error in RGB pixels
    errs = []
    for rp, ip in zip(rgb_pts, ir_pts):
        warped = cv2.perspectiveTransform(ip.reshape(-1, 1, 2), H).reshape(-1, 2)
        errs.append(np.linalg.norm(warped - rp.reshape(-1, 2), axis=1).mean())
    return K_r, D_r, K_i, D_i, H, float(np.median(errs))
```

### 4.2 Per-capture refinement (optional, fast)

For handheld captures, parallax dominates. If the device is the same but slightly handheld-misaligned, run:

```python
# powervision/preproc/align.py
def refine_alignment(rgb_gray: np.ndarray, ir_gray: np.ndarray, H0: np.ndarray) -> np.ndarray:
    """ECC-refine H0 using a coarse-to-fine pyramid. Returns refined 3x3 H."""
    warp = H0.astype(np.float32)
    try:
        _, warp = cv2.findTransformECC(
            templateImage=cv2.equalizeHist(rgb_gray),
            inputImage=cv2.equalizeHist(ir_gray),
            warpMatrix=warp,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4),
        )
    except cv2.error:
        pass
    return warp
```

---

## 5. Ingestion pipeline

### 5.1 `scripts/ingest/ingest_session.py`

End-to-end: validates session JSON → loads RGB + IR → applies calibration → runs IQA gate → writes canonical files → updates `manifest.parquet`.

```python
"""Ingest a raw capture session into dataset_v2/raw_ingest/<session_id>/.

Usage:
  python scripts/ingest/ingest_session.py \
    --session uploads/sessions/S_20260524_0001/session.json \
    --calibration-root calibration/ \
    --out dataset_v2/raw_ingest/

This script does NOT assign splits or copy into images/<split>/ yet — that
happens after labeling. It produces a normalized, aligned, quality-checked
pile of frames ready for CVAT upload.
"""
from __future__ import annotations
import argparse, json, hashlib, uuid, shutil
from pathlib import Path
from datetime import datetime
import cv2, numpy as np, pandas as pd
from PIL import Image, ExifTags

from powervision.preproc.thermal import load_radiometric, encode_thermal_png
from powervision.preproc.quality import iqa_check
from powervision.preproc.transforms import canonicalize_rgb
from powervision.preproc.align  import apply_homography
from powervision.preproc.fusion import fuse_rgb_thermal

def _read_calibration(device_id: str, root: Path) -> dict:
    p = root / device_id / "homography.json"
    if not p.exists():
        raise FileNotFoundError(f"No calibration for {device_id}. Run calibrate_rgb_ir.py first.")
    return json.loads(p.read_text())

def _image_id(session_id: str, frame_id: str) -> str:
    return f"{session_id}__{frame_id}"

def ingest(session_path: Path, calib_root: Path, out_root: Path) -> None:
    sess = json.loads(session_path.read_text())
    device_id = sess["device_id"]
    calib     = _read_calibration(device_id, calib_root)
    H         = np.asarray(calib["H"], dtype=np.float32)

    out_session = out_root / sess["session_id"]
    (out_session / "rgb").mkdir(parents=True, exist_ok=True)
    (out_session / "thermal").mkdir(parents=True, exist_ok=True)
    (out_session / "fused").mkdir(parents=True, exist_ok=True)
    (out_session / "preview").mkdir(parents=True, exist_ok=True)
    rejected: list[dict] = []

    for frame in sess["frames"]:
        image_id = _image_id(sess["session_id"], frame["frame_id"])
        rgb_in = session_path.parent / frame["rgb_path"]
        ir_in  = session_path.parent / frame["thermal_path"] if frame.get("thermal_path") else None

        rgb_bgr = cv2.imread(str(rgb_in), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            rejected.append({"image_id": image_id, "reason": "rgb_unreadable"}); continue

        rgb_canon = canonicalize_rgb(rgb_bgr, max_side=1024)
        ok, iqa = iqa_check(rgb_canon)
        if not ok:
            rejected.append({"image_id": image_id, "reason": "iqa_fail", "iqa": iqa}); continue

        thermal_aligned = None
        thermal_meta = {}
        if ir_in is not None and ir_in.exists():
            ir_array_c, palette = load_radiometric(ir_in)  # float32 in °C
            ir_aligned_c = apply_homography(ir_array_c, H, rgb_canon.shape[:2])
            thermal_aligned = encode_thermal_png(ir_aligned_c)  # uint16
            thermal_meta = {
                "min_c": float(np.nanmin(ir_aligned_c)),
                "max_c": float(np.nanmax(ir_aligned_c)),
                "mean_c": float(np.nanmean(ir_aligned_c)),
                "palette": palette,
                "emissivity": frame.get("ir_meta", {}).get("emissivity"),
            }

        cv2.imwrite(str(out_session / "rgb" / f"{image_id}.jpg"),
                    rgb_canon, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_session / "preview" / f"{image_id}.jpg"),
                    cv2.resize(rgb_canon, (512, int(512 * rgb_canon.shape[0] / rgb_canon.shape[1]))),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        if thermal_aligned is not None:
            cv2.imwrite(str(out_session / "thermal" / f"{image_id}.png"), thermal_aligned)
            fused = fuse_rgb_thermal(rgb_canon, thermal_aligned)
            cv2.imwrite(str(out_session / "fused" / f"{image_id}.png"), fused)
            (out_session / "thermal_meta").mkdir(exist_ok=True)
            (out_session / "thermal_meta" / f"{image_id}.json").write_text(json.dumps(thermal_meta))

    (out_session / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rejected))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--calibration-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ingest(args.session, args.calibration_root, args.out)

if __name__ == "__main__":
    main()
```

After ingestion, frames live at `dataset_v2/raw_ingest/<session_id>/{rgb,thermal,fused,preview}/`. CVAT pulls from `preview/` for QA and `rgb/` for labeling. After labeling, `scripts/taxonomy/cvat_to_canonical.py` (Phase 1) moves the labeled frames into `dataset_v2/images/<modality>/<split>/`.

---

## 6. Preprocessing module (`powervision/preproc/`)

This Python package is the **single** source of preprocessing logic. It is imported by training scripts (Phases 3–5), the inference service (Phase 6), and the ingestion CLI. No phase implements its own preprocessing.

### 6.1 Module layout

```text
powervision/
└── preproc/
    ├── __init__.py
    ├── thermal.py      # radiometric IO + encoding
    ├── align.py        # homography apply + ECC refine
    ├── fusion.py       # RGB + thermal → 4-channel
    ├── transforms.py   # resize, CLAHE, normalize, canonicalize
    ├── quality.py      # IQA (blur, exposure, framing)
    ├── aug.py          # Albumentations pipelines per modality
    └── pipeline.py     # high-level build_train_transform() / build_eval_transform()
```

### 6.2 `transforms.py`

```python
"""Canonical preprocessing — used identically at train and serve."""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

@dataclass(frozen=True)
class PreprocConfig:
    target_size: int = 640      # YOLO native
    clahe_clip:  float = 2.0
    clahe_grid:  int = 8
    apply_clahe: bool = True
    apply_normalize: bool = True
    pad_color:   tuple[int, int, int] = (114, 114, 114)  # YOLO default

def canonicalize_rgb(img_bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Strip EXIF rotation, ensure 3-channel BGR, resize so max(H,W) <= max_side
    while preserving aspect ratio. Used at INGEST time, not at inference."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img_bgr

def letterbox(img_bgr: np.ndarray, size: int, pad_color=(114, 114, 114)
              ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """YOLO-compatible letterbox. Returns (img, scale, (pad_x, pad_y))."""
    h, w = img_bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    pad_h, pad_w = size - nh, size - nw
    top, bot = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return img, s, (left, top)

def apply_clahe_lab(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on L channel of LAB. Robust against lighting variance in cabinets."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    L_eq = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L_eq, a, b]), cv2.COLOR_LAB2BGR)

def to_chw_normalized(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC → RGB float32 CHW, ImageNet-normalized."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img_rgb, (2, 0, 1))

def preprocess_for_yolo(img_bgr: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """The exact transform used at SERVE time for YOLOv11. Returns uint8 BGR
    HWC (Ultralytics handles its own normalization downstream)."""
    if cfg.apply_clahe:
        img_bgr = apply_clahe_lab(img_bgr, cfg.clahe_clip, cfg.clahe_grid)
    img_bgr, _, _ = letterbox(img_bgr, cfg.target_size, cfg.pad_color)
    return img_bgr

def preprocess_for_classifier(img_bgr: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """Used by Phase 5 fault classifier and Phase 4 anomaly heads.
    Returns float32 CHW, ImageNet-normalized."""
    if cfg.apply_clahe:
        img_bgr = apply_clahe_lab(img_bgr, cfg.clahe_clip, cfg.clahe_grid)
    img_bgr = cv2.resize(img_bgr, (cfg.target_size, cfg.target_size), interpolation=cv2.INTER_AREA)
    return to_chw_normalized(img_bgr)
```

### 6.3 `thermal.py`

```python
"""Radiometric IO. Supports FLIR RJPEG (via exiftool) and 16-bit TIFF + CSV."""
from __future__ import annotations
import json, subprocess, tempfile
import numpy as np, cv2
from pathlib import Path

def load_radiometric(path: Path) -> tuple[np.ndarray, str]:
    """Return (temp_celsius_array, palette_name). Falls back to grayscale
    intensity-as-pseudo-temp if no radiometric data is found (logged)."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Cannot read TIFF {path}")
        # Convention: 16-bit raw → centi-degrees + 100°C offset
        return arr.astype(np.float32) / 100.0 - 100.0, "raw_tiff"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        # FLIR RJPEG: extract embedded raw + meta via exiftool
        try:
            meta = json.loads(subprocess.check_output(
                ["exiftool", "-j", "-Emissivity", "-PlanckR1", "-PlanckR2",
                 "-PlanckB", "-PlanckF", "-PlanckO", "-RawThermalImageType", str(path)]))[0]
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                subprocess.check_call(["exiftool", "-b", "-RawThermalImage", str(path)],
                                       stdout=open(tmp.name, "wb"))
                raw = cv2.imread(tmp.name, cv2.IMREAD_UNCHANGED).astype(np.float32)
            R1, R2 = meta["PlanckR1"], meta["PlanckR2"]
            B, F, O = meta["PlanckB"], meta["PlanckF"], meta["PlanckO"]
            radiance = raw / R2 + O
            temp_k = B / np.log(R1 / (R2 * (radiance - O)) + F)
            return temp_k - 273.15, "flir_rjpeg"
        except Exception:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            return gray / 255.0 * 80.0 - 10.0, "pseudo_from_grayscale"
    raise ValueError(f"Unsupported thermal format: {path.suffix}")

def encode_thermal_png(temp_c: np.ndarray) -> np.ndarray:
    """uint16 PNG encoding: pixel = (temp_c + 100) * 100, clipped to [0, 65535]."""
    enc = np.clip((temp_c + 100.0) * 100.0, 0, 65535).astype(np.uint16)
    return enc

def decode_thermal_png(arr_u16: np.ndarray) -> np.ndarray:
    return arr_u16.astype(np.float32) / 100.0 - 100.0
```

### 6.4 `fusion.py`

```python
"""Combine RGB (uint8 BGR) and thermal (uint16) into 4-channel uint8 image."""
import numpy as np, cv2
from .thermal import decode_thermal_png

def fuse_rgb_thermal(rgb_bgr: np.ndarray, thermal_u16: np.ndarray) -> np.ndarray:
    """Returns HxWx4 uint8 (B, G, R, T). T is rescaled to [0,255] over a fixed
    [-10°C, 100°C] range so the channel scale is identical across captures."""
    if thermal_u16.shape[:2] != rgb_bgr.shape[:2]:
        thermal_u16 = cv2.resize(thermal_u16, (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
    temp_c = decode_thermal_png(thermal_u16)
    t_u8 = np.clip((temp_c + 10.0) / 110.0 * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([rgb_bgr, t_u8])
```

### 6.5 `quality.py` (Image Quality Assessment gate)

```python
"""IQA gate. Returns (ok, details). Called at INGEST and again at SERVE
(serve-time failures are flagged but do not block — Phase 3 skips them)."""
import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class IQAThresholds:
    blur_laplacian_min: float = 80.0     # variance of Laplacian; higher = sharper
    exposure_min:       float = 35.0     # mean luminance, 0..255
    exposure_max:       float = 220.0
    saturation_clip_max: float = 0.20    # max fraction of pixels with V > 250
    underexposure_max:   float = 0.30    # max fraction with V < 10
    min_side_px:         int   = 384

def iqa_check(img_bgr, thresh: IQAThresholds = IQAThresholds()) -> tuple[bool, dict]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(gray.mean())
    sat_clip = float((gray > 250).mean())
    under    = float((gray < 10).mean())

    details = {
        "blur_var": blur, "mean_luminance": mean_lum,
        "saturation_clip_frac": sat_clip, "underexposure_frac": under,
        "min_side_px": min(h, w),
    }
    ok = (
        blur >= thresh.blur_laplacian_min
        and thresh.exposure_min <= mean_lum <= thresh.exposure_max
        and sat_clip <= thresh.saturation_clip_max
        and under    <= thresh.underexposure_max
        and min(h, w) >= thresh.min_side_px
    )
    details["ok"] = ok
    return ok, details
```

Rejected images are written to `logs/preproc/<run_id>/rejected.jsonl` and never enter `dataset_v2/`. They are queued back to the engineer for recapture.

### 6.6 `aug.py` (Albumentations — preserves demo recipe and extends it)

```python
"""Augmentation pipelines. RGB-only stack mirrors demo augment.py; the
multi-modal stack treats the thermal channel as an additional image to
keep alignment invariants."""
import albumentations as A
import cv2

# Demo-equivalent pipeline (classes 0–5 augmentation; bbox-safe).
RGB_TRAIN = A.Compose([
    A.LongestMaxSize(max_size=1024, p=1.0),
    A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.7),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.MotionBlur(blur_limit=5, p=0.15),
    A.GaussianBlur(blur_limit=(3, 5), p=0.10),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.20),
    A.ISONoise(p=0.10),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.30),
    A.Affine(rotate=(-10, 10), translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
             cval=(114, 114, 114), p=0.50),
    # Industrial-specific: simulate cabinet shadowing and reflective glare
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4, p=0.25),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.3), src_radius=80, p=0.05),
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.30),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                            min_visibility=0.3, min_area=8))

# Thermal-aware: treat thermal as additional_targets so geometric ops are mirrored.
FUSED_TRAIN = A.Compose([
    A.LongestMaxSize(max_size=1024, p=1.0),
    A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-7, 7), translate_percent=(-0.04, 0.04), scale=(0.95, 1.05), p=0.5),
    # Photometric ops applied ONLY to RGB; thermal kept as-is via additional_targets handling.
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=15, val_shift_limit=10, p=1.0),
    ], p=0.6),
], additional_targets={"thermal": "image"},
   bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))

VAL = A.Compose([
    A.LongestMaxSize(max_size=1024, p=1.0),
    A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
```

Per-modality and per-fault augmentation intensity is configured in `configs/preproc/augmentation_profiles.yaml`:

```yaml
profiles:
  default_rgb:
    pipeline: RGB_TRAIN
    aug_per_image: 1
  scarce_class_rgb:           # for fault IDs with < 100 training images
    pipeline: RGB_TRAIN
    aug_per_image: 8
  thermal_paired:
    pipeline: FUSED_TRAIN
    aug_per_image: 2
  classifier_crops:
    pipeline: A.Compose([...])   # tight-crop appropriate; defined inline
    aug_per_image: 4

fault_overrides:
  9:  scarce_class_rgb         # rectifier_capacitor_bulge — rare
  100: scarce_class_rgb        # fan_blade_broken — rare
  119: scarce_class_rgb        # ip_rating_breach — rare
```

### 6.7 `pipeline.py` (composes everything)

```python
"""High-level entry points used by training and serving."""
from .transforms import (preprocess_for_yolo, preprocess_for_classifier,
                          PreprocConfig)
from .quality import iqa_check, IQAThresholds
from .aug import RGB_TRAIN, FUSED_TRAIN, VAL
from .fusion import fuse_rgb_thermal

def build_train_transform(modality: str = "rgb"):
    if modality == "rgb":
        return RGB_TRAIN
    if modality in {"fused", "thermal"}:
        return FUSED_TRAIN
    raise ValueError(modality)

def build_eval_transform():
    return VAL

def serve_preprocess_yolo(img_bgr, cfg: PreprocConfig | None = None):
    cfg = cfg or PreprocConfig()
    ok, iqa = iqa_check(img_bgr)
    return preprocess_for_yolo(img_bgr, cfg), {"iqa_ok": ok, "iqa": iqa}

def serve_preprocess_classifier(crop_bgr, cfg: PreprocConfig | None = None):
    cfg = cfg or PreprocConfig(target_size=384)
    return preprocess_for_classifier(crop_bgr, cfg)
```

---

## 7. Dataset preparation for downstream phases

### 7.1 Populating `dataset_v2/images/<modality>/<split>/`

After labeling completes, run:

```bash
# Move labeled frames from raw_ingest → splits-resolved canonical tree
python scripts/preproc/finalize_dataset.py \
  --raw-ingest dataset_v2/raw_ingest/ \
  --manifest   dataset_v2/manifest.parquet \
  --splits     dataset_v2/splits/
```

This script:

1. Joins each ingested `image_id` to its `split` from `splits/<split>.txt`.
2. Copies `rgb/<image_id>.jpg` → `images/rgb/<split>/<image_id>.jpg`.
3. Copies `thermal/<image_id>.png` → `images/thermal/<split>/<image_id>.png` (if exists).
4. Copies `fused/<image_id>.png` → `images/fused/<split>/<image_id>.png` (if exists).
5. Verifies that for every image, the matching label files (Phase 1 §6.1) exist.
6. Re-runs `verify_dataset_v2.py` and fails on any missing pair.

### 7.2 Pre-computing crops for Phase 5

```bash
python scripts/preproc/build_classifier_crops.py \
  --manifest dataset_v2/manifest.parquet \
  --labels   dataset_v2/labels/faults \
  --images   dataset_v2/images/rgb \
  --out      dataset_v2/crops \
  --pad-ratio 0.15 \
  --min-side-px 96
```

Iterates over every fault bbox, expands by `pad-ratio`, crops the RGB image, writes to `dataset_v2/crops/<split>/<fault_name>/<image_id>__<bbox_idx>.jpg`. Crops smaller than `min-side-px` are skipped (logged).

### 7.3 Pre-computing normals-only set for Phase 4

```bash
python scripts/preproc/build_normals_only.py \
  --manifest dataset_v2/manifest.parquet \
  --labels   dataset_v2/labels/components \
  --images   dataset_v2/images/rgb \
  --out      dataset_v2/normals_only
```

For each component class, extracts crops from images where `is_normal == True` (image has no fault bboxes) and the component is present. Output layout matches MVTec AD convention so PatchCore reference implementations work out-of-the-box.

```text
dataset_v2/normals_only/
└── electrolytic_capacitor/
    ├── train/good/   # only "good" — PatchCore convention
    │   ├── img_0001.jpg
    │   └── ...
    └── val/
        ├── good/     # for fitting score thresholds
        └── defect/   # crops from fault-labeled images used at eval time
```

---

## 8. Model training pipeline (Phase 2 has no model)

Phase 2 trains nothing. However it owns **two calibrated artifacts** that look like models and must be versioned identically:

| Artifact | Treated like a model? | Versioned how |
|---|---|---|
| Per-device homography + intrinsics | Yes | Tagged `calibration-<device_id>-vYYYYMMDD`; recorded in `manifest.parquet.calibration_version` |
| IQA threshold profile | Yes | `configs/preproc/iqa_thresholds.yaml` versioned via git tag `iqa-vN` |

Drift in either silently degrades every downstream model, so they are gated through the same MLflow registry promotion flow that real models use (Phase 8).

---

## 9. Integration contracts

| Consumer | Contract |
|---|---|
| **Phase 3 (YOLO component)** | Reads images from `dataset_v2/images/rgb/<split>/`; labels from `dataset_v2/labels/components/<split>/`. Imports `powervision.preproc.pipeline.serve_preprocess_yolo` at inference time. |
| **Phase 4 (PatchCore / EfficientAD)** | Reads `dataset_v2/normals_only/<component>/train/good/`; uses `powervision.preproc.serve_preprocess_classifier` for ROI crops at inference time. |
| **Phase 5 (Fault classifier)** | Reads `dataset_v2/crops/<split>/<fault_name>/`; uses `serve_preprocess_classifier`. |
| **Phase 6 (Fault decision unit)** | Calls `serve_preprocess_yolo` and `serve_preprocess_classifier` for new inputs. Reads thermal metadata from `dataset_v2/thermal_meta/<image_id>.json` for rule evaluation (e.g., hot-spot rule). |
| **Phase 7 (LLM report)** | Embeds `thermal_meta` summary (min/max/avg °C) into prompt context. |
| **Phase 8 (HITL)** | Re-ingestion of corrected captures uses the same `ingest_session.py` so reviewed images are bit-identical in preprocessing. |

### 9.1 Preprocessing microservice contract (optional)

For inference deployments that prefer not to bundle OpenCV into every service, expose preprocessing over HTTP.

```python
# services/preproc/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import io, numpy as np, cv2
from powervision.preproc.pipeline import serve_preprocess_yolo, serve_preprocess_classifier

app = FastAPI(title="PowerVision Preproc")

@app.post("/preprocess/yolo")
async def preprocess_yolo(file: UploadFile = File(...)):
    buf = np.frombuffer(await file.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    out, meta = serve_preprocess_yolo(img)
    ok, encoded = cv2.imencode(".png", out)
    return StreamingResponse(io.BytesIO(encoded.tobytes()),
                              media_type="image/png",
                              headers={"X-IQA-OK": str(meta["iqa_ok"]).lower()})

@app.post("/preprocess/classifier")
async def preprocess_classifier(file: UploadFile = File(...)):
    buf = np.frombuffer(await file.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    chw = serve_preprocess_classifier(img)
    return {"shape": list(chw.shape), "dtype": str(chw.dtype),
            "tensor_b64": chw.tobytes().hex()}
```

Contract: requests are multipart `image/*`; response is either a PNG (YOLO) or a JSON-serialized tensor (classifier). Both expose `X-IQA-OK` header so the caller's confidence gate (Phase 6) can short-circuit bad input.

---

## 10. Code structure for Phase 2

```text
scripts/
├── preproc/
│   ├── calibrate_rgb_ir.py
│   ├── finalize_dataset.py
│   ├── build_classifier_crops.py
│   ├── build_normals_only.py
│   ├── preview_alignment.py        # debug: side-by-side RGB+IR+overlay grid
│   └── preview_augmentation.py     # replaces demo's preview_augmentation.py
└── ingest/
    └── ingest_session.py

powervision/preproc/
├── __init__.py
├── thermal.py
├── align.py
├── fusion.py
├── transforms.py
├── quality.py
├── aug.py
└── pipeline.py

services/preproc/
└── main.py

configs/preproc/
├── iqa_thresholds.yaml
└── augmentation_profiles.yaml

calibration/                 # per-device, DVC-tracked
└── <device_id>/

dataset_v2/raw_ingest/       # ephemeral, gitignored
dataset_v2/thermal_meta/     # DVC-tracked
```

### 10.1 Extended `dvc.yaml` stages for Phase 2

```yaml
stages:
  ingest_session:
    cmd: python scripts/ingest/ingest_session.py
         --session ${session_path}
         --calibration-root calibration/
         --out dataset_v2/raw_ingest/
    deps:
      - powervision/preproc/
      - calibration/${device_id}/homography.json
      - scripts/ingest/ingest_session.py

  finalize_dataset:
    cmd: python scripts/preproc/finalize_dataset.py
         --raw-ingest dataset_v2/raw_ingest/
         --manifest   dataset_v2/manifest.parquet
         --splits     dataset_v2/splits/
    deps:
      - dataset_v2/raw_ingest
      - dataset_v2/manifest.parquet
      - dataset_v2/splits
    outs:
      - dataset_v2/images
      - dataset_v2/thermal_meta

  build_classifier_crops:
    cmd: python scripts/preproc/build_classifier_crops.py
         --manifest dataset_v2/manifest.parquet
         --labels   dataset_v2/labels/faults
         --images   dataset_v2/images/rgb
         --out      dataset_v2/crops
    deps:
      - dataset_v2/images/rgb
      - dataset_v2/labels/faults
      - scripts/preproc/build_classifier_crops.py
    outs:
      - dataset_v2/crops

  build_normals_only:
    cmd: python scripts/preproc/build_normals_only.py
         --manifest dataset_v2/manifest.parquet
         --labels   dataset_v2/labels/components
         --images   dataset_v2/images/rgb
         --out      dataset_v2/normals_only
    deps:
      - dataset_v2/images/rgb
      - dataset_v2/labels/components
      - dataset_v2/manifest.parquet
      - scripts/preproc/build_normals_only.py
    outs:
      - dataset_v2/normals_only
```

---

## 11. Evaluation of Phase 2 itself

Phase 2 has no model metrics; it has **pipeline-quality metrics** written to `dataset_v2/qa/preproc_report.json`.

| Metric | Target |
|---|---|
| Calibration reprojection error (median, RGB px) | ≤ 1.5 |
| Frames rejected by IQA gate | ≤ 8% of incoming |
| Thermal–RGB alignment IoU on calibration checkerboard | ≥ 0.85 |
| Preprocessing latency, p95 on Intel i7 / no GPU | ≤ 80 ms per RGB-only image at 640² |
| Preprocessing latency, p95 with thermal fusion | ≤ 220 ms per image |
| Train/serve preprocessing parity (hash equality on a 500-image fixture) | 100% |

The parity test is essential. It hashes 500 images put through the **training** transform vs the **serve** transform and asserts they are byte-identical:

```python
# tests/preproc/test_train_serve_parity.py
import hashlib, cv2
from powervision.preproc.pipeline import build_eval_transform, serve_preprocess_yolo

def _hash(arr): return hashlib.sha256(arr.tobytes()).hexdigest()

def test_train_serve_parity(fixture_500):
    eval_t = build_eval_transform()
    for img_path in fixture_500:
        img = cv2.imread(str(img_path))
        train_out = eval_t(image=img, bboxes=[], class_labels=[])["image"]
        serve_out, _ = serve_preprocess_yolo(img)
        assert _hash(train_out) == _hash(serve_out), img_path
```

---

## 12. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Thermal–RGB drift after camera rig knocked | Anomaly heatmaps offset; PatchCore false positives | Per-session ECC refinement (`align.refine_alignment`); recalibration alert when reprojection error > 2.5 px |
| FLIR firmware change breaks RJPEG parsing | Thermal ingestion crashes | `load_radiometric` falls back to pseudo-grayscale + flag in `manifest.quality_flags` |
| Engineer turns off auto-exposure → blown highlights | IQA gate rejects; engineer must reshoot | Capture wizard enforces exposure check on-device; ingest CLI hard-rejects |
| Lighting variance across sites (cabinet open vs closed) | Model under-performs at one site | CLAHE in preprocessing + `RandomShadow`/`RandomSunFlare` augmentations during training |
| Two engineers use different smartphones | Color cast differs | Canonical sRGB profile applied in `canonicalize_rgb`; per-device color profile (ICC) supported in `calibration/<device_id>/icc.profile` |
| Train-serve preprocessing skew | Production model worse than val mAP | Parity test (§11) in CI; any preprocessing change requires re-running `verify_dataset_v2.py` |
| Heavy thermal noise on cheap sensors | Anomaly scoring inflated | NLMeans denoising step inside `load_radiometric` for sensors with NETD > 50 mK; flagged in `thermal_meta.json` |
| Air-gapped industrial site → no `exiftool` | Radiometric extraction fails | Fallback to vendor SDK (FLIR Atlas, Hikmicro) inside `load_radiometric`; CI tests both code paths |
| GDPR / PII in raw images (engineer's face reflected in cabinet glass) | Compliance | Optional face-blur stage `powervision/preproc/privacy.py` toggled by `configs/preproc/privacy.yaml` |
| Misframed multi-angle pair (subject only in 1 of 3 angles) | Inference returns "no component" for that angle, voting in Phase 6 silently loses a quorum | IQA gate flags `is_misframed`; Phase 6 vote weights down missing-angle votes rather than treating them as "no detection" |

---

## 13. Phase 2 exit checklist

- [ ] `powervision/preproc/` package implements all 7 modules, unit-tested.
- [ ] `calibrate_rgb_ir.py` produces calibration with median reprojection error ≤ 1.5 px on the heated checkerboard fixture.
- [ ] `ingest_session.py` accepts the canonical session JSON and writes to `dataset_v2/raw_ingest/`.
- [ ] `finalize_dataset.py` populates `dataset_v2/images/<modality>/<split>/` and passes `verify_dataset_v2.py`.
- [ ] `build_classifier_crops.py` and `build_normals_only.py` populate Phase 4 / Phase 5 input directories.
- [ ] Train/serve preprocessing parity test green on 500-image fixture.
- [ ] Preprocessing microservice (`services/preproc/main.py`) passes contract tests against `serve_preprocess_yolo` / `serve_preprocess_classifier`.
- [ ] `dvc.yaml` Phase 2 stages run clean.
- [ ] Demo project (`api/streamlit_app.py`, `api/main.py`) still serves inference unchanged because their preprocessing path remains backward-compatible (no new required imports for the legacy code path).

Phase 2 is **frozen** when all boxes are checked. Phase 3 begins.
