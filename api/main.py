"""FastAPI server for YOLO fault detection inference."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "runs" / "phase_b" / "weights" / "best.pt"
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fault Detection API", version="1.0.0")

model: YOLO | None = None

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_upload_filename(original: str | None) -> str:
    """Use the client filename (basename only, sanitized) so prediction outputs match.

    Ultralytics saves annotated images as ``save_dir / <same basename as source>``,
    so the uploaded file must keep the user's filename stem (never use a random UUID).
    """
    raw = (original or "").strip()
    raw = Path(raw).name if raw else "upload.jpg"
    if not raw or raw == ".":
        raw = "upload.jpg"
    suffix = Path(raw).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        suffix = ".jpg"
    stem = Path(raw).stem
    stem = re.sub(r"[^\w\-.]", "_", stem).strip("._")[:200]
    if not stem:
        stem = "upload"
    return f"{stem}{suffix}"


@app.on_event("startup")
def load_model() -> None:
    """Load YOLO weights once at startup."""
    global model
    if not WEIGHTS_PATH.is_file():
        raise RuntimeError(
            f"Weights not found: {WEIGHTS_PATH}. Train Phase B first or set the correct path."
        )
    model = YOLO(str(WEIGHTS_PATH))


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if model is None:
        return JSONResponse(
            {"message": "Model not loaded", "detections": []},
            status_code=503,
        )

    filename = _safe_upload_filename(file.filename)
    input_path = UPLOAD_FOLDER / filename

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=str(input_path),
        imgsz=640,
        conf=0.25,
        iou=0.5,
        max_det=20,
        save=True,
        project=str(OUTPUT_FOLDER),
        name="predictions",
        exist_ok=True,
    )

    detections: list[dict] = []
    names = model.names

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_idx = int(box.cls[0])
            class_name = names[cls_idx] if isinstance(names, dict) else names[cls_idx]
            detections.append(
                {
                    "class_id": cls_idx,
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
            )

    return JSONResponse(
        {
            "message": "Prediction successful",
            "detections": detections,
        }
    )


def _primary_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None
    return max(detections, key=lambda d: float(d.get("confidence", 0.0)))


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Run inference on multiple uploads; returns one entry per file (no image bytes)."""
    if model is None:
        return JSONResponse(
            {"message": "Model not loaded", "results": []},
            status_code=503,
        )
    if not files:
        return JSONResponse({"message": "No files", "results": []})

    names = model.names
    out: list[dict[str, Any]] = []

    for i, upload in enumerate(files):
        filename = _safe_upload_filename(upload.filename)
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        unique_name = f"{stem}_b{i}{suffix}"
        input_path = UPLOAD_FOLDER / unique_name
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        results = model.predict(
            source=str(input_path),
            imgsz=640,
            conf=0.25,
            iou=0.5,
            max_det=20,
            save=False,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_idx = int(box.cls[0])
                class_name = names[cls_idx] if isinstance(names, dict) else names[cls_idx]
                detections.append(
                    {
                        "class_id": cls_idx,
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

        primary = _primary_detection(detections)
        out.append(
            {
                "filename": unique_name,
                "original_filename": upload.filename or unique_name,
                "has_fault": bool(detections),
                "detections": detections,
                "primary": primary,
            }
        )

    return JSONResponse({"message": "Prediction successful", "results": out})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "weights": str(WEIGHTS_PATH)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
