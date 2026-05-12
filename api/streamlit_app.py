"""Minimal Streamlit demo: upload image file as-is, run YOLO from disk path (same as API)."""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import streamlit as st
import yaml
from ultralytics import YOLO

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DEFAULT_IDEAL_CONFIG = API_DIR / "ideal_images.yaml"
CONF, IOU = 0.25, 0.5
_ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        hint = (
            "Windows paths: do **not** wrap `C:\\...` in **double** quotes in YAML "
            "(backslash starts escapes, e.g. `\\U` in `\\Users`). "
            "Use **single** quotes `'C:\\Users\\...'` or forward slashes `C:/Users/...`."
        )
        raise ValueError(f"{path}:\n{e}\n\n{hint}") from e
    return data if isinstance(data, dict) else {}


def _class_names() -> dict[int, str]:
    names = _load_yaml(DATASET_YAML).get("names")
    if not isinstance(names, dict):
        return {}
    out: dict[int, str] = {}
    for k, v in names.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def _ideal_cfg_path() -> Path:
    env = os.environ.get("FAULT_IDEAL_YAML", "").strip()
    return Path(env) if env else DEFAULT_IDEAL_CONFIG


def _weights_path(cfg: dict[str, Any]) -> Path:
    w = cfg.get("weights")
    if isinstance(w, str) and w.strip():
        p = Path(w.strip()).expanduser()
        if p.is_file():
            return p
    return DEFAULT_WEIGHTS


def _ideal_path(class_id: int, ideal_map: dict[str, str]) -> Path | None:
    if class_id in (0, 1, 2, 4, 5):
        key = "group_01245"
    elif class_id in (3, 6, 7, 8):
        key = f"class_{class_id}"
    else:
        return None
    raw = ideal_map.get(key, "")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw.strip()).expanduser()
    return p if p.is_file() else None


@st.cache_resource
def _model(path_str: str) -> YOLO:
    return YOLO(path_str)


def _load_model_hint(exc: BaseException) -> str:
    msg = str(exc)
    if "C3k2" in msg or "can't get attribute" in msg.lower():
        return (
            f"{msg}\n\n"
            "This usually means **best.pt** was trained with **Ultralytics 8.3+** (YOLO11) but your "
            "environment has an older **8.2.x** build. Upgrade and restart Streamlit:\n\n"
            "`pip install -U \"ultralytics>=8.3.100\"`"
        )
    return msg


def _suffix_from_upload(name: str | None) -> str:
    suf = Path(name or "").suffix.lower()
    return suf if suf in _ALLOWED_SUFFIX else ".jpg"


def main() -> None:
    st.set_page_config(page_title="Fault demo", layout="centered")
    st.header("Fault detection demo")

    try:
        cfg = _load_yaml(_ideal_cfg_path())
    except ValueError as e:
        st.error(str(e))
        st.stop()
    ideal_map = cfg.get("ideal_images")
    ideal_map = ideal_map if isinstance(ideal_map, dict) else {}
    wpath = _weights_path(cfg)
    names = _class_names()

    if not wpath.is_file():
        st.error(f"Missing weights: `{wpath}`")
        st.stop()

    try:
        model = _model(str(wpath.resolve()))
    except Exception as e:
        st.error(_load_model_hint(e))
        st.stop()

    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is None:
        return

    raw = up.getvalue()
    suffix = _suffix_from_upload(up.name)

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw)
        res = model.predict(
            source=tmp_path,
            imgsz=640,
            conf=CONF,
            iou=IOU,
            max_det=20,
            verbose=False,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    r0 = res[0] if res else None
    if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
        st.warning("No fault detected.")
        st.image(BytesIO(raw), use_container_width=True)
        return

    best_i = int(r0.boxes.conf.argmax())
    cid = int(r0.boxes.cls[best_i])
    conf = float(r0.boxes.conf[best_i])
    label = names.get(cid, str(cid))

    st.subheader(label.replace("_", " "))
    st.caption(f"Confidence {conf:.0%}")

    plot_bgr = r0.plot()
    plot_rgb = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)
    ideal = _ideal_path(cid, ideal_map)

    c1, c2 = st.columns(2)
    with c1:
        st.image(plot_rgb, caption="Tagged", use_container_width=True)
    with c2:
        if ideal is not None:
            st.image(str(ideal), caption="Ideal", use_container_width=True)
        else:
            st.caption("Ideal: set path in `api/ideal_images.yaml`")


if __name__ == "__main__":
    main()
