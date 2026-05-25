"""PowerVision AI — Streamlit UI: multi-upload, analysis, aggregate report, per-image detail."""

from __future__ import annotations

import html
import os
import shutil
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from ultralytics import YOLO

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DEFAULT_IDEAL_CONFIG = API_DIR / "ideal_images.yaml"
CONF, IOU = 0.25, 0.5
_MAX_FILES = 10
_MAX_BYTES_PER_FILE = 10 * 1024 * 1024
_ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_UPLOAD_TILE = 140
# Detail page: analyzed vs ideal shown in identical-sized frames (cover crop).
_DETAIL_COMPARE_W = 720
_DETAIL_COMPARE_H = 480

_UI_CSS = """
<style>
  /* Mockup-aligned palette: light gray canvas, white cards, blue primary, red fault, green clear */
  div[data-testid="stAppViewContainer"] > .main {
    background: #f0f2f5 !important;
  }
  div[data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
  .main .block-container {
    padding-top: 0.65rem !important;
    padding-bottom: 1.25rem !important;
    max-width: 1080px !important;
  }
  h1, h2, h3, h4, h5, h6 {
    margin-top: 0 !important;
    margin-bottom: 0.25rem !important;
    color: #1a1d21 !important;
    font-weight: 600 !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e5e8ed !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    margin-bottom: 0.5rem !important;
  }
  button[kind="primary"] {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
  }
  button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
  }
  .pv-header {
    background: #ffffff;
    border: 1px solid #e5e8ed;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .pv-brand {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a1d21;
    letter-spacing: -0.02em;
  }
  .pv-brand-dot { color: #2563eb; font-size: 1.25rem; margin-right: 0.35rem; vertical-align: middle; }
  .pv-sub { font-size: 0.8rem; color: #6b7280; margin-top: 0.15rem; }
  .pv-hint { font-size: 0.82rem; color: #6b7280; line-height: 1.45; margin: 0.15rem 0 0.4rem 0; }
  .pv-section-title { font-size: 0.95rem; font-weight: 600; color: #1a1d21; margin: 0.35rem 0 0.25rem 0; }
  .fault-pill {
    display: inline-block;
    border: 1px solid #f87171;
    color: #dc2626;
    background: #fef2f2;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .ok-pill {
    display: inline-block;
    border: 1px solid #6ee7b7;
    color: #047857;
    background: #ecfdf5;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .dot-green {
    width: 6px; height: 6px; border-radius: 50%; background: #10b981;
    display: inline-block; vertical-align: middle; margin-right: 4px;
  }
  .pv-ready {
    font-size: 0.72rem; color: #047857; border: 1px solid #d1fae5; background: #f0fdf4;
    padding: 2px 8px; border-radius: 6px; display: inline-block; margin-top: 2px;
  }
  .log-box {
    background: #1e293b;
    color: #e2e8f0;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 0.78rem;
    padding: 0.75rem;
    border-radius: 10px;
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid #334155;
    line-height: 1.4;
  }
  .log-line { margin: 0.1rem 0; white-space: pre-wrap; }
  .modal-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    border: 1px solid #e5e8ed;
    font-size: 0.9rem;
    margin-top: 0.4rem;
    color: #374151;
  }
  .detail-alert {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
  }
  .detail-meta {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 1rem 1.1rem 1.35rem 1.1rem;
    margin-top: 0.5rem;
    margin-bottom: 1.25rem;
    font-size: 0.88rem;
    color: #44403c;
    line-height: 1.65;
    box-sizing: border-box;
  }
  .detail-meta .k { color: #78716c; font-size: 0.8rem; }
  .detail-meta-outer {
    display: block;
    margin-top: 0.35rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
  }
  .pv-file-line {
    font-size: 0.82rem;
    color: #374151;
    padding: 0.2rem 0;
    border-bottom: 1px solid #f3f4f6;
  }
  .pv-file-line:last-child { border-bottom: none; }
  .pv-stats {
    font-size: 0.88rem;
    color: #4b5563;
    margin-bottom: 0.35rem;
  }
  .stat-pill {
    display: inline-block;
    margin-right: 0.4rem;
    margin-bottom: 0.2rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
  }
  .stat-blue { background: #e8f1ff; color: #1d4ed8; }
  .stat-green { background: #ecfdf3; color: #047857; }
  .stat-amber { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
</style>
"""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        hint = (
            "Windows paths: do **not** wrap `C:\\...` in **double** quotes in YAML "
            "(backslash starts escapes). Use **single** quotes or forward slashes."
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


def _fault_label_and_crisp(class_name: str) -> tuple[str, str]:
    """Short fault title and one-line crisp description (table + detail)."""
    rows: dict[str, tuple[str, str]] = {
        "input_cable_fault": (
            "Input cable fault",
            "Electromill input cables look incorrectly routed or landed.",
        ),
        "output_cable_fault": (
            "Output cable fault",
            "Electromill output cables look incorrectly routed or terminated.",
        ),
        "loose_connection": (
            "Loose connection",
            "Signal (or control) cable appears loose or not fully seated.",
        ),
        "screw_faults": (
            "Screw fault",
            "Screw size or style does not match the ideal hardware for this spot.",
        ),
        "signal_cable_mismatch": (
            "Signal cable mismatch",
            "Wrong signal cable appears connected at this junction.",
        ),
        "ri_cable_mismatch": (
            "RI cable mismatch",
            "RI cable pairing or routing does not match the expected layout.",
        ),
        "J14_cable_mismatch": (
            "J14 cable mismatch",
            "J14 harness is wired incorrectly for this connector.",
        ),
        "red_white_mismatch": (
            "Red / white mismatch",
            "Red/white pair is reversed or crossed versus the reference scheme.",
        ),
        "ferrule_mismatch": (
            "Ferrule mismatch",
            "Ferrule or end sleeve does not match the reference termination.",
        ),
    }
    if class_name in rows:
        return rows[class_name]
    short = class_name.replace("_", " ").strip().title() or "Anomaly"
    return (short, "Review the highlighted region against approved drawings.")


@st.cache_resource
def _model(path_str: str) -> YOLO:
    return YOLO(path_str)


def _load_model_hint(exc: BaseException) -> str:
    msg = str(exc)
    if "C3k2" in msg or "can't get attribute" in msg.lower():
        return (
            f"{msg}\n\n"
            "This usually means **best.pt** was trained with **Ultralytics 8.3+** but your "
            "environment is older. Upgrade: `pip install -U \"ultralytics>=8.3.100\"`"
        )
    return msg


def _dims_from_bytes(raw: bytes) -> tuple[int, int]:
    try:
        im = Image.open(BytesIO(raw))
        w, h = im.size
        return int(w), int(h)
    except Exception:
        return 0, 0


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _clear_workdir() -> None:
    d = st.session_state.get("work_tmpdir")
    if isinstance(d, str) and d and os.path.isdir(d):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    st.session_state.work_tmpdir = None


def _init_state() -> None:
    defaults = {
        "phase": "upload",
        "pending_files": [],
        "results": [],
        "analysis_logs": [],
        "analysis_pct": 0,
        "analysis_cursor": 0,
        "selected_idx": None,
        "abort_requested": False,
        "show_complete_modal": False,
        "work_tmpdir": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_header() -> None:
    st.markdown(_UI_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="pv-header">'
        '<span class="pv-brand-dot">&#9679;</span><span class="pv-brand">PowerVision AI Inspection System</span>'
        '<div class="pv-sub">Industrial image inspection</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _log_line(msg: str, status: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    safe = html.escape(msg, quote=True)
    return f"[{ts}] {safe}  <span style='float:right;color:#94a3b8'>{html.escape(status)}</span>"


def _ensure_work_tmp() -> Path:
    t = st.session_state.get("work_tmpdir")
    if isinstance(t, str) and t and os.path.isdir(t):
        return Path(t)
    tmp = tempfile.mkdtemp(prefix="powervision_")
    st.session_state.work_tmpdir = tmp
    return Path(tmp)


def _process_one_image(
    model: YOLO, names: dict[int, str], ideal_map: dict[str, str], step: int, item: dict[str, Any], tmp: Path
) -> None:
    name = item["name"]
    raw: bytes = item["bytes"]
    n = max(len(st.session_state.pending_files), 1)
    st.session_state.analysis_pct = int((step / n) * 100)
    st.session_state.analysis_logs.append(
        _log_line(
            f'YOLO.predict on "{name}" ({step + 1}/{n}) · imgsz=640 · conf≥{CONF} · IoU={IOU}…',
            "RUNNING",
        )
    )
    path = tmp / Path(name).name
    safe = path.name
    if Path(safe).suffix.lower() not in _ALLOWED_SUFFIX:
        safe = Path(safe).stem + ".jpg"
    path = tmp / safe
    path.write_bytes(raw)
    res = model.predict(
        source=str(path),
        imgsz=640,
        conf=CONF,
        iou=IOU,
        max_det=20,
        verbose=False,
    )
    r0 = res[0] if res else None
    has_fault = bool(r0 and r0.boxes is not None and len(r0.boxes) > 0)
    if not has_fault:
        st.session_state.results.append(
            {
                "filename": name,
                "has_fault": False,
                "fault_label": "",
                "description": "No fault flagged by the model for this image.",
                "confidence": 0.0,
                "class_name": "",
                "class_id": -1,
                "annotated_path": str(path),
                "raw_path": str(path),
                "ideal_path": None,
                "bytes_len": len(raw),
                "dims": _dims_from_bytes(raw),
            }
        )
        st.session_state.analysis_logs[-1] = _log_line(f'YOLO.predict finished for "{name}".', "OK")
        st.session_state.analysis_logs.append(
            _log_line(f"No boxes above conf={CONF}; image marked clear.", "DONE")
        )
    else:
        best_i = int(r0.boxes.conf.argmax())
        cid = int(r0.boxes.cls[best_i])
        conf = float(r0.boxes.conf[best_i])
        cname = names.get(cid, str(cid))
        short, crisp = _fault_label_and_crisp(cname)
        plot_bgr = r0.plot()
        ann_path = tmp / f"{path.stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), plot_bgr)
        ideal = _ideal_path(cid, ideal_map)
        st.session_state.results.append(
            {
                "filename": name,
                "has_fault": True,
                "fault_label": short,
                "description": crisp,
                "confidence": conf,
                "class_name": cname,
                "class_id": cid,
                "annotated_path": str(ann_path),
                "raw_path": str(path),
                "ideal_path": str(ideal) if ideal else None,
                "bytes_len": len(raw),
                "dims": _dims_from_bytes(raw),
            }
        )
        st.session_state.analysis_logs[-1] = _log_line(f'YOLO.predict finished for "{name}".', "OK")
        st.session_state.analysis_logs.append(
            _log_line(
                f'Top box: class `{cname}` at {conf:.1%} conf (best of {len(r0.boxes)}); overlay saved.',
                "DONE",
            )
        )


def _analysis_dialog(model: YOLO, names: dict[int, str], ideal_map: dict[str, str]) -> None:
    """Modal overlay for in-progress analysis (requires Streamlit >= 1.36)."""
    if not hasattr(st, "dialog"):
        st.error("PowerVision AI needs Streamlit 1.36 or newer. Run: `pip install -U \"streamlit>=1.36\"`")
        st.session_state.phase = "upload"
        return

    @st.dialog("PowerVision AI — Analysis in progress", width="large")
    def _inner() -> None:
        st.markdown("**Analysis in progress**")
        st.caption("Evaluating images for cabling, hardware, and connection conformity.")
        pct = int(st.session_state.analysis_pct)
        st.progress(min(max(pct, 0), 100) / 100.0, text=f"{pct}%")
        log_html = (
            '<div class="log-box">'
            + "".join(f'<div class="log-line">{line}</div>' for line in st.session_state.analysis_logs)
            + "</div>"
        )
        st.markdown(log_html, unsafe_allow_html=True)
        if st.button("Abort analysis", type="secondary", use_container_width=True):
            st.session_state.abort_requested = True

        files: list[dict[str, Any]] = st.session_state.pending_files
        cursor = int(st.session_state.analysis_cursor)
        tmp_path = st.session_state.work_tmpdir

        if st.session_state.abort_requested:
            if cursor < len(files):
                st.session_state.analysis_cursor = len(files)
            if not st.session_state.show_complete_modal:
                st.session_state.analysis_logs.append(_log_line("Aborted by user.", "WARN"))
                st.session_state.analysis_pct = 100
                st.session_state.show_complete_modal = True
                st.rerun()
        elif cursor < len(files) and isinstance(tmp_path, str) and tmp_path:
            _process_one_image(model, names, ideal_map, cursor, files[cursor], Path(tmp_path))
            st.session_state.analysis_cursor = cursor + 1
            time.sleep(0.06)
            st.rerun()
        elif len(files) == 0:
            st.warning("No files to analyze.")
            st.session_state.phase = "upload"
            st.rerun()
        elif not st.session_state.show_complete_modal:
            st.session_state.analysis_logs.append(
                _log_line(f"Batch finished: {len(files)} image(s) inferred; report ready.", "OK")
            )
            st.session_state.analysis_pct = 100
            st.session_state.show_complete_modal = True
            st.rerun()

        if st.session_state.show_complete_modal:
            st.markdown(
                '<div class="modal-card" style="margin-top:0.75rem">Analysis finished successfully. '
                "Use Next to open the combined report.</div>",
                unsafe_allow_html=True,
            )
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.phase = "report"
                st.session_state.show_complete_modal = False
                st.rerun()

    _inner()


def _rectangle_cover(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize and center-crop so output is exactly ``out_w``×``out_h`` (cover)."""
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return rgb
    scale = max(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - out_w) // 2)
    y0 = max(0, (nh - out_h) // 2)
    return resized[y0 : y0 + out_h, x0 : x0 + out_w]


def _thumb_square_cover(rgb: np.ndarray, side: int) -> np.ndarray:
    """Resize and center-crop to `side`×`side` so every thumbnail matches."""
    return _rectangle_cover(rgb, side, side)


def _load_rgb(path: str) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _detail_compare_box(path: str) -> np.ndarray | None:
    """Fixed canvas for detail view so analyzed and ideal panels share one box size."""
    rgb = _load_rgb(path)
    if rgb is None:
        return None
    return _rectangle_cover(rgb, _DETAIL_COMPARE_W, _DETAIL_COMPARE_H)


def _detail_meta_html(filename: str, ts: str, fault_type: str) -> str:
    fn = html.escape(filename)
    tl = html.escape(ts)
    ft = html.escape(fault_type)
    return (
        '<div class="detail-meta-outer">'
        '<div class="detail-meta"><strong>Analysis metadata</strong><br/>'
        f'<span class="k">Image name</span> · {fn}<br/>'
        f'<span class="k">Time stamp</span> · {tl}<br/>'
        f'<span class="k">Fault type</span> · {ft}'
        "</div></div>"
    )


_GRID_COLS = 5


def main() -> None:
    st.set_page_config(page_title="PowerVision AI", layout="wide", initial_sidebar_state="collapsed")
    _init_state()

    try:
        cfg = _load_yaml(_ideal_cfg_path())
    except ValueError as e:
        st.error(str(e))
        st.stop()
    ideal_map = cfg.get("ideal_images")
    ideal_map = ideal_map if isinstance(ideal_map, dict) else {}
    wpath = _weights_path(cfg)
    names = _class_names()

    _render_header()

    with st.sidebar:
        if st.button("Reset workflow", type="secondary"):
            _clear_workdir()
            st.session_state.phase = "upload"
            st.session_state.pending_files = []
            st.session_state.results = []
            st.session_state.selected_idx = None
            st.session_state.show_complete_modal = False
            st.session_state.analysis_cursor = 0
            st.session_state.analysis_logs = []
            st.session_state.analysis_pct = 0
            st.session_state.abort_requested = False
            st.rerun()

    if not wpath.is_file():
        st.error(f"Missing weights: `{wpath}`")
        st.stop()
    try:
        model = _model(str(wpath.resolve()))
    except Exception as e:
        st.error(_load_model_hint(e))
        st.stop()

    # Detail view
    if st.session_state.phase == "detail" and st.session_state.selected_idx is not None:
        idx = st.session_state.selected_idx
        if 0 <= idx < len(st.session_state.results):
            r = st.session_state.results[idx]
            if st.button("← Back", key="detail_back"):
                st.session_state.phase = "report"
                st.session_state.selected_idx = None
                st.rerun()

            with st.container(border=True):
                st.markdown(
                    '<p class="pv-section-title" style="margin:0 0 0.35rem 0;">Inspection detail</p>',
                    unsafe_allow_html=True,
                )
                ann = r.get("annotated_path") or ""
                ip = r.get("ideal_path") or ""
                ts = datetime.now().strftime("%d-%m-%Y %I:%M %p")

                # Images: analyzed overlay and ideal reference side-by-side
                col_a, col_b = st.columns(2, gap="medium")
                with col_a:
                    st.caption("**Analyzed · model overlay**")
                    if ann and os.path.isfile(ann):
                        boxed_ann = _detail_compare_box(ann)
                        if boxed_ann is not None:
                            st.image(boxed_ann, use_container_width=True)
                        else:
                            st.caption("—")
                    else:
                        st.caption("—")
                with col_b:
                    st.caption("**Ideal · reference**")
                    if ip and os.path.isfile(ip):
                        boxed_ideal = _detail_compare_box(ip)
                        if boxed_ideal is not None:
                            st.image(boxed_ideal, use_container_width=True)
                        else:
                            st.caption("—")
                    elif r["has_fault"]:
                        st.caption("Configure `api/ideal_images.yaml` for this class.")
                    else:
                        st.caption("—")

                st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

                # Description + metadata below the images
                if r["has_fault"]:
                    st.markdown(
                        '<div class="detail-alert"><strong style="color:#b91c1c">Fault detected</strong> · '
                        f'<span style="font-size:1.05rem;font-weight:700">{r["confidence"]:.1%}</span>'
                        " <span style=\"color:#64748b\">detection confidence</span></div>",
                        unsafe_allow_html=True,
                    )
                    ft = html.escape(
                        (r.get("fault_label") or "").strip()
                        or (str(r.get("class_name") or "").replace("_", " ").title())
                        or "Fault",
                    )
                    st.markdown(
                        f'<p style="margin:0.35rem 0 0.2rem 0;font-size:0.9rem;">'
                        f'<span style="color:#64748b;font-weight:600;">Fault type</span> · '
                        f'<span style="color:#1d4ed8;font-weight:600;">{ft}</span></p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Description**")
                    st.markdown(
                        f'<p style="margin:0.1rem 0 0.65rem 0;font-size:0.88rem;color:#4b5563;line-height:1.45;">{html.escape(r["description"])}</p>',
                        unsafe_allow_html=True,
                    )
                    fl_meta = (r.get("fault_label") or "").strip() or "—"
                    st.markdown(
                        _detail_meta_html(r["filename"], ts, fl_meta),
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("No fault detected for this image.")
                    st.markdown(
                        '<p style="margin:0.15rem 0 0.2rem 0;font-size:0.9rem;">'
                        '<span style="color:#64748b;font-weight:600;">Fault type</span> · '
                        '<span style="color:#047857;font-weight:600;">Clear</span></p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Description**")
                    st.markdown(
                        f'<p style="margin:0 0 0.65rem 0;font-size:0.88rem;color:#4b5563;">{html.escape(r["description"])}</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _detail_meta_html(r["filename"], ts, "Clear"),
                        unsafe_allow_html=True,
                    )
            st.stop()
        st.session_state.phase = "report"
        st.session_state.selected_idx = None

    if st.session_state.phase == "upload":
        with st.container(border=True):
            st.markdown('<p class="pv-section-title" style="margin:0;">Upload component images</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="pv-hint">JPEG, PNG, or TIFF · max 10 MB each · up to 10 files.</p>',
                unsafe_allow_html=True,
            )
            up = st.file_uploader(
                "Choose files",
                type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if up:
                if len(up) > _MAX_FILES:
                    st.warning(f"Only the first {_MAX_FILES} files are used.")
                rows: list[dict[str, Any]] = []
                for f in up[:_MAX_FILES]:
                    b = f.getvalue()
                    if len(b) > _MAX_BYTES_PER_FILE:
                        st.error(f"`{f.name}` exceeds 10 MB — remove or compress it.")
                        st.stop()
                    rows.append({"name": f.name, "bytes": b})
                st.session_state.pending_files = rows

            if st.session_state.pending_files:
                items = st.session_state.pending_files
                n = len(items)
                st.markdown('<p class="pv-section-title">Uploaded payload</p>', unsafe_allow_html=True)
                st.progress(min(n / _MAX_FILES, 1.0), text=f"{n} / {_MAX_FILES} images")
                nrows = (n + _GRID_COLS - 1) // _GRID_COLS
                for row_i in range(nrows):
                    cols = st.columns(_GRID_COLS, gap="small")
                    for c in range(_GRID_COLS):
                        idx = row_i * _GRID_COLS + c
                        with cols[c]:
                            if idx < n:
                                item = items[idx]
                                rgb = np.array(Image.open(BytesIO(item["bytes"])).convert("RGB"))
                                tile = _thumb_square_cover(rgb, _UPLOAD_TILE)
                                st.image(tile, width=_UPLOAD_TILE)
                                st.markdown(
                                    '<span class="pv-ready"><span class="dot-green"></span>Ready</span>',
                                    unsafe_allow_html=True,
                                )
                lines: list[str] = []
                for item in items:
                    w, h = _dims_from_bytes(item["bytes"])
                    res = f"{w}×{h}" if w else "—"
                    lines.append(
                        '<div class="pv-file-line">'
                        f"<strong>{html.escape(item['name'])}</strong>"
                        f" · {_fmt_size(len(item['bytes']))} · {html.escape(res)}"
                        "</div>"
                    )
                st.markdown("".join(lines), unsafe_allow_html=True)

            run = st.columns([1, 1], gap="small")
            with run[1]:
                start = st.button(
                    "Start analysis",
                    type="primary",
                    use_container_width=True,
                    disabled=not st.session_state.pending_files,
                )
            if start:
                _clear_workdir()
                st.session_state.work_tmpdir = tempfile.mkdtemp(prefix="powervision_")
                st.session_state.phase = "analyzing"
                st.session_state.analysis_cursor = 0
                st.session_state.results = []
                nq = len(st.session_state.pending_files)
                wn = Path(wpath).name
                st.session_state.analysis_logs = [
                    _log_line(f"Analysis batch started: {nq} file(s) queued.", "OK"),
                    _log_line(
                        f"Detector: Ultralytics YOLO using `{wn}` — loaded once at startup and reused. "
                        f"Per image: model.predict(imgsz=640, conf={CONF}, iou={IOU}, max_det=20).",
                        "OK",
                    ),
                ]
                st.session_state.show_complete_modal = False
                st.session_state.abort_requested = False
                st.session_state.analysis_pct = 0
                st.rerun()

    elif st.session_state.phase == "analyzing":
        _analysis_dialog(model, names, ideal_map)

    elif st.session_state.phase == "report":
        if not st.session_state.results:
            with st.container(border=True):
                st.info("No results yet. Upload images to begin.")
                if st.button("Go to upload"):
                    st.session_state.phase = "upload"
                    st.rerun()
            st.stop()

        results = st.session_state.results
        n_tot = len(results)
        n_fault = sum(1 for x in results if x["has_fault"])
        with st.container(border=True):
            st.markdown('<p class="pv-section-title" style="margin:0;">Detailed findings</p>', unsafe_allow_html=True)
            st.markdown(
                '<div class="pv-stats">'
                f'<span class="stat-pill stat-blue">{n_tot} images</span>'
                f'<span class="stat-pill stat-amber">{n_fault} faults</span>'
                f'<span class="stat-pill stat-green">{n_tot - n_fault} clear</span>'
                "</div>",
                unsafe_allow_html=True,
            )

        for i, r in enumerate(results):
            status_html = (
                '<span class="fault-pill">Fault detected</span>'
                if r["has_fault"]
                else '<span class="ok-pill">Clear</span>'
            )
            with st.container(border=True):
                top = st.columns([0.42, 0.58, 2.55, 0.52], gap="small")
                with top[0]:
                    st.markdown(f"**{i + 1}**")
                with top[1]:
                    p = r["annotated_path"] if r["has_fault"] and os.path.isfile(r["annotated_path"]) else r["raw_path"]
                    rgb_t = _load_rgb(p)
                    if rgb_t is not None:
                        st.image(_thumb_square_cover(rgb_t, 56), width=56)
                with top[2]:
                    st.markdown(
                        f'<div style="font-size:0.84rem;font-weight:600;margin-bottom:0.2rem;">{html.escape(r["filename"])}</div>',
                        unsafe_allow_html=True,
                    )
                    if r["has_fault"]:
                        ft_rep = (r.get("fault_label") or "").strip() or (
                            str(r.get("class_name") or "").replace("_", " ").title()
                        )
                        ft_rep = html.escape(ft_rep or "Fault")
                        st.markdown(
                            f'<div style="font-size:0.8rem;font-weight:600;color:#1d4ed8;margin-bottom:0.15rem;">'
                            f"Fault type · {ft_rep}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="font-size:0.8rem;font-weight:600;color:#047857;margin-bottom:0.15rem;">'
                            "Fault type · Clear</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#334155;margin-bottom:0.25rem;">{html.escape(r["description"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(status_html, unsafe_allow_html=True)
                with top[3]:
                    if st.button("Open", key=f"open_{i}", use_container_width=True):
                        st.session_state.selected_idx = i
                        st.session_state.phase = "detail"
                        st.rerun()

        if st.button("New upload", type="secondary"):
            _clear_workdir()
            st.session_state.phase = "upload"
            st.session_state.pending_files = []
            st.session_state.results = []
            st.session_state.analysis_cursor = 0
            st.session_state.analysis_logs = []
            st.session_state.abort_requested = False
            st.session_state.show_complete_modal = False
            st.rerun()


if __name__ == "__main__":
    main()
