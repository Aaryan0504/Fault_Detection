# Phase 8 — MLOps: Retraining, Versioning, and Human-in-the-Loop Pipeline

> **Scope:** Backend + infrastructure. Closes the production loop: harvests human-reviewed escalations from Phase 6's HITL queue, ingests them into the dataset under Phase 1 contracts, triggers automated retraining of any model that crosses freshness or quality thresholds (Phases 3–5), runs a full evaluation gate, registers and promotes models through a versioned registry, and rolls out new artifacts with canary deployment, monitoring, and one-command rollback. Also covers per-tenant data isolation, drift detection, and the auditable lineage that ties every served prediction back to its exact dataset version, code commit, and model artifact.

---

## 1. Phase objective

The first seven phases ship a system. Phase 8 keeps it accurate as the world changes — new UPS models, new vendor parts, new fault modes, new sites with different lighting, drift in component populations.

Deliverables:

| Deliverable | Artifact |
|---|---|
| Active learning loop from HITL | `scripts/mlops/harvest_hitl.py` + `services/active_learning/main.py` |
| Label feedback ingestion | `scripts/mlops/ingest_resolutions.py` |
| Automated retraining triggers | `services/orchestrator/` (Airflow/Prefect/Dagster — example uses **Prefect 3**) |
| Model registry | MLflow (self-hosted) backed by Postgres + S3 |
| Promotion gates | `scripts/mlops/promote_model.py` with quality + safety checks |
| Canary deployment | `services/router/main.py` (traffic-splitting in front of model servers) |
| Drift detection | `services/drift/main.py` (Evidently AI + custom industrial metrics) |
| Lineage manifest | `lineage/manifest.parquet` (every served prediction → exact artifacts) |
| Per-tenant data isolation | `policies/tenant_isolation.yaml` + DVC remotes per tenant |
| Cost + capacity dashboards | Grafana dashboards under `dashboards/mlops/` |
| Rollback CLI | `scripts/mlops/rollback.py` |
| Per-UPS-type adapters (LoRA) | `scripts/mlops/train_lora_adapter.py` |
| Incident runbook | `docs/runbooks/mlops_incidents.md` |

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Format |
|---|---|---|
| HITL resolutions | Phase 6 `hitl_tickets` Postgres table | JSON |
| Live inference logs | Decision service (Phase 6) | JSONL (`logs/decisions/<batch_id>.jsonl`) |
| Drift signals | Live preprocessing + classifier confidence distribution | Prometheus + custom store |
| Model registry state | MLflow | API |
| Current dataset version | `dataset_v2/manifest.parquet` + DVC tags | Parquet + git |
| Code commits | Git | git |

### 2.2 Outputs

```text
lineage/
├── manifest.parquet                # served_prediction → artifact_hashes
├── runs/                           # MLflow tracking dir mirror (S3 in prod)
└── dataset_versions/<tax_ver>/...

mlops/
├── retraining_queue.jsonl          # pending retraining jobs
├── promotion_log.jsonl             # every promotion / rollback
├── drift_alerts.jsonl              # drift events
└── canary_metrics/<deploy_id>.json # canary performance vs baseline

datasets_pending/
└── <ingestion_id>/                 # staged HITL-derived examples awaiting verification
    ├── images/
    ├── labels/
    └── provenance.json
```

---

## 3. Active learning loop

### 3.1 Two-source candidate harvesting

A new "candidate for labeling" comes from either:

1. **HITL escalations** (Phase 6 → `hitl_tickets`): low-confidence predictions that a human has now adjudicated.
2. **Unlabeled production traffic** that meets active-learning criteria (high uncertainty, far from training distribution, near-decision-boundary, novel embeddings).

```python
# scripts/mlops/harvest_hitl.py
"""Pull resolved HITL tickets from Postgres → stage in datasets_pending/<id>/."""
from __future__ import annotations
import argparse, json, uuid, shutil
from datetime import datetime, timedelta
from pathlib import Path
import psycopg, yaml

OUT = Path("datasets_pending")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-days", type=int, default=7)
    ap.add_argument("--max-rows", type=int, default=2000)
    args = ap.parse_args()

    dsn = "postgresql://" + yaml.safe_load(open("configs/mlops/db.yaml"))["dsn"]
    since = datetime.utcnow() - timedelta(days=args.since_days)
    ingestion_id = f"hitl_{datetime.utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    out_dir = OUT / ingestion_id
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels/faults").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels/components").mkdir(parents=True, exist_ok=True)

    n = 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT ticket_id, image_verdict, resolution
               FROM hitl_tickets
               WHERE resolved = true AND resolved_at >= %s
               ORDER BY resolved_at LIMIT %s""",
            (since, args.max_rows))
        for tid, iv, res in cur:
            verdict = iv if isinstance(iv, dict) else json.loads(iv)
            resolution = res if isinstance(res, dict) else json.loads(res)
            _stage(out_dir, verdict, resolution)
            n += 1

    (out_dir / "provenance.json").write_text(json.dumps({
        "ingestion_id": ingestion_id,
        "source": "hitl",
        "rows": n,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))
    print(f"Staged {n} HITL examples into {out_dir}")

def _stage(out_dir: Path, verdict: dict, resolution: dict):
    image_id = verdict["image_id"]
    src_img  = Path(verdict["annotated_image_path"]).with_suffix(".jpg") \
                if verdict.get("annotated_image_path") else None
    # Copy original (un-annotated) image if available; fall back to annotated.
    raw = Path(verdict.get("raw_image_path", src_img))
    if not raw or not raw.exists(): return
    dest = out_dir / "images" / f"{image_id}.jpg"
    shutil.copy(raw, dest)

    # Convert corrections into YOLO label files
    fault_lines, comp_lines = [], []
    H, W = _read_size(dest)
    for c in resolution.get("bbox_corrections", []):
        x1, y1, x2, y2 = c["bbox_xyxy"]
        cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        if "fault_id" in c:
            fault_lines.append(f"{c['fault_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if "component_id" in c:
            comp_lines.append(f"{c['component_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if resolution["is_fault"] and not fault_lines:
        # Reviewer marked fault but did not redraw; reuse the verdict's evidence bboxes
        for ev in verdict.get("evidence", []):
            if ev.get("fused_label") and ev["fused_label"] != "no_fault":
                bb = ev["component"]["bbox"]
                cx, cy = ((bb["x1"] + bb["x2"]) / 2) / W, ((bb["y1"] + bb["y2"]) / 2) / H
                bw, bh = (bb["x2"] - bb["x1"]) / W, (bb["y2"] - bb["y1"]) / H
                fault_lines.append(f"{resolution['fault_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    (out_dir / "labels/faults"     / f"{image_id}.txt").write_text("\n".join(fault_lines))
    (out_dir / "labels/components" / f"{image_id}.txt").write_text("\n".join(comp_lines))

def _read_size(p):
    from PIL import Image
    with Image.open(p) as im: return im.height, im.width

if __name__ == "__main__":
    main()
```

### 3.2 Uncertainty-based mining (live traffic, unlabeled)

```python
# services/active_learning/main.py (sketch)
"""For each successful inference, decide whether to enqueue the image for
labeling based on uncertainty and novelty. Embeddings from DINOv2 are used
to skip near-duplicates of already-labeled data."""

UNCERTAINTY_RULES = [
    # Predicate over the per-image audit log row → returns True to enqueue
    lambda r: 0.40 <= r["max_classifier_top1_prob"] <= 0.65,
    lambda r: r["anomaly_fused_score"] is not None and abs(r["anomaly_fused_score"] - r["anomaly_threshold"]) < 0.05,
    lambda r: r["novelty_score"] > 0.85,    # set by §3.3
]
```

### 3.3 Novelty score via DINOv2 + nearest neighbors

```python
# powervision/active/novelty.py
"""Cosine distance from query DINOv2 embedding to the nearest 'known' embedding
in the training set. High = likely novel; queue for labeling."""
import torch, numpy as np
from powervision.anomaly.dinov2_backbone import DINOv2Patches

class NoveltyDetector:
    def __init__(self, train_embeds: torch.Tensor, k: int = 5):
        self.embeds = torch.nn.functional.normalize(train_embeds, dim=-1)
        self.k = k
        self.backbone = DINOv2Patches(model="dinov2_vitl14", layers=(3,)).eval()

    @torch.inference_mode()
    def novelty_score(self, img_chw_tensor) -> float:
        feats = self.backbone(img_chw_tensor.unsqueeze(0)).mean(dim=1).squeeze(0)  # global pool
        feats = torch.nn.functional.normalize(feats, dim=-1)
        sims = (self.embeds @ feats.cpu())
        return 1.0 - float(sims.topk(self.k).values.mean())     # ∈ [0, 2]
```

A nightly batch precomputes embeddings for every training image (1024-d × N). Queries during inference compare against an HNSW index (`faiss`). Top-N% novelty per day go to the labeling queue.

### 3.4 Ingestion pipeline (`scripts/mlops/ingest_resolutions.py`)

After harvesting / mining, staged data must enter `dataset_v2/` via the **same** Phase 1 contracts:

```python
"""Verify staged ingestion → promote to dataset_v2/, bump manifest, run dataset
verifier. Always runs verify_taxonomy + verify_dataset_v2 before merging."""
from __future__ import annotations
import argparse, json, subprocess, shutil
from pathlib import Path
import pandas as pd

STAGE = Path("datasets_pending")
DEST  = Path("dataset_v2")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingestion-id", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    src = STAGE / args.ingestion_id
    prov = json.loads((src / "provenance.json").read_text())

    # 1. Validate every label file against taxonomy ids
    subprocess.check_call(["python", "scripts/taxonomy/verify_taxonomy.py"])

    # 2. Copy into dataset_v2/
    for img in (src / "images").glob("*"):
        dest = DEST / "images" / "rgb" / args.split / img.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest)
        for kind in ("faults", "components"):
            sl = src / "labels" / kind / f"{img.stem}.txt"
            if sl.exists():
                dl = DEST / "labels" / kind / args.split / sl.name
                dl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(sl, dl)

    # 3. Re-build manifest + splits + verify
    subprocess.check_call(["python", "scripts/taxonomy/build_manifest.py"])
    subprocess.check_call(["python", "scripts/taxonomy/build_splits.py"])
    subprocess.check_call(["python", "scripts/taxonomy/verify_dataset_v2.py"])

    # 4. Update provenance — every newly added image_id is tagged with the ingestion
    df = pd.read_parquet(DEST / "manifest.parquet")
    new_ids = [img.stem for img in (src / "images").glob("*")]
    mask = df["image_id"].isin(new_ids)
    df.loc[mask, "provenance_source"] = "hitl"
    df.loc[mask, "labeling_session_id"] = args.ingestion_id
    df.to_parquet(DEST / "manifest.parquet", index=False)

    # 5. DVC add + commit
    subprocess.check_call(["dvc", "add", str(DEST / "images"), str(DEST / "labels")])
    subprocess.check_call(["git", "add", "."])
    subprocess.check_call(["git", "commit", "-m",
                            f"hitl: ingest {prov['rows']} examples ({args.ingestion_id})"])
    subprocess.check_call(["dvc", "push"])

    print(f"Ingestion {args.ingestion_id} promoted into dataset_v2/{args.split}.")

if __name__ == "__main__":
    main()
```

---

## 4. Automated retraining triggers

### 4.1 Trigger types

| Trigger | Threshold | Affects |
|---|---|---|
| **Data freshness** | ≥ 500 new HITL-corrected examples since last retrain | Phase 3, 4, 5 |
| **Per-class data freshness** | ≥ 50 new examples for any single fault class | Phase 5 |
| **Drift** | Phase 6's `pv_review_rate` > 25% sustained for 7 days **OR** classifier confidence distribution KS-test p < 0.01 vs baseline | Phase 5 (classifier first), then Phase 3 |
| **Anomaly score drift** | PatchCore FPR > 25% on rolling-window normals | Phase 4 (re-fit memory bank only) |
| **Taxonomy bump** | New fault IDs added | Phase 5 first, then Phase 6 routing + Phase 7 KB pull |
| **Calendar** | Quarterly mandatory retrain regardless | All trainable |

### 4.2 Orchestration with Prefect

```python
# services/orchestrator/flows.py
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import subprocess
from prefect import flow, task, get_run_logger
from prefect.schedules import IntervalSchedule
from prefect.deployments import Deployment

@task(retries=2, retry_delay_seconds=120)
def harvest():
    subprocess.check_call(["python", "scripts/mlops/harvest_hitl.py", "--since-days", "7"])

@task
def discover_ingestions() -> list[str]:
    return [p.name for p in Path("datasets_pending").iterdir()
            if (p / "provenance.json").exists() and not (p / ".promoted").exists()]

@task(retries=1)
def ingest(ingestion_id: str):
    subprocess.check_call(["python", "scripts/mlops/ingest_resolutions.py",
                            "--ingestion-id", ingestion_id])
    (Path("datasets_pending") / ingestion_id / ".promoted").touch()

@task
def evaluate_triggers() -> list[str]:
    """Return list of pipelines to fire: subset of ['component','classifier','anomaly']."""
    out = subprocess.check_output(["python", "scripts/mlops/evaluate_triggers.py", "--json"])
    return __import__("json").loads(out)["pipelines"]

@task(retries=0, timeout_seconds=60 * 60 * 24)
def retrain_component():
    subprocess.check_call(["dvc", "repro", "component_phase_a"])
    subprocess.check_call(["dvc", "repro", "component_phase_b"])
    subprocess.check_call(["dvc", "repro", "component_validate"])
    subprocess.check_call(["dvc", "repro", "component_tune_thresholds"])

@task(retries=0, timeout_seconds=60 * 60 * 12)
def retrain_classifier():
    subprocess.check_call(["dvc", "repro", "classifier_phase_a"])
    subprocess.check_call(["dvc", "repro", "classifier_phase_b"])
    subprocess.check_call(["dvc", "repro", "classifier_calibrate"])
    subprocess.check_call(["dvc", "repro", "classifier_evaluate"])

@task(retries=0, timeout_seconds=60 * 60 * 6)
def retrain_anomaly():
    subprocess.check_call(["python", "scripts/anomaly/train_all_components.py", "--mode", "delta"])

@task
def gate_and_promote(pipelines: list[str]):
    for p in pipelines:
        subprocess.check_call(["python", "scripts/mlops/promote_model.py",
                                "--pipeline", p, "--target-stage", "Staging"])

@flow(name="mlops-loop")
def mlops_loop():
    log = get_run_logger()
    harvest()
    for iid in discover_ingestions():
        ingest(iid)
    pipelines = evaluate_triggers()
    log.info(f"Triggers: {pipelines}")
    if "component" in pipelines:  retrain_component()
    if "classifier" in pipelines: retrain_classifier()
    if "anomaly" in pipelines:    retrain_anomaly()
    if pipelines:                 gate_and_promote(pipelines)

if __name__ == "__main__":
    Deployment.build_from_flow(
        flow=mlops_loop, name="nightly",
        schedule=IntervalSchedule(interval=timedelta(hours=24)),
    ).apply()
```

### 4.3 Trigger evaluation logic

```python
# scripts/mlops/evaluate_triggers.py
"""Read drift store + manifest delta → emit which pipelines to retrain."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timedelta
import pandas as pd, psycopg, yaml
import mlflow

def evaluate() -> dict:
    cfg = yaml.safe_load(open("configs/mlops/triggers.yaml"))
    out = {"pipelines": []}

    # 1. Data freshness
    delta = _new_examples_since_last_retrain()
    if delta["total"] >= cfg["data_freshness_threshold"]:
        out["pipelines"] += ["component", "classifier"]
    for fid, n in delta["per_class"].items():
        if n >= cfg["per_class_threshold"]:
            out["pipelines"] = list(set(out["pipelines"] + ["classifier"]))
            break

    # 2. Drift
    review_rate = _prom_query_avg_24h('avg_over_time(pv_review_rate[7d])')
    if review_rate > cfg["review_rate_max"]:
        out["pipelines"] = list(set(out["pipelines"] + ["classifier", "component"]))

    fpr = _prom_query_avg_24h('avg_over_time(pv_patchcore_fpr[7d])')
    if fpr > cfg["patchcore_fpr_max"]:
        out["pipelines"] = list(set(out["pipelines"] + ["anomaly"]))

    # 3. Taxonomy bump
    if _taxonomy_bumped_since_last_train():
        out["pipelines"] = list(set(out["pipelines"] + ["classifier"]))

    # 4. Calendar
    last_full = _last_full_retrain()
    if datetime.utcnow() - last_full > timedelta(days=cfg["quarterly_days"]):
        out["pipelines"] = ["component", "classifier", "anomaly"]

    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", action="store_true"); args = ap.parse_args()
    res = evaluate()
    print(json.dumps(res) if args.json else res)
```

### 4.4 `configs/mlops/triggers.yaml`

```yaml
version: "1.0.0"
data_freshness_threshold: 500
per_class_threshold: 50
review_rate_max: 0.25
patchcore_fpr_max: 0.25
classifier_conf_ks_pvalue_min: 0.01
quarterly_days: 90
```

---

## 5. Model registry and promotion gates

### 5.1 MLflow as the source of truth

| Registered model | Stages used |
|---|---|
| `powervision-component-detector` | `None → Staging → Production` |
| `powervision-fault-classifier` | `None → Staging → Production` |
| `powervision-patchcore-<component>` | `None → Staging → Production` |
| `powervision-efficientad-<component>` | `None → Staging → Production` |
| `powervision-rule-set` | `None → Staging → Production` (non-model artifact; rule_definitions.yaml hash + tests) |

Every training run logs:

- The exact code commit (`mlflow.set_tag("git_commit", subprocess.check_output(["git","rev-parse","HEAD"]).decode())`).
- The DVC dataset hash (`mlflow.set_tag("dvc_hash", subprocess.check_output(["dvc","status"]).decode()[:8])` or, better, `mlflow.log_artifact("dataset_v2/manifest.parquet")`).
- Taxonomy version.
- Preproc version.
- Hyperparameters.
- Metrics from the eval pack (`evaluation_report.json`).

### 5.2 Promotion gate (`scripts/mlops/promote_model.py`)

```python
"""Promote a model from None → Staging or Staging → Production after the
quality gate AND safety gate pass."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

QUALITY_GATES = {
    "powervision-component-detector": {
        "overall_mAP50":      {"min": 0.85},
        "overall_mAP50_95":   {"min": 0.65},
        "worst_class_mAP50":  {"min": 0.70},
    },
    "powervision-fault-classifier": {
        "top_1_acc":          {"min": 0.88},
        "macro_f1":           {"min": 0.80},
        "worst_class_recall": {"min": 0.70},
        "ece":                {"max": 0.05},
    },
}

REGRESSION_TOLERANCE = {"top_1_acc": 0.005, "macro_f1": 0.005,
                         "overall_mAP50": 0.005, "ece": 0.005}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=["component", "classifier", "anomaly"])
    ap.add_argument("--target-stage", default="Staging", choices=["Staging", "Production"])
    ap.add_argument("--registered-name", default=None)
    args = ap.parse_args()
    client = MlflowClient()

    name = args.registered_name or {"component": "powervision-component-detector",
                                     "classifier": "powervision-fault-classifier"}[args.pipeline]
    latest = client.get_latest_versions(name, stages=["None"])[-1]
    metrics = client.get_run(latest.run_id).data.metrics
    gates = QUALITY_GATES.get(name, {})

    # 1. Absolute quality gate
    for metric, bound in gates.items():
        v = metrics.get(metric)
        if v is None: _fail(f"missing metric {metric}")
        if "min" in bound and v < bound["min"]: _fail(f"{metric}={v:.4f} < {bound['min']}")
        if "max" in bound and v > bound["max"]: _fail(f"{metric}={v:.4f} > {bound['max']}")

    # 2. Regression gate (vs current Production)
    prods = client.get_latest_versions(name, stages=["Production"])
    if prods:
        prod_metrics = client.get_run(prods[-1].run_id).data.metrics
        for metric, tol in REGRESSION_TOLERANCE.items():
            if metric in metrics and metric in prod_metrics:
                d = (prod_metrics[metric] - metrics[metric])
                if metric != "ece":   # smaller is better only for ece
                    if d > tol: _fail(f"regression: {metric} {prod_metrics[metric]:.4f} → {metrics[metric]:.4f}")
                else:
                    if -d > tol: _fail(f"calibration regression: ece {prod_metrics[metric]:.4f} → {metrics[metric]:.4f}")

    # 3. Safety gate: artifact compatibility
    _assert_taxonomy_compatible(client, latest)

    client.transition_model_version_stage(
        name=name, version=latest.version, stage=args.target_stage,
        archive_existing_versions=(args.target_stage == "Production"),
    )
    print(f"Promoted {name} v{latest.version} → {args.target_stage}")

def _fail(msg):
    print(f"PROMOTION GATE FAIL: {msg}"); raise SystemExit(2)
```

### 5.3 Promotion log

Every promotion appends to `mlops/promotion_log.jsonl`:

```json
{"ts": "2026-05-25T07:14:12Z", "name": "powervision-fault-classifier", "version": 12, "from": "Staging", "to": "Production", "actor": "auto", "metrics": {"top_1_acc": 0.891, "macro_f1": 0.812}, "git_commit": "a1b2c3d", "taxonomy_version": "2.0.3"}
```

---

## 6. Per-UPS-type LoRA adapters

When a new UPS type onboards with limited data, full retraining of EfficientNet-B4 (Phase 5) is overkill. Instead, train a **LoRA adapter** on top of the base classifier.

```python
# scripts/mlops/train_lora_adapter.py
"""LoRA fine-tune the fault classifier head on a single ups_type_id slice."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import timm
import yaml, json
from peft import LoraConfig, get_peft_model

from powervision.classify.data import CropDataset, build_class_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/classifier/eff_b4_phase_b/weights/best.pt")
    ap.add_argument("--ups-type-id", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.base, map_location="cpu")
    cfg = ckpt["cfg"]
    ci = json.loads((Path(args.base).parent.parent / "class_index.json").read_text())
    model = timm.create_model(cfg["model"]["name"], num_classes=len(ci), pretrained=False).cuda()
    model.load_state_dict(ckpt["model"])

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                       target_modules=["conv_dw", "fc1", "fc2", "classifier"], bias="none")
    model = get_peft_model(model, lora)

    # Train only the LoRA adapters; filter dataset to the ups_type
    cfg["data"]["filter_ups_type_id"] = args.ups_type_id
    train = CropDataset(cfg["data"], split="train", class_index=ci)
    val   = CropDataset(cfg["data"], split="val",   class_index=ci)
    # ... standard training loop, 5–10 epochs, AdamW lr=5e-4 ...

    out = Path(args.out or f"runs/classifier/lora/{args.ups_type_id}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

if __name__ == "__main__":
    main()
```

Routing at inference: the decision service (Phase 6) chooses the adapter by `ups_type_id` (loaded into memory at startup; switching is a few-MB swap).

---

## 7. Canary deployment

### 7.1 Traffic-splitting router

```python
# services/router/main.py
"""Front-door HTTP proxy that splits traffic between Production and Canary
versions of each model service. Routes by a hash on session_id to keep a
session's images consistent."""
from __future__ import annotations
from fastapi import FastAPI, Request, Response
import httpx, os, hashlib, yaml

app = FastAPI()
ROUTING = yaml.safe_load(open("configs/router/routing.yaml"))
CLIENT = httpx.AsyncClient(timeout=120.0)

@app.api_route("/decide_batch", methods=["POST"])
async def decide_batch(request: Request):
    sid = request.headers.get("X-Session-Id", "anon")
    h = int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 100
    canary_pct = ROUTING["canary_pct"]
    target = ROUTING["canary_url"] if h < canary_pct else ROUTING["production_url"]
    upstream = await CLIENT.post(target + "/decide_batch", content=await request.body(),
                                  headers=request.headers)
    return Response(upstream.content, status_code=upstream.status_code,
                     headers=dict(upstream.headers), media_type=upstream.headers.get("content-type"))
```

### 7.2 Routing config and gradual rollout

```yaml
# configs/router/routing.yaml
production_url: http://decision-prod:8000
canary_url:     http://decision-canary:8000
canary_pct: 5
rollout_schedule:
  - {after_minutes: 60,  canary_pct: 10}
  - {after_minutes: 240, canary_pct: 25}
  - {after_minutes: 720, canary_pct: 50}
  - {after_minutes: 1440, canary_pct: 100}
auto_rollback:
  metric: pv_decision_latency_ms_p95
  baseline_window_min: 30
  canary_window_min:   10
  regression_pct: 30
shadow_eval:
  enabled: true
  duration_min: 240
  required_agreement: 0.92
```

### 7.3 Shadow evaluation

Before any traffic shift, run the canary in **shadow mode**: requests routed to Production are mirrored to Canary; their outputs are diffed. Auto-promote only if agreement ≥ `required_agreement` (default 92%).

```python
# services/router/shadow.py
"""Mirror traffic to canary; record agreement, never serve canary's response."""
async def shadow_evaluate(prod_resp, canary_resp_promise):
    canary_resp = await canary_resp_promise
    _persist_agreement(prod_resp, canary_resp)
    # Returns nothing; metric scraped by Prometheus
```

Agreement metric (per inspection image):

- Same `status`?
- Same `fault_count`?
- Per-evidence Jaccard on `fused_label` set?

`pv_canary_agreement` Prometheus gauge feeds the auto-rollback decision.

### 7.4 One-command rollback

```python
# scripts/mlops/rollback.py
"""Demote current Production back to the previous version. Use in incidents."""
import argparse, mlflow
from mlflow.tracking import MlflowClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()
    c = MlflowClient()
    versions = sorted(c.get_latest_versions(args.name, stages=["Production", "Archived"]),
                       key=lambda v: -int(v.version))
    cur, prev = versions[0], versions[1]
    c.transition_model_version_stage(args.name, prev.version, "Production", archive_existing_versions=True)
    c.set_model_version_tag(args.name, prev.version, "rollback_reason", args.reason)
    print(f"Rolled back {args.name}: v{cur.version} → v{prev.version}")
```

---

## 8. Drift detection

### 8.1 What we monitor

| Signal | Source | Detector |
|---|---|---|
| Component class distribution | YOLO output histogram | χ² vs baseline |
| Classifier confidence histogram | per-prediction `top1_prob` | Kolmogorov-Smirnov vs baseline |
| Anomaly score distribution | PatchCore + EfficientAD scores | Population stability index (PSI) |
| Image stats (brightness, sharpness) | preproc IQA outputs | Distribution shift (KS) |
| Component count per image | YOLO post-process | Outlier rate vs baseline |
| HITL review rate | Phase 6 `pv_review_rate` | Threshold + rolling-window EWMA |
| Per-tenant skew | Embedding distance to global centroid | Cosine drift |

### 8.2 Implementation with Evidently

```python
# services/drift/main.py
"""Daily drift report job using Evidently + custom industrial detectors."""
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd, json
from pathlib import Path
from datetime import datetime, timedelta
import yaml

def run_daily():
    cfg = yaml.safe_load(open("configs/mlops/drift.yaml"))
    baseline = pd.read_parquet(cfg["baseline_parquet"])
    recent = _load_recent_predictions(days=cfg["recent_window_days"])

    rpt = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    rpt.run(reference_data=baseline, current_data=recent)
    summary = rpt.as_dict()

    drifted = [m for m in summary["metrics"] if m.get("result", {}).get("dataset_drift")]
    out = Path("mlops/drift_alerts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z", "drift_features": drifted}) + "\n")
    rpt.save_html(f"mlops/drift_reports/{datetime.utcnow():%Y%m%d}.html")

def _load_recent_predictions(days: int) -> pd.DataFrame:
    """Pull from logs/decisions/*.jsonl + classifier audit logs."""
    ...
```

A Prefect schedule runs this daily; alerts above a threshold create `drift_alerts.jsonl` entries that trigger the retraining flow (§4).

---

## 9. Lineage and reproducibility

### 9.1 What gets recorded per served prediction

For every `ImageVerdict` written by Phase 6, append a row to `lineage/manifest.parquet`:

| Column | Source |
|---|---|
| `image_id` | Phase 6 |
| `verdict_id` | Phase 6 |
| `served_at` | Phase 6 |
| `tenant_id` | Request header |
| `component_detector_model_uri` | MLflow URI: `models:/powervision-component-detector/Production` |
| `component_detector_version` | int |
| `classifier_model_uri` | same |
| `classifier_version` | int |
| `classifier_temperature` | float |
| `anomaly_models_used` | list[str], per-component (e.g. `["patchcore@v3", "efficientad@v2"]`) |
| `rule_set_version` | str |
| `taxonomy_version` | str |
| `preproc_version` | str |
| `decision_schema_version` | str |
| `code_git_commit` | str |
| `router_canary_share` | float (the % at the time of request) |

Any served prediction is **fully reproducible** from these IDs alone: pull dataset version, pull model artifacts, replay the image through the decision unit, expect the same `ImageVerdict`.

### 9.2 Replay tool

```python
# scripts/mlops/replay.py
"""Reproduce a historical prediction. Pulls all needed artifacts and re-runs."""
import argparse, pandas as pd, mlflow, subprocess
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-id", required=True)
    args = ap.parse_args()
    lin = pd.read_parquet("lineage/manifest.parquet")
    row = lin[lin["verdict_id"] == args.verdict_id].iloc[0]
    # 1. checkout the code commit
    subprocess.check_call(["git", "checkout", row["code_git_commit"]])
    # 2. dvc pull at the data hash
    subprocess.check_call(["dvc", "checkout"])
    # 3. download model artifacts
    mlflow.artifacts.download_artifacts(row["component_detector_model_uri"], "_replay/component/")
    mlflow.artifacts.download_artifacts(row["classifier_model_uri"], "_replay/classifier/")
    # 4. re-run decision_unit on the stored image
    subprocess.check_call(["python", "scripts/mlops/_replay_runner.py",
                            "--verdict-id", args.verdict_id])
```

---

## 10. Per-tenant data isolation

Industrial customers commonly have NDA-protected images. Phase 8 enforces:

### 10.1 Policy file

```yaml
# policies/tenant_isolation.yaml
version: "1.0.0"
tenants:
  tenantA:
    dvc_remote: s3://tenantA-private/dvc
    consent_for_global_training: false   # data must not feed the global classifier
    consent_for_aggregate_metrics: true
    label_session_prefix: "tenantA_"
  tenantB:
    dvc_remote: s3://tenantB-private/dvc
    consent_for_global_training: true
    label_session_prefix: "tenantB_"
  default:
    dvc_remote: s3://powervision-datasets/v2
    consent_for_global_training: true
```

### 10.2 Enforcement

- `build_splits.py` (Phase 1) filters out images whose `device_id` belongs to a tenant with `consent_for_global_training: false` **from the global training set**. They train **only** tenant-private models (per-tenant LoRA adapters, §6).
- Cross-tenant prediction comparisons (drift, evaluation) only use tenants that opted into aggregate metrics.
- `harvest_hitl.py` writes the tenant prefix into `image_id`; verifier rejects ingestion if a tenant's data lands in another tenant's remote.

---

## 11. Cost & capacity

| Resource | Tracking | Alert threshold |
|---|---|---|
| GPU hours per training | MLflow tags + cloud billing API | per-pipeline monthly budget |
| LLM tokens | Phase 7 Prometheus exporters | per-tenant budget (Phase 7 §12) |
| Storage (S3) | DVC remote size | 80% of quota |
| HITL queue depth | Postgres count | > 1000 open |
| Drift alerts open | jsonl tail | > 5 unresolved 7 days |

Grafana dashboards under `dashboards/mlops/{overview.json, costs.json, drift.json, canary.json}`.

---

## 12. Incident runbook (excerpt)

```markdown
# docs/runbooks/mlops_incidents.md

## I-001  Canary spike in latency p95
1. Check `pv_decision_latency_ms_p95` for canary endpoint.
2. If >30% above baseline for >10 min, the router auto-rolls back canary_pct → 0.
3. If auto-rollback didn't engage:
   `python scripts/mlops/rollback.py --name powervision-fault-classifier --reason latency_p95`
4. Open MLflow run for the canary version, compare to previous prod run.
5. File ticket; do not redeploy until perf regression understood.

## I-002  Classifier accuracy drop reported by customer
1. Pull last 7d `evaluation_report.json` from Production model.
2. Pull last 7d drift report.
3. If review_rate > 25% → trigger HITL queue clearance.
4. If classifier metrics in registry drifted < gates → emergency rollback.
5. Open retraining flow with a higher `per_class_threshold` override.

## I-003  Data ingestion fails verify_taxonomy
1. `ingest_resolutions.py` exits with non-zero — fault id in label files unknown.
2. Run `python scripts/taxonomy/verify_taxonomy.py` to see exact mismatch.
3. Either bump taxonomy (publish new ID) and retry, or correct the resolution
   in the HITL queue and re-run harvest.
```

---

## 13. Code structure

```text
scripts/mlops/
├── harvest_hitl.py
├── ingest_resolutions.py
├── evaluate_triggers.py
├── promote_model.py
├── rollback.py
├── train_lora_adapter.py
├── replay.py
└── _replay_runner.py

services/
├── active_learning/main.py
├── orchestrator/flows.py         # Prefect flow
├── router/{main.py, shadow.py}
└── drift/main.py

configs/mlops/
├── triggers.yaml
├── drift.yaml
├── db.yaml
└── tenants.yaml

configs/router/routing.yaml

policies/tenant_isolation.yaml

powervision/active/
└── novelty.py

lineage/
├── manifest.parquet
└── replay/...

dashboards/mlops/
├── overview.json
├── costs.json
├── drift.json
└── canary.json

docs/runbooks/
└── mlops_incidents.md
```

### 13.1 Extended `dvc.yaml` stages

```yaml
stages:
  harvest_hitl:
    cmd: python scripts/mlops/harvest_hitl.py --since-days 7
    always_changed: true
    outs:
      - datasets_pending

  ingest_pending:
    cmd: |
      for d in datasets_pending/*/; do
        iid=$(basename "$d")
        test -f "$d.promoted" || python scripts/mlops/ingest_resolutions.py --ingestion-id "$iid"
      done
    deps:
      - datasets_pending
      - scripts/mlops/ingest_resolutions.py

  evaluate_triggers:
    cmd: python scripts/mlops/evaluate_triggers.py --json > mlops/last_triggers.json
    deps:
      - configs/mlops/triggers.yaml
      - scripts/mlops/evaluate_triggers.py
    outs:
      - mlops/last_triggers.json
```

---

## 14. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| HITL backlog never drains | Reviewers overloaded, retraining starves | Auto-prioritization by severity; per-tenant SLAs; weekly digest to reviewers |
| Drift detector false alarms | Retraining storm | Hysteresis: drift must persist 7 d before retraining; per-feature confidence thresholds |
| Promotion gate too strict | Improved models never promoted | Tracked as `gate_block_rate`; review monthly; relax thresholds with sign-off |
| Canary regresses but auto-rollback misses it | Service degraded | Manual rollback CLI; SRE on-call paged on `pv_canary_agreement` < 85% |
| Replay impossible (stored image deleted, S3 lifecycle) | Cannot audit | Lineage stores SHA + ensures storage retention class is `compliant` (≥ 1 year) |
| Tenant consent revoked retroactively | Data must be expunged | `scripts/mlops/expunge_tenant.py` removes from manifest + DVC + retrains affected pipelines |
| LoRA adapter conflict with new base | Adapter loads but math is off | Adapter version pinned to base version; loader refuses mismatch |
| Code commit lost (force push) | Replay fails | CI policy: force push to `main` forbidden; tag every promotion |
| Schema bump in `BatchReport` (Phase 6) | Phase 7 + dashboard break | `decision_schema_version` lineage column; Phase 7 dispatches by schema version |
| MLflow Postgres outage | Cannot promote / lookup | Read-replica + emergency fallback to `models/registry_fallback/` (local pinned) |
| Distributed retrain races | Two workers retrain same model | Prefect flow uses `concurrency-limit-tag` per pipeline |
| DVC remote quota exceeded | Push fails, retraining loop halts | Quarterly dataset compaction; archive older taxonomy versions to colder storage |

---

## 15. Phase 8 exit checklist

- [ ] `scripts/mlops/harvest_hitl.py` runs daily; staging directory populated.
- [ ] `scripts/mlops/ingest_resolutions.py` promotes staged data and passes `verify_taxonomy` + `verify_dataset_v2`.
- [ ] Prefect deployment `mlops-loop` scheduled and observable.
- [ ] MLflow registry contains every model from Phases 3–5, plus rule-set artifact.
- [ ] Promotion gates implemented for each registered model; tested with synthetic regressing checkpoints.
- [ ] Router service deployed; canary at 5% by default; shadow eval active.
- [ ] Drift service deployed; daily HTML report and `drift_alerts.jsonl`.
- [ ] Lineage parquet writes from the decision service per request.
- [ ] Replay tool reproduces a verdict end-to-end.
- [ ] Per-tenant DVC remotes wired and policy enforcement integrated into `build_splits.py`.
- [ ] Grafana dashboards live; alerts wired to on-call.
- [ ] Incident runbook published.
- [ ] Demo (`api/main.py`, `api/streamlit_app.py`) continues to serve unchanged — it is unaffected by the production MLOps path.

Phase 8 is **frozen** when all boxes are checked. The system is now production-ready: it ingests, learns, deploys, monitors, and rolls back automatically while remaining fully auditable from any served prediction back to the exact dataset commit, code commit, and model artifact that produced it.
