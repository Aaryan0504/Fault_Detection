# Phase 6 — Rule Engine and Fault Decision Unit

> **Scope:** Backend only — no model training. This phase implements the **deterministic, auditable** layer that converts raw model outputs (component detections, anomaly scores, classifier probabilities, thermal stats) into the final fault verdict per image and per batch. It hosts the **structural rule engine**, **score fusion**, **severity scoring**, **confidence gate**, and **human review escalation queue**. Every decision is reproducible from the persisted inputs and the versioned rule set.

---

## 1. Phase objective

Models alone cannot make a deployable verdict. The fault decision unit:

1. **Reconciles** three parallel signals — Phase 3 component detector + Phase 4 anomaly verdicts + Phase 5 fault classifier — into a single per-image verdict.
2. **Applies structural rules** the models do not learn from images alone (e.g., "every input terminal block must have 4 cables", "fan must be present for any UPS ≥ 10 kVA").
3. **Computes severity and urgency** using `taxonomy/severity_matrix.yaml` plus runtime context (thermal hotspot magnitude, fault count, criticality of UPS).
4. **Gates by confidence**: low-confidence verdicts are routed to a human-review queue with full evidence, and the labeled outcomes feed back into Phase 8.
5. **Aggregates N per-image verdicts** into a batch-level pass/fail with per-image breakdown.
6. **Emits a versioned, schema-validated `BatchReport`** consumed by Phase 7 (LLM report generator).

Deliverables:

| Deliverable | Artifact |
|---|---|
| Rule DSL (YAML) | `rules/rule_definitions.yaml` |
| Rule engine runtime | `powervision/decision/rule_engine.py` |
| Score fusion | `powervision/decision/fusion.py` |
| Severity scorer | `powervision/decision/severity.py` |
| Confidence gate | `powervision/decision/confidence_gate.py` |
| Image / batch decision API | `powervision/decision/decision_unit.py` |
| Pydantic schemas | `powervision/decision/schemas.py` |
| FastAPI service | `services/decision/main.py` |
| HITL queue | `powervision/decision/hitl_queue.py` (backed by Postgres or Redis) |
| Rule tests | `tests/decision/test_rules.py` |
| End-to-end pipeline test | `tests/decision/test_pipeline.py` |

---

## 2. Inputs and outputs

### 2.1 Inputs (per inspection image)

| Source | Provided by | Object |
|---|---|---|
| Image bytes + EXIF | API request | `bytes`, `dict` |
| Session/UPS metadata | API request | `SessionContext` (see §3.1) |
| Component detections | Phase 3 inference adapter | `list[Detection]` |
| Anomaly verdicts per component crop | Phase 4 fusion | `list[AnomalyVerdict]` |
| Classifier predictions per component crop | Phase 5 adapter | `list[ClassifierPrediction]` |
| Thermal metadata | Phase 2 `thermal_meta/<image_id>.json` | `dict` (optional) |
| Taxonomy | `taxonomy/*.yaml` | loaded at startup |
| Rule definitions | `rules/rule_definitions.yaml` | loaded at startup |
| Severity matrix | `taxonomy/severity_matrix.yaml` | loaded at startup |

### 2.2 Outputs

| Object | Type | Purpose |
|---|---|---|
| `ImageVerdict` | Pydantic model | Single image's verdict, persisted to PostgreSQL |
| `BatchReport` | Pydantic model | N-image aggregated report, handed to Phase 7 |
| `HitlTicket` | Pydantic model | Escalation entry queued for human review |
| `DecisionAudit` | JSON event | Append-only log of every rule evaluated and every score, in `logs/decisions/<batch_id>.jsonl` |

### 2.3 Format contract

`schemas.py` is the **single source of truth** for the JSON shape Phase 7 and the dashboard consume. Bumping any field in `schemas.py` requires:

- A version-suffixed copy (`schemas_v3.py`) maintained alongside the old one for ≥ 2 release cycles.
- A migration script `scripts/decision/migrate_schema.py`.

---

## 3. Schemas

### 3.1 `powervision/decision/schemas.py`

```python
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

SCHEMA_VERSION = "decision_v1.0.0"

class Severity(str, Enum):
    info = "info"; low = "low"; medium = "medium"; high = "high"; critical = "critical"

class VerdictStatus(str, Enum):
    fault = "fault"; clear = "clear"; review = "review"; rejected = "rejected"

class Bbox(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    @property
    def area(self): return max(0.0, (self.x2 - self.x1)) * max(0.0, (self.y2 - self.y1))

class SessionContext(BaseModel):
    session_id: str
    device_id: str
    ups_type_id: str
    captured_at: datetime
    engineer_id: str | None = None
    site_id: str | None = None

class ComponentObservation(BaseModel):
    component_id: int
    component_name: str
    bbox: Bbox
    detection_confidence: float
    crop_path: str | None = None

class AnomalySignal(BaseModel):
    model: Literal["patchcore", "efficientad", "fused"]
    score: float
    threshold: float
    triggered: bool

class ClassifierPrediction(BaseModel):
    top1_fault_id: int
    top1_fault_name: str
    top1_prob: float
    top3: list[dict]
    masked_by_routing: bool

class FaultEvidence(BaseModel):
    """All signals tied to ONE component instance."""
    component: ComponentObservation
    anomaly_signals: list[AnomalySignal] = Field(default_factory=list)
    classifier: ClassifierPrediction | None = None
    rule_hits:    list[RuleHit] = Field(default_factory=list)
    fused_score:  float
    fused_label:  str | None
    fused_severity: Severity | None
    fused_confidence: float

class RuleHit(BaseModel):
    rule_id: str
    rule_version: str
    description: str
    fault_id: int | None
    severity_override: Severity | None
    fired_at_path: str    # human-readable rule path for audit

class ImageVerdict(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    schema_version: str = SCHEMA_VERSION
    image_id: str
    session: SessionContext
    status: VerdictStatus
    overall_severity: Severity
    fault_count: int
    iqa_passed: bool
    rejection_reason: str | None = None
    evidence: list[FaultEvidence] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    thermal_meta: dict | None = None
    annotated_image_path: str | None = None
    heatmap_paths: list[str] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    model_versions: dict[str, str]    # {component_detector: "v3", classifier: "v2", ...}

class BatchReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    batch_id: str
    session: SessionContext
    n_images: int
    n_faults: int
    n_clear: int
    n_review: int
    n_rejected: int
    overall_status: VerdictStatus              # fault if ANY image has fault, etc.
    overall_severity: Severity                 # max severity across images
    images: list[ImageVerdict]
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    model_versions: dict[str, str]
    rule_version: str
    taxonomy_version: str

class HitlTicket(BaseModel):
    ticket_id: str
    image_verdict: ImageVerdict
    reason: str
    priority: Literal["low", "normal", "high"]
    created_at: datetime
    assigned_to: str | None = None
    resolved: bool = False
    resolution: dict | None = None
```

---

## 4. Rule engine — DSL design

### 4.1 Why a DSL (and not Python plugins)?

- Rules change far more often than model code and must be auditable by non-engineers (field SMEs).
- A typed YAML DSL is human-reviewable, diffable, version-controlled, and unit-testable.
- A constrained DSL prevents arbitrary code execution — important when SMEs ship rules to air-gapped sites.

### 4.2 `rules/rule_definitions.yaml`

```yaml
version: "1.0.0"
rule_set_id: powervision_default

# Each rule has:
#   id, scope, when (predicates), then (consequences), severity, description
# All numeric comparisons are JSON-pointer / dotted-path references against
# the IMAGE EVALUATION CONTEXT (see §4.3).

rules:
  # ---------- STRUCTURAL: component count ----------
  - id: R001_input_cable_count
    scope: image
    description: "Every input_terminal_block must have 3 or 4 cable_run_input."
    when:
      all:
        - {comp_count: input_terminal_block, ge: 1}
        - {comp_count: cable_run_input, lt_per_input_terminal_block: 3}
    then:
      fault_id: 110          # ground_bar_loose? wait — use input_cable_fault adjacency
      fault_name: input_cable_fault
      severity: high
      message: "Expected 3 input cables, found {count}."

  - id: R002_battery_strap_present
    scope: image
    description: "If any battery_terminal_post is detected, ≥ 1 battery_strap must be detected too."
    when:
      all:
        - {comp_count: battery_terminal_post, ge: 2}
        - {comp_count: battery_strap, lt: 1}
    then:
      fault_id: 49
      fault_name: battery_strap_missing
      severity: high

  - id: R003_fan_required_for_large_ups
    scope: image
    description: "UPS ≥ 10 kVA must show at least one cooling_fan in front_open view."
    when:
      all:
        - {ups_kva: {ge: 10}}
        - {capture_view: front_open}
        - {comp_count: cooling_fan, eq: 0}
    then:
      fault_id: 103
      fault_name: fan_cable_unplugged          # placeholder; per UPS subtype
      severity: medium
      message: "Cooling fan expected but not detected. Check Phase 3 missed-detection or fan removed."

  # ---------- COLOR / POSITION: wiring sanity ----------
  - id: R010_red_white_polarity
    scope: image
    description: "Cable color pair around battery_terminal_post must be (red, white) with red on positive."
    when:
      all:
        - {comp_count: battery_terminal_post, ge: 1}
        - {color_pair_violation: battery}
    then:
      fault_id: 7
      fault_name: red_white_mismatch
      severity: high

  # ---------- THERMAL ----------
  - id: R020_terminal_hotspot
    scope: per_evidence
    component_filter: [input_terminal_block, output_terminal_block, battery_terminal_post, bus_bar_segment]
    description: "Component bbox max temperature > 20°C above image mean → likely loose connection."
    when:
      all:
        - {thermal_available: true}
        - {component_thermal_delta_c: {ge: 20}}
    then:
      fault_id: 1
      fault_name: loose_connection
      severity: high
      message: "Hotspot Δ{delta_c:.1f}°C above ambient on {component_name}."

  - id: R021_critical_overheat
    scope: per_evidence
    description: "Any component > 90°C is critical."
    when:
      all:
        - {thermal_available: true}
        - {component_max_temp_c: {ge: 90.0}}
    then:
      severity_override: critical

  # ---------- PRESENCE / ABSENCE ----------
  - id: R030_ground_cable_required
    scope: image
    description: "Every UPS panel must show a ground_bar component."
    when:
      all:
        - {capture_view: front_open}
        - {comp_count: ground_bar, eq: 0}
    then:
      fault_id: 110
      fault_name: ground_bar_loose         # detector missed → either physically missing or misframed
      severity: medium

  # ---------- META: misframing / IQA ----------
  - id: R900_misframed
    scope: image
    description: "If component count is 0 and image area > 0.4 covered by glass/cabinet panel only → reject."
    when:
      all:
        - {comp_count: any, eq: 0}
        - {iqa_passed: true}
    then:
      reject_with: "Misframed — no UPS components detected in IQA-passing image."

  - id: R901_low_quality
    scope: image
    description: "IQA failed → reject and request recapture."
    when: {iqa_passed: false}
    then:
      reject_with: "Image quality below threshold. Recapture required."

  # ---------- UPS-TYPE-AWARE WHITELIST ----------
  - id: R950_fault_not_applicable_to_ups
    scope: per_evidence
    description: "Suppress classifier predictions for faults not whitelisted for this ups_type_id."
    when:
      all:
        - {classifier_top1: present}
        - {fault_not_in_ups_whitelist: true}
    then:
      suppress_evidence: true
      audit_only: true
```

### 4.3 Image evaluation context (the rule input)

The rule engine evaluates against a flat dict assembled per image:

```python
{
  "image_id": "S_...__F001",
  "ups_type_id": "ups_type_001",
  "ups_kva": 10,
  "capture_view": "front_open",
  "iqa_passed": True,
  "comp_count": {                       # component_name → int
    "input_terminal_block": 1,
    "cable_run_input": 3,
    "battery_terminal_post": 8,
    "battery_strap": 4,
    "cooling_fan": 1,
    # ... 30 entries possibly zero ...
    "any": 17,                          # sum across all components
  },
  "thermal_available": True,
  "image_mean_temp_c": 38.4,
  "image_max_temp_c": 71.2,
  "color_pair_violation": {"battery": False, "input": False},
  "fault_whitelist": [0, 1, 2, ..., 24, 45, ...],     # from taxonomy/ups_system_types.yaml
}
```

Per-evidence context adds component-level fields:

```python
{
  "component_id": 12,
  "component_name": "electrolytic_capacitor",
  "component_max_temp_c": 88.3,
  "component_thermal_delta_c": 50.1,    # max temp inside bbox - image mean
  "anomaly_fused_score": 0.71,
  "anomaly_triggered": True,
  "classifier_top1_name": "rectifier_capacitor_bulge",
  "classifier_top1_prob": 0.83,
  "classifier_top1": "present",
  "fault_not_in_ups_whitelist": False,
}
```

### 4.4 Rule engine implementation

```python
# powervision/decision/rule_engine.py
"""Constrained DSL evaluator. NEVER eval/exec. Only typed predicates."""
from __future__ import annotations
from dataclasses import dataclass
import yaml, re
from pathlib import Path
from typing import Any
from .schemas import Severity, RuleHit

@dataclass(frozen=True)
class CompiledRule:
    id: str; description: str; scope: str
    when: dict; then: dict
    component_filter: list[str] | None = None
    version: str = ""

class RuleEngine:
    def __init__(self, rules_path: Path):
        raw = yaml.safe_load(rules_path.read_text())
        self.version = raw["version"]
        self.rules: list[CompiledRule] = [
            CompiledRule(r["id"], r.get("description", ""), r.get("scope", "image"),
                          r.get("when", {}), r.get("then", {}),
                          r.get("component_filter"), self.version)
            for r in raw["rules"]
        ]

    # ---- Public API ----
    def evaluate_image(self, ctx: dict) -> list[RuleHit]:
        return [self._fire(r, ctx) for r in self.rules
                if r.scope == "image" and self._matches(r.when, ctx)]

    def evaluate_evidence(self, evidence_ctx: dict, image_ctx: dict) -> list[RuleHit]:
        hits = []
        merged = {**image_ctx, **evidence_ctx}
        for r in self.rules:
            if r.scope != "per_evidence": continue
            if r.component_filter and evidence_ctx.get("component_name") not in r.component_filter:
                continue
            if self._matches(r.when, merged):
                hits.append(self._fire(r, merged))
        return hits

    # ---- Predicate engine ----
    _OPS = {"eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b,
            "in": lambda a, b: a in b}

    def _matches(self, when: Any, ctx: dict) -> bool:
        if not when: return True
        if isinstance(when, dict):
            # boolean combinators
            if "all" in when: return all(self._matches(c, ctx) for c in when["all"])
            if "any" in when: return any(self._matches(c, ctx) for c in when["any"])
            if "not" in when: return not self._matches(when["not"], ctx)
            # single predicate dict
            return all(self._match_predicate(k, v, ctx) for k, v in when.items())
        return bool(when)

    def _match_predicate(self, key: str, val: Any, ctx: dict) -> bool:
        # comp_count: <name>, op spec
        if key == "comp_count":
            # short forms: comp_count: input_terminal_block (with second key in same dict)
            # but for clarity we accept dict form too
            raise RuntimeError("comp_count is a special predicate; use lt/le/eq/ge inside a dict at the call site")
        if key.startswith("comp_count_"):
            name = key[len("comp_count_"):]
            return self._compare(ctx["comp_count"].get(name, 0), val)
        if key == "ups_kva":   return self._compare(ctx.get("ups_kva", 0), val)
        if key == "iqa_passed":return ctx.get("iqa_passed", False) is val
        if key == "capture_view": return ctx.get("capture_view") == val
        if key == "thermal_available": return ctx.get("thermal_available", False) is val
        if key == "component_thermal_delta_c":
            return self._compare(ctx.get("component_thermal_delta_c", 0.0), val)
        if key == "component_max_temp_c":
            return self._compare(ctx.get("component_max_temp_c", 0.0), val)
        if key == "color_pair_violation":
            return ctx.get("color_pair_violation", {}).get(val, False)
        if key == "classifier_top1": return val == "present" and ctx.get("classifier_top1_name") is not None
        if key == "fault_not_in_ups_whitelist":
            f = ctx.get("classifier_top1_fault_id")
            return (f is not None and f not in set(ctx.get("fault_whitelist", []))) is val
        # generic dotted-path get
        return self._compare(ctx.get(key), val)

    @classmethod
    def _compare(cls, value: Any, spec: Any) -> bool:
        if isinstance(spec, dict):
            return all(cls._OPS[op](value, target) for op, target in spec.items())
        return value == spec

    def _fire(self, r: CompiledRule, ctx: dict) -> RuleHit:
        then = r.then
        return RuleHit(
            rule_id=r.id, rule_version=r.version, description=r.description,
            fault_id=then.get("fault_id"),
            severity_override=Severity(then["severity_override"]) if "severity_override" in then else None,
            fired_at_path=f"{r.scope}/{r.id}",
        )
```

### 4.5 Special predicate: `comp_count` with `lt_per_input_terminal_block`

Some rules need ratios (cables per terminal). Implement these as **named helpers** registered at engine init:

```python
# powervision/decision/rule_helpers.py
def cables_per_input_terminal_block(image_ctx) -> float:
    n_tb = image_ctx["comp_count"].get("input_terminal_block", 0)
    n_c  = image_ctx["comp_count"].get("cable_run_input", 0)
    return (n_c / n_tb) if n_tb > 0 else float("inf")
```

Rules use them via:

```yaml
when:
  all:
    - {helper: cables_per_input_terminal_block, lt: 3.0}
```

The engine resolves `helper` via a lookup table at startup; **only registered helper names are callable** (no eval).

---

## 5. Score fusion

### 5.1 Per-component fusion algorithm

```python
# powervision/decision/fusion.py
"""Fuse classifier + anomaly + rule signals for ONE component instance."""
from __future__ import annotations
from .schemas import (FaultEvidence, ComponentObservation, AnomalySignal,
                       ClassifierPrediction, RuleHit, Severity)
from .severity import severity_from_taxonomy, max_severity

def fuse_evidence(component: ComponentObservation,
                   anomaly_signals: list[AnomalySignal],
                   classifier:      ClassifierPrediction | None,
                   rule_hits:       list[RuleHit],
                   taxonomy_index:  dict[int, dict],
                   per_class_thresholds: dict[str, float]) -> FaultEvidence:
    """Weighted fusion:
        score_a = max anomaly_signal.score / (2 * threshold)     ∈ [0, ∞)
        score_c = classifier.top1_prob if classifier present else 0
        score_r = 1.0 if any rule_hit with fault_id else 0
        fused   = max(score_a_clipped_to_1, score_c, score_r)

    Decision logic:
        if rule_hit with fault_id → label = rule's fault_name (rules win over models)
        elif classifier.top1_prob >= per_class_threshold[classifier.top1] → label = classifier.top1
        elif any anomaly_signal.triggered → label = "unknown_anomaly" (Phase 4 catches novel)
        else → no fault on this component
    """
    score_a = max((s.score / (2 * s.threshold) for s in anomaly_signals if s.threshold > 0), default=0.0)
    score_a = min(1.0, score_a)
    score_c = classifier.top1_prob if classifier else 0.0
    score_r = 1.0 if any(h.fault_id is not None for h in rule_hits) else 0.0
    fused_score = max(score_a, score_c, score_r)

    label = None; severity = None
    rule_with_fault = next((h for h in rule_hits if h.fault_id is not None), None)
    if rule_with_fault:
        label = taxonomy_index[rule_with_fault.fault_id]["name"]
        severity = severity_from_taxonomy(rule_with_fault.fault_id, taxonomy_index)
    elif classifier:
        thr = per_class_thresholds.get(classifier.top1_fault_name,
                                         per_class_thresholds["__default__"])
        if classifier.top1_prob >= thr and classifier.top1_fault_id != _NO_FAULT_ID:
            label = classifier.top1_fault_name
            severity = severity_from_taxonomy(classifier.top1_fault_id, taxonomy_index)
    elif any(s.triggered for s in anomaly_signals):
        label = "unknown_anomaly"
        severity = Severity.medium     # default for unknown

    # Apply rule severity overrides
    overrides = [h.severity_override for h in rule_hits if h.severity_override]
    if overrides:
        severity = max_severity([severity] + overrides) if severity else max_severity(overrides)

    return FaultEvidence(
        component=component,
        anomaly_signals=anomaly_signals,
        classifier=classifier,
        rule_hits=rule_hits,
        fused_score=float(fused_score),
        fused_label=label,
        fused_severity=severity,
        fused_confidence=_confidence(score_a, score_c, score_r, len(rule_hits)),
    )

_NO_FAULT_ID = 120   # sentinel; matches no_fault in fault_taxonomy

def _confidence(score_a: float, score_c: float, score_r: float, n_rules: int) -> float:
    """Confidence increases when multiple modalities agree."""
    agreement_boost = 0.10 * (int(score_a >= 0.5) + int(score_c >= 0.5) + int(score_r >= 0.5))
    base = 0.5 * score_c + 0.3 * score_a + 0.2 * (1.0 if n_rules else 0.0)
    return min(1.0, base + agreement_boost)
```

### 5.2 Per-image aggregation

```python
def aggregate_image(evidences: list[FaultEvidence]) -> tuple[VerdictStatus, Severity, int]:
    faults = [e for e in evidences if e.fused_label and e.fused_label != "no_fault"]
    if not faults:
        return VerdictStatus.clear, Severity.info, 0
    sev = max_severity([e.fused_severity for e in faults if e.fused_severity])
    return VerdictStatus.fault, sev, len(faults)
```

---

## 6. Severity scoring

`taxonomy/severity_matrix.yaml` provides the baseline (Phase 1 §3.5). Runtime adjustments:

| Adjustment | Trigger | Effect |
|---|---|---|
| Thermal extreme override | any rule with `severity_override` fires | clamp to max(current, override) |
| Multi-fault compounding | image fault count ≥ 3 | bump severity one tier (e.g., `medium` → `high`) |
| UPS criticality multiplier | `ups_type_id` flagged `critical_load: true` in `ups_system_types.yaml` | bump severity one tier |
| Battery thermal runaway | any battery component with > 70°C max | force `critical` |

```python
# powervision/decision/severity.py
import yaml
from pathlib import Path
from .schemas import Severity

_ORDER = [Severity.info, Severity.low, Severity.medium, Severity.high, Severity.critical]
_RANK  = {s: i for i, s in enumerate(_ORDER)}

def max_severity(sevs: list[Severity]) -> Severity:
    return max((s for s in sevs if s is not None),
                key=lambda s: _RANK[s], default=Severity.info)

def bump(sev: Severity, steps: int = 1) -> Severity:
    return _ORDER[min(len(_ORDER) - 1, _RANK[sev] + steps)]

def severity_from_taxonomy(fault_id: int, taxonomy_index: dict) -> Severity:
    return Severity(taxonomy_index[fault_id]["severity"])

class SeverityScorer:
    def __init__(self, severity_matrix_path: Path, ups_types_path: Path):
        sm = yaml.safe_load(severity_matrix_path.read_text())
        self.overrides = sm.get("overrides", {})
        self.ups_types = {u["id"]: u for u in yaml.safe_load(ups_types_path.read_text())["ups_types"]}

    def apply_runtime(self, base: Severity, ctx: dict, n_faults: int) -> Severity:
        sev = base
        if n_faults >= 3: sev = bump(sev, 1)
        ups = self.ups_types.get(ctx.get("ups_type_id"))
        if ups and ups.get("critical_load"): sev = bump(sev, 1)
        if ctx.get("battery_max_temp_c", 0) >= 70: sev = Severity.critical
        return sev
```

---

## 7. Confidence gate and HITL escalation

### 7.1 Gate logic

For each `ImageVerdict`:

```python
# powervision/decision/confidence_gate.py
from .schemas import ImageVerdict, VerdictStatus, HitlTicket, Severity
from datetime import datetime
import uuid

class ConfidenceGate:
    def __init__(self, conf_threshold: float = 0.60,
                 critical_conf_threshold: float = 0.50,
                 review_when_disagree: bool = True):
        self.thr = conf_threshold
        self.thr_crit = critical_conf_threshold
        self.review_when_disagree = review_when_disagree

    def gate(self, verdict: ImageVerdict) -> ImageVerdict:
        if verdict.status == VerdictStatus.rejected:
            return verdict
        for ev in verdict.evidence:
            if ev.fused_label is None: continue
            thr = self.thr_crit if ev.fused_severity == Severity.critical else self.thr
            if ev.fused_confidence < thr:
                verdict.status = VerdictStatus.review
                return verdict
            if self.review_when_disagree and self._models_disagree(ev):
                verdict.status = VerdictStatus.review
                return verdict
        return verdict

    @staticmethod
    def _models_disagree(ev) -> bool:
        a_fires = any(s.triggered for s in ev.anomaly_signals)
        c_fires = ev.classifier is not None and ev.classifier.top1_fault_name != "no_fault"
        # Disagree if one says fault and the other says clear AND confidence is borderline
        return (a_fires != c_fires) and 0.40 <= ev.fused_confidence <= 0.70
```

### 7.2 Routing to review

```python
# powervision/decision/hitl_queue.py
"""Append a HitlTicket to a durable queue. Backed by Postgres (default) or Redis."""
from __future__ import annotations
import json
from datetime import datetime
import uuid
from .schemas import ImageVerdict, HitlTicket, VerdictStatus, Severity

class HitlQueue:
    def __init__(self, backend):              # implements push / pop / list
        self.backend = backend

    def maybe_enqueue(self, verdict: ImageVerdict) -> HitlTicket | None:
        if verdict.status != VerdictStatus.review:
            return None
        priority = "high" if verdict.overall_severity in {Severity.high, Severity.critical} else "normal"
        ticket = HitlTicket(
            ticket_id=f"H-{uuid.uuid4().hex[:12]}",
            image_verdict=verdict,
            reason=self._reason(verdict),
            priority=priority,
            created_at=datetime.utcnow(),
        )
        self.backend.push(ticket.model_dump(mode="json"))
        return ticket

    @staticmethod
    def _reason(verdict: ImageVerdict) -> str:
        low_conf = [e for e in verdict.evidence if e.fused_confidence < 0.6]
        if low_conf:
            return f"Low confidence on {len(low_conf)} evidence(s); needs human review."
        return "Model disagreement; needs human review."
```

Backends are interchangeable; production default is Postgres (`hitl_tickets` table with JSONB payload + indexed `priority`, `created_at`, `resolved`). Phase 8 polls this table for retraining seeds.

### 7.3 HITL outcome ingestion

Reviewers' resolutions write back into the ticket and emit a `LabelEvent` consumed by Phase 8 (`scripts/hitl/ingest_resolutions.py`). Resolution schema:

```json
{
  "ticket_id": "H-ab12cd34ef56",
  "resolved": true,
  "resolution": {
    "is_fault": true,
    "fault_id": 9,
    "fault_name": "rectifier_capacitor_bulge",
    "bbox_corrections": [{"component_id": 12, "bbox_xyxy": [10, 20, 110, 220]}],
    "annotator_id": "rev_007",
    "resolved_at": "2026-05-24T15:42:11Z",
    "notes": "Genuine bulge; model under-confident due to glare."
  }
}
```

---

## 8. Decision unit — end-to-end

```python
# powervision/decision/decision_unit.py
from __future__ import annotations
import time, json, uuid
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml

from powervision.detect.component_detector import ComponentDetector
from powervision.classify.fault_classifier import FaultClassifier
from powervision.anomaly.patchcore   import PatchCore
from powervision.anomaly.efficientad import EfficientAD
from powervision.anomaly.fusion      import fuse as fuse_anomaly
from powervision.preproc.pipeline    import serve_preprocess_yolo
from .schemas import (BatchReport, ImageVerdict, SessionContext, FaultEvidence,
                       ComponentObservation, AnomalySignal, ClassifierPrediction,
                       VerdictStatus, Bbox)
from .rule_engine import RuleEngine
from .fusion import fuse_evidence
from .severity import SeverityScorer, max_severity
from .confidence_gate import ConfidenceGate
from .hitl_queue import HitlQueue

class DecisionUnit:
    def __init__(self,
                 component_detector: ComponentDetector,
                 fault_classifier: FaultClassifier,
                 patchcore: dict[str, PatchCore],
                 efficientad: dict[str, EfficientAD],
                 rule_engine: RuleEngine,
                 severity_scorer: SeverityScorer,
                 confidence_gate: ConfidenceGate,
                 hitl_queue: HitlQueue,
                 taxonomy_index: dict,
                 per_class_thresholds: dict,
                 model_versions: dict[str, str]):
        self._assert_compatible_versions(component_detector, fault_classifier)
        self.det = component_detector
        self.clf = fault_classifier
        self.pc = patchcore
        self.ad = efficientad
        self.rules = rule_engine
        self.sev = severity_scorer
        self.gate = confidence_gate
        self.hitl = hitl_queue
        self.taxonomy = taxonomy_index
        self.per_class_thr = per_class_thresholds
        self.model_versions = model_versions

    def _assert_compatible_versions(self, det, clf):
        det_v = det.metadata.get("taxonomy_version")
        clf_v = clf.metadata.get("taxonomy_version")
        if det_v != clf_v:
            raise RuntimeError(f"taxonomy_version mismatch: detector={det_v}, classifier={clf_v}")

    def decide_image(self, image_id: str, img_bgr, session: SessionContext,
                      thermal_meta: dict | None = None,
                      audit_writer=None) -> ImageVerdict:
        # 1. IQA + preprocessing
        _, iqa = serve_preprocess_yolo(img_bgr)
        iqa_passed = iqa["iqa_ok"]

        # 2. Component detection
        detections = self.det.predict(img_bgr) if iqa_passed else []
        crops      = self.det.crop_components(img_bgr) if iqa_passed else []

        # 3. Per-component evidence
        evidences: list[FaultEvidence] = []
        image_ctx = self._build_image_ctx(detections, session, thermal_meta, iqa_passed)
        image_rule_hits = self.rules.evaluate_image(image_ctx)

        for det_obj, crop_bgr in crops:
            name = det_obj.component_name
            anom_signals = []
            if name in self.pc:
                pc_s, _ = self.pc[name].score(crop_bgr)
                anom_signals.append(AnomalySignal(model="patchcore", score=pc_s,
                                                   threshold=self.pc[name].threshold,
                                                   triggered=pc_s >= self.pc[name].threshold))
            if name in self.ad:
                ad_s, _ = self.ad[name].score(crop_bgr)
                anom_signals.append(AnomalySignal(model="efficientad", score=ad_s,
                                                   threshold=self.ad[name].threshold,
                                                   triggered=ad_s >= self.ad[name].threshold))
            if anom_signals:
                fused = fuse_anomaly(
                    anom_signals[0].score if anom_signals[0].model == "patchcore" else 0,
                    self.pc[name].threshold if name in self.pc else 1.0,
                    next((s.score for s in anom_signals if s.model == "efficientad"), 0),
                    self.ad[name].threshold if name in self.ad else 1.0)
                anom_signals.append(AnomalySignal(model="fused", score=fused.fused_score,
                                                   threshold=0.5, triggered=fused.is_anomalous))

            clf_pred = ClassifierPrediction(**self.clf.predict(crop_bgr, component_name=name))
            evidence_ctx = self._build_evidence_ctx(det_obj, crop_bgr, anom_signals, clf_pred,
                                                     image_ctx, thermal_meta)
            ev_rule_hits = self.rules.evaluate_evidence(evidence_ctx, image_ctx)

            # Apply audit_only / suppress
            if any(h.fired_at_path.endswith("/R950_fault_not_applicable_to_ups") for h in ev_rule_hits):
                clf_pred = None

            comp_obs = ComponentObservation(
                component_id=det_obj.component_id, component_name=name,
                bbox=Bbox(x1=det_obj.bbox_xyxy[0], y1=det_obj.bbox_xyxy[1],
                           x2=det_obj.bbox_xyxy[2], y2=det_obj.bbox_xyxy[3]),
                detection_confidence=det_obj.confidence)

            ev = fuse_evidence(comp_obs, anom_signals, clf_pred,
                                ev_rule_hits + image_rule_hits,  # image rules apply to every evidence
                                self.taxonomy, self.per_class_thr)
            evidences.append(ev)

            if audit_writer:
                audit_writer.write(json.dumps({
                    "image_id": image_id, "component": name,
                    "anomaly": [s.model_dump() for s in anom_signals],
                    "classifier": clf_pred.model_dump() if clf_pred else None,
                    "rule_hits": [h.model_dump() for h in ev_rule_hits],
                    "fused_label": ev.fused_label,
                    "fused_confidence": ev.fused_confidence,
                }) + "\n")

        # 4. Aggregate verdict
        rejection_reason = None
        if not iqa_passed:
            status = VerdictStatus.rejected
            overall_sev = max_severity([])
            rejection_reason = "IQA failure"
        elif any(h.fired_at_path.endswith("/R900_misframed") for h in image_rule_hits):
            status = VerdictStatus.rejected
            overall_sev = max_severity([])
            rejection_reason = "Misframed — no components detected."
        else:
            fault_count = sum(1 for e in evidences if e.fused_label and e.fused_label != "no_fault")
            if fault_count == 0:
                status, overall_sev = VerdictStatus.clear, max_severity([])
            else:
                status = VerdictStatus.fault
                overall_sev = max_severity([e.fused_severity for e in evidences if e.fused_severity])
                overall_sev = self.sev.apply_runtime(overall_sev, image_ctx, fault_count)

        verdict = ImageVerdict(
            image_id=image_id, session=session, status=status,
            overall_severity=overall_sev,
            fault_count=sum(1 for e in evidences if e.fused_label and e.fused_label != "no_fault"),
            iqa_passed=iqa_passed,
            rejection_reason=rejection_reason,
            evidence=evidences,
            quality_flags=iqa["iqa"].get("flags", []) if isinstance(iqa["iqa"], dict) else [],
            thermal_meta=thermal_meta,
            model_versions=self.model_versions,
        )
        verdict = self.gate.gate(verdict)
        self.hitl.maybe_enqueue(verdict)
        return verdict

    def decide_batch(self, batch_id: str, session: SessionContext,
                      images: list[tuple[str, "np.ndarray", dict | None]]) -> BatchReport:
        started = datetime.utcnow()
        t0 = time.monotonic()
        audit_path = Path(f"logs/decisions/{batch_id}.jsonl")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        verdicts: list[ImageVerdict] = []
        with audit_path.open("w") as audit:
            for image_id, img, thermal in images:
                verdicts.append(self.decide_image(image_id, img, session,
                                                    thermal_meta=thermal, audit_writer=audit))
        finished = datetime.utcnow()
        return BatchReport(
            batch_id=batch_id, session=session,
            n_images=len(verdicts),
            n_faults=sum(1 for v in verdicts if v.status == VerdictStatus.fault),
            n_clear=sum(1 for v in verdicts if v.status == VerdictStatus.clear),
            n_review=sum(1 for v in verdicts if v.status == VerdictStatus.review),
            n_rejected=sum(1 for v in verdicts if v.status == VerdictStatus.rejected),
            overall_status=(VerdictStatus.fault if any(v.status == VerdictStatus.fault for v in verdicts)
                             else VerdictStatus.review if any(v.status == VerdictStatus.review for v in verdicts)
                             else VerdictStatus.clear),
            overall_severity=max_severity([v.overall_severity for v in verdicts]),
            images=verdicts,
            started_at=started, finished_at=finished,
            duration_ms=int((time.monotonic() - t0) * 1000),
            model_versions=self.model_versions,
            rule_version=self.rules.version,
            taxonomy_version=self.det.metadata.get("taxonomy_version", "unknown"),
        )

    # ---- Context builders (rule input) ----
    def _build_image_ctx(self, detections, session, thermal_meta, iqa_passed) -> dict:
        comp_count = {}
        for d in detections:
            comp_count[d.component_name] = comp_count.get(d.component_name, 0) + 1
        comp_count["any"] = sum(comp_count.values())
        ups = self.sev.ups_types.get(session.ups_type_id, {})
        return {
            "image_id": "",
            "ups_type_id": session.ups_type_id,
            "ups_kva": ups.get("rated_kva", 0),
            "capture_view": "unknown",
            "iqa_passed": iqa_passed,
            "comp_count": comp_count,
            "thermal_available": thermal_meta is not None,
            "image_mean_temp_c": thermal_meta.get("mean_c") if thermal_meta else 0.0,
            "image_max_temp_c":  thermal_meta.get("max_c")  if thermal_meta else 0.0,
            "color_pair_violation": {"battery": False, "input": False, "output": False},
            "fault_whitelist": ups.get("applicable_fault_ids", []),
            "battery_max_temp_c": 0.0,        # set by per-evidence loop if battery component sees hotspot
        }

    def _build_evidence_ctx(self, det_obj, crop, anom_signals, clf_pred, image_ctx, thermal_meta) -> dict:
        ctx = {
            "component_id": det_obj.component_id,
            "component_name": det_obj.component_name,
            "classifier_top1_name": clf_pred.top1_fault_name if clf_pred else None,
            "classifier_top1_fault_id": clf_pred.top1_fault_id if clf_pred else None,
            "classifier_top1_prob": clf_pred.top1_prob if clf_pred else 0.0,
            "anomaly_triggered": any(s.triggered for s in anom_signals),
        }
        # Thermal slice if thermal aligned
        if thermal_meta:
            ctx["component_max_temp_c"] = self._thermal_max_in_bbox(det_obj.bbox_xyxy, thermal_meta)
            ctx["component_thermal_delta_c"] = (
                ctx["component_max_temp_c"] - image_ctx["image_mean_temp_c"]
            )
        return ctx

    @staticmethod
    def _thermal_max_in_bbox(bbox, thermal_meta):
        # Wire to powervision.preproc.thermal.decode_thermal_png on the actual thermal file
        # if available; here we return a stub max for brevity.
        return float(thermal_meta.get("max_c", 0.0))
```

---

## 9. FastAPI service

```python
# services/decision/main.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import uuid, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np, cv2, yaml

from powervision.detect.component_detector import ComponentDetector
from powervision.classify.fault_classifier import FaultClassifier
from powervision.anomaly.patchcore   import PatchCore
from powervision.anomaly.efficientad import EfficientAD
from powervision.decision.rule_engine import RuleEngine
from powervision.decision.severity import SeverityScorer
from powervision.decision.confidence_gate import ConfidenceGate
from powervision.decision.hitl_queue import HitlQueue, PostgresBackend
from powervision.decision.decision_unit import DecisionUnit
from powervision.decision.schemas import SessionContext

app = FastAPI(title="PowerVision Decision Unit")

# ---- Boot loads everything once (60–90s startup) ----
@app.on_event("startup")
async def _boot():
    global UNIT
    det = ComponentDetector("runs/component_phase_b/weights/best.pt")
    clf = FaultClassifier("runs/classifier/eff_b4_phase_b")
    components = [c["name"] for c in yaml.safe_load(Path("taxonomy/component_taxonomy.yaml").read_text())["components"]]
    pc = {}; ad = {}
    for name in components:
        pc_dir = Path(f"runs/anomaly/patchcore/{name}")
        ad_dir = Path(f"runs/anomaly/efficientad/{name}")
        if pc_dir.exists(): pc[name] = PatchCore.from_run_dir(pc_dir)
        if ad_dir.exists(): ad[name] = EfficientAD.from_run_dir(ad_dir)
    rules = RuleEngine(Path("rules/rule_definitions.yaml"))
    sev = SeverityScorer(Path("taxonomy/severity_matrix.yaml"), Path("taxonomy/ups_system_types.yaml"))
    gate = ConfidenceGate()
    hitl = HitlQueue(PostgresBackend.from_env())
    taxonomy = {f["id"]: f for f in yaml.safe_load(Path("taxonomy/fault_taxonomy.yaml").read_text())["faults"]}
    per_class_thr = yaml.safe_load(Path("configs/classifier/per_class_thresholds.yaml").read_text())
    UNIT = DecisionUnit(det, clf, pc, ad, rules, sev, gate, hitl,
                         taxonomy, {**per_class_thr["overrides"], "__default__": per_class_thr["default"]},
                         model_versions={"component_detector": det.metadata.get("version", "v?"),
                                          "fault_classifier": clf.metadata.get("version", "v?"),
                                          "rule_set": rules.version})

@app.get("/health")
def health():
    return {"status": "ok", "rule_version": UNIT.rules.version,
            "taxonomy_version": UNIT.det.metadata.get("taxonomy_version")}

@app.post("/decide_image")
async def decide_image(file: UploadFile = File(...),
                        session_id: str = Form(...),
                        device_id: str = Form(...),
                        ups_type_id: str = Form(...)):
    buf = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Cannot decode image.")
    session = SessionContext(session_id=session_id, device_id=device_id,
                              ups_type_id=ups_type_id, captured_at=datetime.utcnow())
    verdict = UNIT.decide_image(image_id=str(uuid.uuid4()), img_bgr=img, session=session)
    return JSONResponse(verdict.model_dump(mode="json"))

@app.post("/decide_batch")
async def decide_batch(files: list[UploadFile] = File(...),
                        session_id: str = Form(...),
                        device_id: str = Form(...),
                        ups_type_id: str = Form(...)):
    imgs: list[tuple[str, np.ndarray, dict | None]] = []
    for f in files:
        buf = np.frombuffer(await f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None: continue
        imgs.append((f.filename, img, None))
    session = SessionContext(session_id=session_id, device_id=device_id,
                              ups_type_id=ups_type_id, captured_at=datetime.utcnow())
    report = UNIT.decide_batch(batch_id=str(uuid.uuid4()), session=session, images=imgs)
    return JSONResponse(report.model_dump(mode="json"))
```

Endpoint contract (`POST /decide_batch` response is a `BatchReport` JSON conforming to §3.1 schema).

---

## 10. Integration contracts

| Consumer | Contract |
|---|---|
| **Phase 7** (LLM report) | Receives `BatchReport` JSON; relies on `evidence[*].fused_label`, `fused_severity`, `rule_hits[*].description`, `thermal_meta`, `quality_flags`. |
| **Phase 8** (HITL retraining) | Pulls from `hitl_tickets` Postgres table; reads `image_verdict.evidence` for label correction targets. |
| **Phase 3/4/5** (model promotion) | Decision unit refuses to load if `taxonomy_version` mismatch (§8 assertion). |
| **Demo legacy** (`api/main.py`) | Continues to run independently of this service; production path goes through `services/decision/main.py`. |
| **External monitoring** | Decision unit emits OpenTelemetry spans (`decision_unit.decide_image`, `rule_engine.evaluate_image`) and Prometheus metrics (`pv_decision_latency_ms`, `pv_review_rate`, `pv_per_rule_fires`). |

---

## 11. Code structure

```text
powervision/decision/
├── __init__.py
├── schemas.py
├── rule_engine.py
├── rule_helpers.py
├── fusion.py
├── severity.py
├── confidence_gate.py
├── hitl_queue.py
├── decision_unit.py
└── backends/
    ├── postgres.py
    └── redis.py

rules/
├── rule_definitions.yaml
└── tests/
    ├── fixtures/
    │   ├── ctx_missing_input_cable.json
    │   ├── ctx_thermal_hotspot.json
    │   └── ...
    └── test_rules.py

services/decision/
├── main.py
└── Dockerfile

tests/decision/
├── test_rule_engine.py
├── test_fusion.py
├── test_severity.py
├── test_confidence_gate.py
└── test_pipeline_end_to_end.py
```

### 11.1 Extended `dvc.yaml` stages

```yaml
stages:
  rules_verify:
    cmd: python scripts/decision/verify_rules.py
    deps:
      - rules/rule_definitions.yaml
      - powervision/decision/rule_engine.py
      - rules/tests
    always_changed: false

  decision_integration_test:
    cmd: pytest tests/decision -x -q
    deps:
      - powervision/decision
      - runs/component_phase_b/weights/best.pt
      - runs/classifier/eff_b4_phase_b
      - runs/anomaly
      - rules/rule_definitions.yaml
```

---

## 12. Testing strategy

Phase 6 has no model training; correctness is enforced through tests.

| Test class | Examples |
|---|---|
| Rule-level unit tests | `test_rule_R001_input_cable_count`, `test_rule_R020_terminal_hotspot` — synthetic image_ctx dicts → expected rule_hits |
| Fusion unit tests | All combinations of (classifier_present, anomaly_triggered, rule_fault_id) → expected `fused_label` and `fused_confidence` |
| Severity tests | Compound severity cases (`high` + critical override → `critical`) |
| Confidence gate tests | Borderline confidence triggers review; critical fault with conf 0.51 still escalates |
| HITL queue tests | Push/pop/persist with both backends |
| End-to-end golden tests | 25 hand-curated images with frozen expected `BatchReport` JSON; CI diff-compares; any drift requires re-baseline + sign-off |
| Property-based tests (Hypothesis) | Random `ImageVerdict` should round-trip through `BatchReport` JSON without loss |

Golden corpus lives at `tests/decision/golden/{input,expected}/`.

---

## 13. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Rule explosion (200+ rules slow startup) | Service start > 30 s | Rules compiled once; helper LRU-cache; rule budget alarm in CI (n_rules ≤ 250) |
| Rule contradicts itself across versions | Verdict flips on the same image after deploy | Golden tests must update *intentionally*; CI fails on unexpected diff; PR review required |
| Per-class threshold too aggressive → high review rate | Reviewer queue overwhelmed | `pv_review_rate` Prometheus alert > 25% triggers threshold review; Phase 8 monitors weekly |
| Rule fires on wrong evidence (cross-component leakage) | `component_filter` mismatch in `per_evidence` rules | Lint script `scripts/decision/lint_rules.py` validates `component_filter` against `component_taxonomy.yaml` |
| Confidence gate too lenient → critical faults not escalated | Missed escalation | `critical_conf_threshold` separate and lower; tests assert critical-flagged evidence with conf < 0.50 always goes to review |
| Schema bump breaks Phase 7 | LLM report rejects payload | Schemas versioned (`SCHEMA_VERSION`), Phase 7 reads the version and dispatches to the right template |
| HITL backlog inflation | Queue grows unboundedly | Auto-archive resolved tickets older than 90 days; metric `pv_hitl_open` capped via alert |
| Auditability gap | Cannot reproduce a verdict | All inputs hashed and stored under `logs/decisions/<batch_id>/inputs/`; rule version + taxonomy version included in `BatchReport` |
| Time skew (server clock vs captured_at) | Late tickets mis-prioritized | Use `session.captured_at` for SLA timers, not `decided_at` |
| Postgres outage | HITL push fails → verdicts disappear | `HitlQueue` falls back to local SQLite mirror; replays on Postgres recovery |
| Latency regression after rule additions | p95 decision latency creeps | Per-rule timing in audit; rules > 5 ms median flagged for refactor |

---

## 14. Phase 6 exit checklist

- [ ] `powervision/decision/` package implemented and unit-tested.
- [ ] `rules/rule_definitions.yaml` v1.0.0 written and lints clean (`lint_rules.py`).
- [ ] Schemas (`schemas.py`) reviewed and frozen as `decision_v1.0.0`.
- [ ] Score fusion contract documented; consistent with Phase 4 §8.3 and Phase 5 §8.3.
- [ ] Severity matrix runtime overrides implemented and tested.
- [ ] Confidence gate covers low-confidence + critical-severity + model-disagreement paths.
- [ ] HITL queue with Postgres backend, indexed and migration-scripted.
- [ ] FastAPI service `/decide_image`, `/decide_batch`, `/health` work end-to-end.
- [ ] Golden tests (25 images) pass.
- [ ] Prometheus + OpenTelemetry instrumentation live.
- [ ] Demo (`api/main.py`, `api/streamlit_app.py`) unaffected.

Phase 6 is **frozen** when all boxes are checked. Phase 7 begins.
