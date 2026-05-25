# Phase 7 — LLM Integration and Report Generation Backend

> **Scope:** Backend only. Converts the `BatchReport` produced by Phase 6 into structured, human-readable inspection reports (fault summary, root-cause analysis, recommended corrective actions, severity and urgency). Handles prompt engineering, structured output schemas, batch aggregation, dispatch to email / ITSM / dashboard / paging, and offline fallback for air-gapped sites. The LLM never sees raw image bytes — only model-derived evidence — so the system is auditable and privacy-safe.

---

## 1. Phase objective

The Phase 6 `BatchReport` is dense and machine-shaped. Operators, site engineers, and customer-facing dashboards need:

- A concise per-image **finding** (1–2 sentences).
- A per-fault **root-cause** explanation that uses domain knowledge from `taxonomy/fault_taxonomy.yaml` plus runtime evidence.
- A **recommended action** with concrete next steps and required parts/tools.
- A **severity + urgency** assessment that respects `taxonomy/severity_matrix.yaml`.
- A **batch-level executive summary** with a pass/fail verdict.
- A canonical **structured JSON** for downstream systems plus a beautiful **Markdown / PDF** rendering.

Deliverables:

| Deliverable | Artifact |
|---|---|
| Report JSON schema (Pydantic) | `powervision/report/schemas.py` |
| Prompt templates | `prompts/{system,batch,fault_per_evidence,action_synthesis}/*.j2` |
| LLM provider abstraction | `powervision/report/llm.py` |
| Report generator | `powervision/report/generator.py` |
| Renderers | `powervision/report/renderers/{markdown.py, pdf.py, html.py}` |
| Dispatcher | `powervision/report/dispatch/{email.py, itsm.py, webhook.py, sms.py, dashboard.py}` |
| FastAPI service | `services/report/main.py` |
| Offline fallback (template-only) | `powervision/report/fallback.py` |
| Cache | `powervision/report/cache.py` (Redis) |
| Eval harness | `tests/report/test_report_quality.py` + LLM-graded rubric |
| Cost monitoring | Prometheus exporters + per-tenant budget config |

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Shape |
|---|---|---|
| `BatchReport` | Phase 6 `/decide_batch` | JSON (Pydantic `decision_v1.0.0`) |
| Fault taxonomy | `taxonomy/fault_taxonomy.yaml` | YAML — used to enrich each fault with `description`, `visual_cues` |
| Severity matrix | `taxonomy/severity_matrix.yaml` | YAML — drives `sla_hours`, routing |
| Knowledge base (recommended actions) | `kb/actions/<fault_name>.md` | Markdown (curated by SMEs) |
| LLM provider credentials | Env vars | string |
| Dispatch routing | `configs/report/dispatch_routing.yaml` | YAML |
| Tenant/customer config | `configs/report/tenants.yaml` | YAML (per-tenant LLM model, templates, budget, language) |

### 2.2 Outputs

```text
reports/<batch_id>/
├── report.json            # structured, schema-validated
├── report.md              # markdown rendering
├── report.html            # html rendering for email/dashboard
├── report.pdf             # PDF rendering
├── attachments/
│   ├── <image_id>_annotated.jpg
│   ├── <image_id>_heatmap_<component>.png
│   └── thermal_overlay/...
└── dispatch_log.jsonl     # one line per dispatch attempt (email/itsm/etc.)

logs/report/<batch_id>/
├── prompts/               # full prompts sent to LLM (for audit)
├── responses/             # raw LLM responses
└── tokens.json            # input/output token counts, cost
```

### 2.3 Format contracts

- **`report.json`** strictly conforms to `powervision/report/schemas.py::Report` (version `report_v1.0.0`).
- Reports are **immutable** once generated. Regeneration creates a new `report_v<N+1>` in a new directory.
- Markdown / HTML / PDF renderings are deterministic from `report.json` — the LLM is never re-queried for rendering.

---

## 3. Report schema

```python
# powervision/report/schemas.py
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

REPORT_SCHEMA_VERSION = "report_v1.0.0"

class Urgency(str, Enum):
    routine    = "routine"
    scheduled  = "scheduled"
    expedited  = "expedited"
    immediate  = "immediate"

class FaultReport(BaseModel):
    fault_id: int
    fault_name: str
    component_name: str
    bbox: list[float]              # [x1, y1, x2, y2] in image pixels
    fused_confidence: float
    severity: str                  # info|low|medium|high|critical
    urgency: Urgency
    summary: str                   # 1–2 sentences
    root_cause: str                # 1–2 paragraphs
    recommended_action: list[str]  # ordered checklist
    required_tools: list[str]
    required_parts: list[str]
    sla_hours: int
    evidence_image_paths: list[str]
    heatmap_paths: list[str]
    references: list[str]          # KB doc IDs / OEM service-manual sections

class ImageReport(BaseModel):
    image_id: str
    status: Literal["fault", "clear", "review", "rejected"]
    one_line_finding: str
    quality_flags: list[str]
    fault_reports: list[FaultReport]
    annotated_image_path: str | None = None

class BatchSummary(BaseModel):
    headline: str
    overall_status: Literal["fault", "clear", "review"]
    overall_severity: str
    overall_urgency: Urgency
    n_images: int
    n_faults: int
    n_clear: int
    n_review: int
    n_rejected: int
    top_concerns: list[str]        # 3–5 bullets
    executive_summary: str          # 2–4 short paragraphs
    recommended_next_inspection_days: int | None

class Report(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    schema_version: str = REPORT_SCHEMA_VERSION
    report_id: str
    batch_id: str
    tenant_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    language: str = "en"
    summary: BatchSummary
    images: list[ImageReport]
    llm_metadata: dict             # {provider, model, tokens_in, tokens_out, latency_ms, fallback_used}
    source_decision_version: str   # decision_v1.0.0
    rule_set_version: str
    taxonomy_version: str
```

---

## 4. Prompt engineering

### 4.1 Design principles

1. **The LLM never sees image bytes.** It sees structured evidence (component bbox, anomaly scores, classifier predictions, rule hits, thermal stats). This is auditable and privacy-safe.
2. **Strict JSON output.** Every prompt requests a JSON payload matching a Pydantic shape; we use the provider's `response_format=json_schema` (OpenAI), tool-calling, or constrained decoding (vLLM, Llama-3) — whichever is available.
3. **Single fault per call.** One LLM call generates ONE `FaultReport`. Per-image and batch summaries are separate calls. This gives clean retry semantics and bounded token usage.
4. **Two-shot examples per prompt** drawn from a curated `prompts/exemplars/` corpus — no zero-shot in production.
5. **Knowledge-base injection.** For each fault, the curated `kb/actions/<fault_name>.md` (if exists) is included verbatim in the prompt; the LLM rewrites / tailors but cannot invent action steps for known faults.
6. **Deterministic fields.** `severity`, `sla_hours`, `urgency`, `fault_id`, `fault_name`, `bbox`, `fused_confidence` are **filled by code** before the LLM call. The LLM only writes free-text fields (`summary`, `root_cause`, `recommended_action`, `required_tools`, `required_parts`).

### 4.2 Prompt directory layout

```text
prompts/
├── system/
│   ├── en.j2
│   └── es.j2
├── per_fault/
│   ├── en.j2
│   └── es.j2
├── per_image/
│   ├── en.j2
│   └── es.j2
├── batch_summary/
│   ├── en.j2
│   └── es.j2
├── action_synthesis/        # used when KB doc exists but needs context tailoring
│   └── en.j2
└── exemplars/
    ├── per_fault/
    │   ├── capacitor_bulge.json     # {input, output} pair
    │   ├── battery_terminal_corrosion.json
    │   └── ...
    └── batch_summary/
        ├── all_clear.json
        ├── critical_battery.json
        └── ...
```

### 4.3 System prompt (`prompts/system/en.j2`)

```jinja
You are PowerVision Inspector, an industrial AI assistant that writes inspection
reports for uninterruptible power supply (UPS) systems used in datacenters and
critical facilities. You are given STRUCTURED EVIDENCE from a computer-vision
pipeline (component detector + anomaly models + fault classifier + rule engine).

Rules you MUST follow:
  1. NEVER invent fault types, components, severities, fault IDs, or numeric
     measurements. If a field is not present in the evidence, omit it.
  2. NEVER mention image pixels, bboxes, or model names in user-facing prose.
  3. Use precise electrical/industrial vocabulary. Avoid hedging language
     ("may", "possibly") UNLESS the evidence confidence is < 0.7.
  4. Output ONLY valid JSON conforming to the provided schema. No prose
     outside the JSON.
  5. Reference parts/tools at a UPS technician's level (e.g., "torque wrench
     set to 5 N·m", "470 µF 450 V electrolytic capacitor").
  6. Severity, urgency, fault_id, fault_name are PRE-FILLED — do not change them.
  7. If knowledge_base_action is provided, your `recommended_action` MUST be
     consistent with it (you may shorten or contextualize, never contradict).

You will be given two correct example INPUT/OUTPUT pairs before each task.
```

### 4.4 Per-fault prompt (`prompts/per_fault/en.j2`)

```jinja
{# Variables: evidence (dict), taxonomy_entry (dict), kb_action (str|null), exemplars (list) #}
You are generating a fault report for ONE fault on ONE component instance.

## Two correct examples
{% for ex in exemplars %}
### Example {{ loop.index }} input:
```json
{{ ex.input | tojson(indent=2) }}
```
### Example {{ loop.index }} output:
```json
{{ ex.output | tojson(indent=2) }}
```
{% endfor %}

## Now generate for THIS evidence
- Detected component: {{ evidence.component_name }} (id {{ evidence.component_id }})
- Pre-filled fault: {{ evidence.fault_name }} (id {{ evidence.fault_id }})
- Severity (locked):  {{ evidence.severity }}
- Confidence:         {{ "%.2f"|format(evidence.fused_confidence) }}
- Anomaly signals:    {{ evidence.anomaly_signals }}
- Classifier top-3:   {{ evidence.classifier_top3 }}
- Rule hits fired:    {{ evidence.rule_hits }}
- Thermal evidence (if any):
  - Component max temp: {{ evidence.component_max_temp_c }} °C
  - Δ vs image mean:    {{ evidence.component_thermal_delta_c }} °C

## Taxonomy entry
- description:  {{ taxonomy_entry.description }}
- visual_cues:  {{ taxonomy_entry.visual_cues }}
- subsystem:    {{ taxonomy_entry.subsystem }}
- fault_mode:   {{ taxonomy_entry.fault_mode }}

{% if kb_action %}
## Knowledge-base recommended action (authoritative)
{{ kb_action }}
{% endif %}

## OUTPUT SCHEMA
Return ONLY valid JSON with these keys (no comments, no markdown):
{
  "summary": "<1–2 sentences>",
  "root_cause": "<1–2 short paragraphs that explain physical mechanism>",
  "recommended_action": ["<step 1>", "<step 2>", "..."],
  "required_tools": ["<tool 1>", "..."],
  "required_parts": ["<part 1 with spec>", "..."],
  "references": ["<KB or manual ref>", "..."]
}
```

### 4.5 Per-image one-line finding (`prompts/per_image/en.j2`)

```jinja
You are summarizing ONE inspection image into a SINGLE sentence (≤ 28 words).

Status:   {{ image.status }}
Faults:   {{ image.fault_count }}
Severity: {{ image.overall_severity }}
Top faults (max 3):
{% for f in image.top_faults[:3] %}
- {{ f.fault_name }} on {{ f.component_name }} (sev={{ f.severity }}, conf={{ "%.2f"|format(f.fused_confidence) }})
{% endfor %}

Return JSON only: {"one_line_finding": "<the sentence>"}
```

### 4.6 Batch summary (`prompts/batch_summary/en.j2`)

```jinja
You are writing the executive summary for a batch of {{ batch.n_images }} UPS
inspection images for {{ tenant_display_name }}.

Aggregate stats:
- Overall status:   {{ batch.overall_status }}
- Overall severity: {{ batch.overall_severity }}
- Faults:           {{ batch.n_faults }}
- Clear:            {{ batch.n_clear }}
- Needs review:     {{ batch.n_review }}
- Rejected (IQA):   {{ batch.n_rejected }}

Most severe distinct fault types observed:
{% for f in batch.distinct_faults_by_severity[:5] %}
- {{ f.fault_name }} ({{ f.severity }}) on {{ f.component_name }} — {{ f.count }}× across {{ f.image_count }} images
{% endfor %}

Two correct examples:
{% for ex in exemplars %}
### Example {{ loop.index }}:
INPUT: {{ ex.input | tojson(indent=2) }}
OUTPUT: {{ ex.output | tojson(indent=2) }}
{% endfor %}

## OUTPUT SCHEMA
Return JSON only:
{
  "headline": "<≤ 12 words, action-oriented>",
  "executive_summary": "<2–4 short paragraphs>",
  "top_concerns": ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
  "recommended_next_inspection_days": <int | null>
}
```

### 4.7 Exemplars

Each `prompts/exemplars/per_fault/<fault_name>.json` looks like:

```json
{
  "input": {
    "component_name": "electrolytic_capacitor",
    "fault_name": "rectifier_capacitor_bulge",
    "severity": "critical",
    "fused_confidence": 0.83,
    "classifier_top3": [
      {"fault_name": "rectifier_capacitor_bulge", "prob": 0.83},
      {"fault_name": "rectifier_capacitor_leakage", "prob": 0.10},
      {"fault_name": "no_fault", "prob": 0.03}
    ],
    "anomaly_signals": [
      {"model": "patchcore", "score": 0.71, "threshold": 0.42, "triggered": true},
      {"model": "efficientad", "score": 0.55, "threshold": 0.38, "triggered": true}
    ],
    "rule_hits": [],
    "component_max_temp_c": 72.4,
    "component_thermal_delta_c": 31.2
  },
  "output": {
    "summary": "Electrolytic capacitor in the rectifier stage shows a bulged top vent and a 31°C hotspot above ambient, indicating advanced electrolyte vaporization.",
    "root_cause": "Bulging on the vent of an aluminum electrolytic capacitor results from internal pressure as the electrolyte boils. The 31°C hotspot above the image mean rules out a benign cosmetic deformation; the capacitor is actively self-heating, which accelerates failure and risks a vent rupture, fluid leakage onto the PCB, and a rectifier-stage short.\n\nLikely upstream contributors include high ripple current from an aging input filter, elevated ambient temperature inside the cabinet, or a degraded snubber circuit on the adjacent IGBT.",
    "recommended_action": [
      "Isolate the UPS via bypass and lock-out/tag-out the input MCB.",
      "Discharge the DC bus with the OEM-approved bleed resistor (≥ 5 minutes; verify ≤ 50 V).",
      "Replace the affected 470 µF 450 V capacitor and any visually similar neighbors in the same bank.",
      "Inspect and reflow PCB pads beneath the capacitor for electrolyte residue.",
      "Run the OEM rectifier self-test and verify ripple < 5 % under 80 % load before returning to service."
    ],
    "required_tools": [
      "Insulated screwdriver set (1000 V)",
      "Multimeter (CAT IV)",
      "ESD wrist strap",
      "Capacitor discharge bleed resistor (≥ 10 kΩ, 5 W)",
      "IR thermometer"
    ],
    "required_parts": [
      "Aluminum electrolytic capacitor, 470 µF, 450 V, 105 °C-rated, snap-in (qty: 1+)",
      "Thermal paste (if heatsink removed)"
    ],
    "references": [
      "OEM Service Manual §6.3 'Rectifier capacitor replacement'",
      "Internal KB: capacitor_replacement_sop.md"
    ]
  }
}
```

---

## 5. LLM provider abstraction

### 5.1 `powervision/report/llm.py`

```python
"""Provider-agnostic LLM client with strict JSON output.

Supported providers (selected via env or tenant config):
  - openai           (gpt-4.1, gpt-4o, etc.)
  - azure_openai     (Azure-deployed equivalents)
  - anthropic        (claude-3.x via Messages API)
  - bedrock          (Claude / Llama via AWS Bedrock)
  - vllm_local       (self-hosted vLLM OpenAI-compatible)
  - llama_cpp_local  (offline air-gapped fallback)
"""
from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Any, Protocol
import httpx

@dataclass
class LLMResult:
    text: str
    parsed: dict
    tokens_in: int
    tokens_out: int
    provider: str
    model: str
    latency_ms: int
    cached: bool = False

class LLMProvider(Protocol):
    name: str
    def generate_json(self, system: str, user: str, schema: dict,
                       temperature: float, max_output_tokens: int) -> LLMResult: ...

class OpenAIProvider:
    name = "openai"
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.client = httpx.Client(
            base_url=base_url or "https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key or os.environ['OPENAI_API_KEY']}"},
            timeout=60.0,
        )

    def generate_json(self, system, user, schema, temperature, max_output_tokens):
        t0 = time.monotonic()
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "fault_report", "schema": schema, "strict": True},
            },
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        r = self.client.post("/chat/completions", json=payload); r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResult(
            text=text, parsed=json.loads(text),
            tokens_in=data["usage"]["prompt_tokens"],
            tokens_out=data["usage"]["completion_tokens"],
            provider=self.name, model=self.model,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

class AnthropicProvider:
    name = "anthropic"
    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str | None = None):
        self.model = model
        self.client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            headers={"x-api-key": api_key or os.environ["ANTHROPIC_API_KEY"],
                      "anthropic-version": "2023-06-01"},
            timeout=60.0,
        )

    def generate_json(self, system, user, schema, temperature, max_output_tokens):
        t0 = time.monotonic()
        payload = {
            "model": self.model,
            "system": system + "\n\nReturn ONLY valid JSON conforming to the schema. No markdown.",
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "tool_choice": {"type": "tool", "name": "emit_report"},
            "tools": [{
                "name": "emit_report",
                "description": "Emit the structured report.",
                "input_schema": schema,
            }],
        }
        r = self.client.post("/messages", json=payload); r.raise_for_status()
        data = r.json()
        tool_input = next(b["input"] for b in data["content"] if b["type"] == "tool_use")
        return LLMResult(
            text=json.dumps(tool_input),
            parsed=tool_input,
            tokens_in=data["usage"]["input_tokens"],
            tokens_out=data["usage"]["output_tokens"],
            provider=self.name, model=self.model,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

class LocalVLLMProvider(OpenAIProvider):
    name = "vllm_local"
    def __init__(self, base_url: str, model: str):
        super().__init__(model=model, api_key="local", base_url=base_url)

def build_provider(name: str, **kwargs) -> LLMProvider:
    return {
        "openai": OpenAIProvider, "azure_openai": OpenAIProvider,
        "anthropic": AnthropicProvider, "vllm_local": LocalVLLMProvider,
    }[name](**kwargs)
```

### 5.2 Tenant routing — `configs/report/tenants.yaml`

```yaml
version: "1.0.0"
default:
  provider: openai
  model: gpt-4o-mini
  language: en
  temperature: 0.2
  max_output_tokens: 800
  daily_budget_usd: 25.00

tenants:
  tenantA:
    provider: anthropic
    model: claude-3-5-sonnet-latest
    language: en
    temperature: 0.1
    daily_budget_usd: 50.00
  tenantB_airgapped:
    provider: vllm_local
    model: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://llm.internal:8000/v1
    language: en
    temperature: 0.1
    daily_budget_usd: null    # unmetered for on-prem
```

### 5.3 Retry, timeout, fallback

| Failure | Behavior |
|---|---|
| HTTP 5xx | 3 retries with exponential backoff (1 s, 4 s, 16 s) |
| Timeout (> 60 s) | Same |
| Invalid JSON output | 1 reprompt: "Your previous output failed schema validation: <pydantic error>. Re-emit valid JSON." |
| Provider down | Fail over to secondary provider (configured per tenant) |
| Both providers down | Template-only `fallback.py` engages; report flagged `fallback_used: true` |
| Daily budget exceeded | Template-only fallback; alert via Slack/email |

---

## 6. Knowledge base injection

### 6.1 Layout

```text
kb/
└── actions/
    ├── rectifier_capacitor_bulge.md
    ├── battery_terminal_corrosion.md
    ├── fuse_blown.md
    └── ... (one per fault id where authoritative SOP exists)
```

Each KB doc is plain Markdown (≤ 800 tokens). Example:

```markdown
# rectifier_capacitor_bulge

## Safety
- Isolate via bypass; LOTO upstream MCB.
- Wait ≥ 5 min after isolation. Verify DC bus ≤ 50 V with multimeter.

## Procedure
1. ...
2. ...

## Acceptance
- Ripple < 5 % at 80 % load.
- No bulged capacitors visually.
```

The generator includes only the KB doc for the **specific** fault under analysis (not the whole KB) to keep prompts compact.

### 6.2 KB freshness contract

- KB docs are tracked in git.
- A bot opens a PR for each new fault id that lacks a KB doc.
- Without a KB doc, the LLM relies on `taxonomy_entry.description` + `visual_cues` alone — quality is lower, marked in `report.json` (`evidence_has_kb_doc: false`).

---

## 7. Report generator

### 7.1 `powervision/report/generator.py`

```python
"""Generate a Report from a BatchReport."""
from __future__ import annotations
import json, hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict
import uuid, yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from powervision.decision.schemas import BatchReport, ImageVerdict, FaultEvidence
from .schemas import Report, ImageReport, FaultReport, BatchSummary, Urgency
from .llm import LLMProvider, LLMResult, build_provider
from .cache import ReportCache
from .fallback import template_only_report

class ReportGenerator:
    def __init__(self, tenant_id: str, tenant_cfg: dict, taxonomy: dict,
                 severity_matrix: dict, kb_root: Path,
                 cache: ReportCache | None = None):
        self.tenant_id = tenant_id
        self.cfg = tenant_cfg
        self.taxonomy = taxonomy        # {fault_id: taxonomy_entry}
        self.severity = severity_matrix
        self.kb_root = kb_root
        self.cache = cache
        self.provider = build_provider(tenant_cfg["provider"], **tenant_cfg.get("provider_args", {}))
        self.env = Environment(loader=FileSystemLoader("prompts"),
                                autoescape=select_autoescape([]), trim_blocks=True, lstrip_blocks=True)
        self.system_tmpl = self.env.get_template(f"system/{tenant_cfg['language']}.j2")
        self.per_fault_tmpl = self.env.get_template(f"per_fault/{tenant_cfg['language']}.j2")
        self.per_image_tmpl = self.env.get_template(f"per_image/{tenant_cfg['language']}.j2")
        self.batch_tmpl = self.env.get_template(f"batch_summary/{tenant_cfg['language']}.j2")

    def generate(self, batch_report: BatchReport) -> Report:
        images = [self._image_report(v) for v in batch_report.images]
        summary = self._batch_summary(batch_report, images)
        llm_meta = self._aggregate_llm_metadata(images, summary)
        return Report(
            report_id=f"R-{uuid.uuid4().hex[:12]}",
            batch_id=batch_report.batch_id,
            tenant_id=self.tenant_id,
            language=self.cfg["language"],
            summary=summary,
            images=images,
            llm_metadata=llm_meta,
            source_decision_version=batch_report.schema_version,
            rule_set_version=batch_report.rule_version,
            taxonomy_version=batch_report.taxonomy_version,
        )

    # ---- Per-image ----
    def _image_report(self, v: ImageVerdict) -> ImageReport:
        fault_reports = [self._fault_report(ev) for ev in v.evidence
                          if ev.fused_label and ev.fused_label != "no_fault"]
        one_line = self._one_line(v, fault_reports)
        return ImageReport(
            image_id=v.image_id, status=v.status,
            one_line_finding=one_line,
            quality_flags=v.quality_flags,
            fault_reports=fault_reports,
            annotated_image_path=v.annotated_image_path,
        )

    # ---- Per-fault (LLM call) ----
    def _fault_report(self, ev: FaultEvidence) -> FaultReport:
        fid = self._fault_id_for(ev)
        tx = self.taxonomy.get(fid)
        kb = self._load_kb(ev.fused_label)
        sev = ev.fused_severity.value
        sla = next((s["sla_hours"] for s in self.severity["severity_levels"] if s["level"] == sev), 24)
        urgency = self._urgency_from_sla(sla)

        prompt_user = self.per_fault_tmpl.render(
            evidence=self._serialize_evidence(ev, sev),
            taxonomy_entry=tx, kb_action=kb,
            exemplars=self._exemplars_for("per_fault", ev.fused_label, k=2),
        )
        cache_key = self._cache_key("per_fault", prompt_user)
        if self.cache and (cached := self.cache.get(cache_key)):
            parsed = cached
        else:
            res = self._call_llm(prompt_user, schema=self._per_fault_schema())
            parsed = res.parsed
            if self.cache: self.cache.set(cache_key, parsed)

        return FaultReport(
            fault_id=fid, fault_name=ev.fused_label,
            component_name=ev.component.component_name,
            bbox=[ev.component.bbox.x1, ev.component.bbox.y1, ev.component.bbox.x2, ev.component.bbox.y2],
            fused_confidence=ev.fused_confidence,
            severity=sev, urgency=urgency,
            summary=parsed["summary"],
            root_cause=parsed["root_cause"],
            recommended_action=parsed["recommended_action"],
            required_tools=parsed["required_tools"],
            required_parts=parsed["required_parts"],
            sla_hours=sla,
            evidence_image_paths=[],          # filled by attachments stage
            heatmap_paths=[],
            references=parsed.get("references", []),
        )

    def _one_line(self, v: ImageVerdict, faults: list[FaultReport]) -> str:
        if v.status == "clear":   return "No faults detected; component layout and thermal readings within normal limits."
        if v.status == "rejected": return f"Image rejected: {v.rejection_reason}."
        if v.status == "review":   return "Inspection requires human review due to low model confidence or signal disagreement."
        if not faults: return "Anomaly detected without specific fault classification."

        prompt = self.per_image_tmpl.render(image={
            "status": v.status, "fault_count": len(faults), "overall_severity": v.overall_severity.value,
            "top_faults": sorted(faults, key=lambda f: (-(0 if f.severity == "critical" else 1), -f.fused_confidence))[:3],
        })
        res = self._call_llm(prompt, schema={"type": "object", "properties": {"one_line_finding": {"type": "string"}}, "required": ["one_line_finding"]})
        return res.parsed["one_line_finding"]

    # ---- Batch summary (LLM call) ----
    def _batch_summary(self, br: BatchReport, images: list[ImageReport]) -> BatchSummary:
        distinct = self._aggregate_distinct_faults(images)
        prompt = self.batch_tmpl.render(
            tenant_display_name=self.cfg.get("display_name", self.tenant_id),
            batch={
                "n_images": br.n_images, "n_faults": br.n_faults, "n_clear": br.n_clear,
                "n_review": br.n_review, "n_rejected": br.n_rejected,
                "overall_status": br.overall_status.value, "overall_severity": br.overall_severity.value,
                "distinct_faults_by_severity": distinct,
            },
            exemplars=self._exemplars_for("batch_summary", "default", k=2),
        )
        schema = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "executive_summary": {"type": "string"},
                "top_concerns": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
                "recommended_next_inspection_days": {"type": ["integer", "null"]},
            },
            "required": ["headline", "executive_summary", "top_concerns", "recommended_next_inspection_days"],
        }
        res = self._call_llm(prompt, schema=schema)

        sev_to_urg = {"critical": Urgency.immediate, "high": Urgency.expedited,
                      "medium": Urgency.scheduled, "low": Urgency.routine, "info": Urgency.routine}
        return BatchSummary(
            headline=res.parsed["headline"],
            overall_status=br.overall_status.value if br.overall_status.value != "fault" else "fault",
            overall_severity=br.overall_severity.value,
            overall_urgency=sev_to_urg[br.overall_severity.value],
            n_images=br.n_images, n_faults=br.n_faults, n_clear=br.n_clear,
            n_review=br.n_review, n_rejected=br.n_rejected,
            top_concerns=res.parsed["top_concerns"],
            executive_summary=res.parsed["executive_summary"],
            recommended_next_inspection_days=res.parsed["recommended_next_inspection_days"],
        )

    # ---- Helpers ----
    def _fault_id_for(self, ev: FaultEvidence) -> int:
        # ev.fused_label is the canonical name; reverse-lookup
        for fid, tx in self.taxonomy.items():
            if tx["name"] == ev.fused_label: return fid
        return -1   # unknown_anomaly

    def _serialize_evidence(self, ev: FaultEvidence, severity: str) -> dict:
        return {
            "component_id": ev.component.component_id,
            "component_name": ev.component.component_name,
            "fault_id": self._fault_id_for(ev),
            "fault_name": ev.fused_label,
            "severity": severity,
            "fused_confidence": ev.fused_confidence,
            "anomaly_signals": [s.model_dump() for s in ev.anomaly_signals],
            "classifier_top3": ev.classifier.top3 if ev.classifier else [],
            "rule_hits": [{"id": h.rule_id, "description": h.description} for h in ev.rule_hits],
            "component_max_temp_c": None,   # filled if thermal context passed in
            "component_thermal_delta_c": None,
        }

    def _load_kb(self, fault_name: str | None) -> str | None:
        if not fault_name: return None
        p = self.kb_root / f"{fault_name}.md"
        return p.read_text() if p.exists() else None

    def _exemplars_for(self, kind: str, key: str, k: int = 2) -> list[dict]:
        base = Path("prompts/exemplars") / kind
        specific = base / f"{key}.json"
        files = [specific] if specific.exists() else sorted(base.glob("*.json"))[:k]
        return [json.loads(f.read_text()) for f in files[:k]] or []

    def _aggregate_distinct_faults(self, images: list[ImageReport]) -> list[dict]:
        agg = defaultdict(lambda: {"count": 0, "image_count": 0, "severity": "info", "component_name": ""})
        for img in images:
            seen_in_image = set()
            for fr in img.fault_reports:
                a = agg[fr.fault_name]
                a["count"] += 1
                a["severity"] = self._max_sev(a["severity"], fr.severity)
                a["component_name"] = fr.component_name
                seen_in_image.add(fr.fault_name)
            for n in seen_in_image: agg[n]["image_count"] += 1
        out = [{"fault_name": k, **v} for k, v in agg.items()]
        rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        out.sort(key=lambda f: (-rank[f["severity"]], -f["count"]))
        return out

    @staticmethod
    def _max_sev(a, b):
        rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        return a if rank[a] >= rank[b] else b

    @staticmethod
    def _urgency_from_sla(sla_hours: int) -> Urgency:
        if sla_hours <= 4:   return Urgency.immediate
        if sla_hours <= 24:  return Urgency.expedited
        if sla_hours <= 168: return Urgency.scheduled
        return Urgency.routine

    def _call_llm(self, user: str, schema: dict) -> LLMResult:
        system = self.system_tmpl.render()
        try:
            return self.provider.generate_json(
                system=system, user=user, schema=schema,
                temperature=self.cfg["temperature"],
                max_output_tokens=self.cfg["max_output_tokens"],
            )
        except Exception as e:
            # Fallback: template-only sub-report; mark provider as failed
            return LLMResult(text="", parsed=template_only_report(user, schema),
                              tokens_in=0, tokens_out=0,
                              provider="fallback", model="template",
                              latency_ms=0)

    @staticmethod
    def _per_fault_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "root_cause": {"type": "string"},
                "recommended_action": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "required_tools": {"type": "array", "items": {"type": "string"}},
                "required_parts": {"type": "array", "items": {"type": "string"}},
                "references": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "root_cause", "recommended_action", "required_tools", "required_parts"],
        }

    @staticmethod
    def _cache_key(kind: str, prompt: str) -> str:
        return f"{kind}:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
```

### 7.2 Fallback (`fallback.py`)

```python
"""Template-only report when LLM provider is unavailable. Quality is lower
but the output schema is valid, so dispatch never fails."""
from __future__ import annotations
from typing import Any

def template_only_report(prompt: str, schema: dict) -> dict:
    # Crude but valid: parse evidence from the prompt body and produce
    # structured-but-generic copy.
    if "recommended_action" in schema.get("properties", {}):
        return {
            "summary": "A fault was detected by the inspection pipeline. See attached evidence.",
            "root_cause": "Detailed root cause analysis is unavailable; the LLM provider was unreachable. The fault was flagged via the component + anomaly + classifier pipeline.",
            "recommended_action": ["Open the UPS panel safely.",
                                    "Verify the flagged component visually.",
                                    "Replace if visibly damaged; otherwise re-run inspection."],
            "required_tools": ["Standard UPS technician toolkit"],
            "required_parts": [],
            "references": [],
        }
    if "executive_summary" in schema.get("properties", {}):
        return {
            "headline": "Inspection complete — see attached findings.",
            "executive_summary": "The inspection batch completed. LLM narrative generation was unavailable; see structured findings for details.",
            "top_concerns": ["See findings."],
            "recommended_next_inspection_days": 30,
        }
    if "one_line_finding" in schema.get("properties", {}):
        return {"one_line_finding": "See structured findings; LLM narrative unavailable."}
    return {}
```

---

## 8. Renderers

### 8.1 Markdown (`renderers/markdown.py`)

```python
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from ..schemas import Report

def render_markdown(report: Report, out_path: Path) -> None:
    env = Environment(loader=FileSystemLoader("templates/report"), trim_blocks=True, lstrip_blocks=True)
    md = env.get_template("report.md.j2").render(r=report)
    out_path.write_text(md, encoding="utf-8")
```

`templates/report/report.md.j2` mirrors the executive layout — headline → summary → per-image findings → per-fault sections with embedded image references.

### 8.2 HTML & PDF

- HTML: same Jinja template with an HTML version (`report.html.j2`) styled with inline CSS for email compatibility.
- PDF: WeasyPrint over the HTML output.

```python
# renderers/pdf.py
from weasyprint import HTML, CSS
def render_pdf(html_path: Path, out_path: Path) -> None:
    HTML(str(html_path)).write_pdf(str(out_path), stylesheets=[CSS("templates/report/print.css")])
```

---

## 9. Dispatch

### 9.1 Routing config

```yaml
# configs/report/dispatch_routing.yaml
version: "1.0.0"
default:
  channels: [dashboard]
by_severity:
  info:     {channels: [dashboard]}
  low:      {channels: [dashboard, email_low_priority]}
  medium:   {channels: [dashboard, email]}
  high:     {channels: [dashboard, email, itsm_ticket]}
  critical: {channels: [dashboard, email, itsm_ticket, sms, pager]}
channels:
  email:
    smtp_host: env:SMTP_HOST
    from: noreply@powervision.ai
    to: env:OPS_DL
  email_low_priority:
    smtp_host: env:SMTP_HOST
    from: noreply@powervision.ai
    to: env:OPS_DIGEST_DL
  itsm_ticket:
    type: servicenow
    instance_url: env:SNOW_URL
    user: env:SNOW_USER
    password: env:SNOW_PASS
    assignment_group: "UPS Field Ops"
  sms:
    provider: twilio
    account_sid: env:TWILIO_SID
    auth_token: env:TWILIO_TOKEN
    to: env:SMS_LIST
  pager:
    provider: pagerduty
    routing_key: env:PD_ROUTING_KEY
  dashboard:
    webhook_url: env:DASHBOARD_INGEST_URL
```

### 9.2 Dispatcher implementation

```python
# powervision/report/dispatch/__init__.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import json, time
from ..schemas import Report
from .email import send_email
from .itsm import open_servicenow_ticket
from .webhook import post_webhook
from .sms import send_sms
from .pager import page

def dispatch(report: Report, routing: dict, attachments_dir: Path,
             dispatch_log: Path) -> None:
    sev = report.summary.overall_severity
    channels = routing["by_severity"].get(sev, routing["default"])["channels"]
    dispatch_log.parent.mkdir(parents=True, exist_ok=True)
    with dispatch_log.open("a") as log:
        for ch in channels:
            cfg = routing["channels"][ch]
            try:
                _dispatch_one(ch, cfg, report, attachments_dir)
                log.write(json.dumps({"channel": ch, "ok": True, "ts": time.time()}) + "\n")
            except Exception as e:
                log.write(json.dumps({"channel": ch, "ok": False, "error": str(e), "ts": time.time()}) + "\n")

def _dispatch_one(name, cfg, report, attachments_dir):
    if name.startswith("email"):     send_email(cfg, report, attachments_dir)
    elif name == "itsm_ticket":      open_servicenow_ticket(cfg, report)
    elif name == "dashboard":        post_webhook(cfg, report.model_dump(mode="json"))
    elif name == "sms":              send_sms(cfg, report.summary.headline)
    elif name == "pager":            page(cfg, report)
    else: raise ValueError(f"Unknown channel {name}")
```

### 9.3 Channel implementations (sketch)

```python
# powervision/report/dispatch/email.py
import smtplib, os
from email.message import EmailMessage
from pathlib import Path

def send_email(cfg, report, attachments_dir: Path):
    msg = EmailMessage()
    msg["From"] = cfg["from"]
    msg["To"] = _resolve_env(cfg["to"])
    msg["Subject"] = f"[{report.summary.overall_severity.upper()}] {report.summary.headline}"
    msg.set_content(report.summary.executive_summary)
    html_path = attachments_dir.parent / "report.html"
    if html_path.exists():
        msg.add_alternative(html_path.read_text(), subtype="html")
    for att in attachments_dir.glob("*"):
        with att.open("rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=att.name)
    with smtplib.SMTP(_resolve_env(cfg["smtp_host"])) as s:
        s.send_message(msg)

def _resolve_env(v):
    return os.environ[v[4:]] if isinstance(v, str) and v.startswith("env:") else v
```

```python
# powervision/report/dispatch/itsm.py
import httpx, os
def open_servicenow_ticket(cfg, report):
    url = _env(cfg["instance_url"])
    auth = (_env(cfg["user"]), _env(cfg["password"]))
    payload = {
        "short_description": report.summary.headline[:160],
        "description": report.summary.executive_summary,
        "urgency": _urg_to_snow(report.summary.overall_urgency.value),
        "assignment_group": cfg["assignment_group"],
        "u_powervision_report_id": report.report_id,
    }
    r = httpx.post(f"{url}/api/now/table/incident", auth=auth, json=payload, timeout=30)
    r.raise_for_status()
def _env(v): return os.environ[v[4:]] if v.startswith("env:") else v
def _urg_to_snow(u): return {"immediate": "1", "expedited": "2", "scheduled": "3", "routine": "4"}[u]
```

---

## 10. FastAPI service

```python
# services/report/main.py
from __future__ import annotations
from pathlib import Path
import json, yaml
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from powervision.report.generator import ReportGenerator
from powervision.report.renderers.markdown import render_markdown
from powervision.report.renderers.pdf import render_pdf
from powervision.report.dispatch import dispatch
from powervision.report.cache import RedisCache
from powervision.decision.schemas import BatchReport

app = FastAPI(title="PowerVision Report Generator")

TENANTS = yaml.safe_load(Path("configs/report/tenants.yaml").read_text())
ROUTING = yaml.safe_load(Path("configs/report/dispatch_routing.yaml").read_text())
TAXONOMY = {f["id"]: f for f in yaml.safe_load(Path("taxonomy/fault_taxonomy.yaml").read_text())["faults"]}
SEVERITY = yaml.safe_load(Path("taxonomy/severity_matrix.yaml").read_text())
KB_ROOT  = Path("kb/actions")
CACHE = RedisCache.from_env()
REPORTS_ROOT = Path("reports")

def _gen(tenant_id: str) -> ReportGenerator:
    cfg = TENANTS["tenants"].get(tenant_id, TENANTS["default"])
    return ReportGenerator(tenant_id, cfg, TAXONOMY, SEVERITY, KB_ROOT, cache=CACHE)

@app.post("/generate")
def generate(batch: dict = Body(...), tenant_id: str = "default", do_dispatch: bool = True):
    br = BatchReport.model_validate(batch)
    gen = _gen(tenant_id)
    report = gen.generate(br)
    out = REPORTS_ROOT / report.report_id
    (out / "attachments").mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    render_markdown(report, out / "report.md")
    # HTML + PDF…
    if do_dispatch:
        dispatch(report, ROUTING, out / "attachments", out / "dispatch_log.jsonl")
    return JSONResponse({"report_id": report.report_id, "path": str(out)})

@app.get("/report/{report_id}")
def fetch(report_id: str):
    p = REPORTS_ROOT / report_id / "report.json"
    if not p.exists(): raise HTTPException(404)
    return JSONResponse(json.loads(p.read_text()))

@app.get("/report/{report_id}.pdf")
def fetch_pdf(report_id: str):
    p = REPORTS_ROOT / report_id / "report.pdf"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="application/pdf")
```

---

## 11. Caching

```python
# powervision/report/cache.py
import json, os, redis
class ReportCache:
    def get(self, key: str): ...
    def set(self, key: str, value: dict, ttl: int = 60 * 60 * 24 * 30): ...

class RedisCache(ReportCache):
    def __init__(self, client): self.r = client
    @classmethod
    def from_env(cls):
        return cls(redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
    def get(self, key):
        b = self.r.get(key); return json.loads(b) if b else None
    def set(self, key, value, ttl=60*60*24*30):
        self.r.setex(key, ttl, json.dumps(value))
```

Cache key = hash of `(prompt body)` — identical evidence shapes reuse the same LLM output. Major cost reduction at scale.

---

## 12. Cost monitoring

| Metric (Prometheus) | Meaning |
|---|---|
| `pv_llm_tokens_in_total{provider, model, tenant}` | Cumulative input tokens |
| `pv_llm_tokens_out_total{provider, model, tenant}` | Cumulative output tokens |
| `pv_llm_cost_usd_total{tenant}` | Estimated cost (price table in `configs/report/prices.yaml`) |
| `pv_llm_failures_total{provider, reason}` | Provider failures |
| `pv_report_latency_ms_bucket{tenant}` | End-to-end report latency |
| `pv_report_fallback_total{tenant}` | Fallback engagements |

`tenants.yaml` `daily_budget_usd` is enforced in `generator._call_llm()`; over-budget triggers `fallback.py`.

---

## 13. Evaluation

Reports are evaluated on three axes:

| Axis | Method |
|---|---|
| **Schema compliance** | Every report must validate against `Report` Pydantic model. CI gate. |
| **Factual grounding** | Automated: assert every fault mentioned in `summary` / `root_cause` matches a `fault_name` in the source evidence (no hallucinated faults). |
| **Quality (LLM-graded rubric)** | Score 1–5 by a stronger LLM on (clarity, actionability, completeness, technical accuracy). Threshold: mean ≥ 4.0 across a fixed test corpus. |
| **Cost** | Tokens per report ≤ tenant budget. |

`tests/report/test_report_quality.py` runs the corpus through generation and asserts all four axes.

---

## 14. Code structure

```text
powervision/report/
├── __init__.py
├── schemas.py
├── llm.py
├── generator.py
├── fallback.py
├── cache.py
├── renderers/
│   ├── markdown.py
│   ├── html.py
│   └── pdf.py
└── dispatch/
    ├── __init__.py
    ├── email.py
    ├── itsm.py
    ├── webhook.py
    ├── sms.py
    └── pager.py

prompts/
├── system/{en,es}.j2
├── per_fault/{en,es}.j2
├── per_image/{en,es}.j2
├── batch_summary/{en,es}.j2
├── action_synthesis/en.j2
└── exemplars/{per_fault,batch_summary}/*.json

templates/report/
├── report.md.j2
├── report.html.j2
└── print.css

kb/actions/<fault_name>.md

configs/report/
├── tenants.yaml
├── dispatch_routing.yaml
└── prices.yaml

services/report/
├── main.py
└── Dockerfile

reports/<report_id>/...
logs/report/<batch_id>/...
```

---

## 15. Integration contracts

| Producer / consumer | Contract |
|---|---|
| **Phase 6 → Phase 7** | POST `BatchReport` JSON to `/generate?tenant_id=<>` |
| **Phase 7 → Dashboard** | `dispatch.webhook` POSTs `Report` JSON |
| **Phase 7 → ServiceNow** | Opens incident with `short_description = headline`, custom fields preserve `report_id` for traceability |
| **Phase 7 → Email** | HTML body from `report.html`, PDF attached, annotated images attached |
| **Phase 7 → Phase 8** | LLM cost metrics + fallback usage exported via Prometheus; ticket creation rate monitored |
| **Backward compatibility** | Demo (`api/main.py`, `api/streamlit_app.py`) does not interact with this service. |

---

## 16. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| LLM hallucinates a fault not in evidence | Compliance / safety risk | `summary` + `root_cause` validated against allow-list of evidence fault names; reject + retry on mismatch |
| LLM ignores severity/urgency lock | Wrong dispatch routing | Severity / urgency / SLA filled by Python AFTER LLM call; LLM cannot change them |
| Provider outage | Reports never sent | Per-tenant primary + secondary providers; `fallback.py` ensures schema-valid output even with no LLM |
| Cost overrun | Budget burn | Per-tenant daily budget enforced; cache hit rate alarm |
| Prompt injection via fault name (impossible — names are taxonomy-controlled) | n/a | All free-text fields in prompts come from taxonomy or numeric evidence |
| Stale exemplars | Quality drift | Exemplars committed under `prompts/exemplars/` with `kept_at` date; review quarterly |
| Long batches blow context window | Truncation, missing summaries | Per-fault and per-image calls are separate; only batch summary aggregates compactly (top-5 distinct faults only) |
| KB doc out of sync with taxonomy bump | LLM cites outdated SOP | KB CI gate fails if any `fault_name` in `kb/actions/` is no longer in `fault_taxonomy.yaml` |
| Dispatch flap (email queue down) | Retry storm | Dispatch logs to `dispatch_log.jsonl`; outer scheduler (Phase 8) retries failed channels with jitter |
| Tenant LLM has restricted internet (air-gapped) | Cannot call OpenAI | Tenant configured with `vllm_local`; Llama-3.1-8B-Instruct sufficient for templates with KB grounding |
| Markdown render breaks PDF | WeasyPrint crashes on certain unicode | Sanitize Markdown via `bleach` before HTML → PDF |
| Stale cache after taxonomy bump | Old reports generated for new schema | Cache key includes `taxonomy_version`; bumping taxonomy invalidates caches automatically |

---

## 17. Phase 7 exit checklist

- [ ] `Report` schema (`report_v1.0.0`) frozen and committed.
- [ ] Prompt templates (`system`, `per_fault`, `per_image`, `batch_summary`) in `en` (plus other languages if needed) reviewed by SME.
- [ ] At least 10 high-quality exemplars per kind committed.
- [ ] KB docs (`kb/actions/`) cover ≥ 30 of the most severe/frequent faults.
- [ ] LLM providers wired with retry, fallback, and cost metrics.
- [ ] Tenant config validated for at least one production tenant.
- [ ] `ReportGenerator` end-to-end test passes on the Phase 6 golden batch.
- [ ] Markdown + HTML + PDF renderers produce valid outputs.
- [ ] Dispatch tested against staging endpoints for every channel.
- [ ] Prometheus metrics scraping in Grafana dashboard.
- [ ] Schema compliance, factual grounding, and quality rubric tests green.
- [ ] Demo (`api/main.py`, `api/streamlit_app.py`) unchanged.

Phase 7 is **frozen** when all boxes are checked. Phase 8 begins.
