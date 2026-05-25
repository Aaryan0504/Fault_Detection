# Phase 1 — Fault Taxonomy and Dataset Architecture

> **Scope:** Backend + data only. This phase defines **what** the production system can ever detect, **how** every downstream model (component detector, anomaly models, fault classifier, rule engine) speaks the same language, and **where** every artifact lives. Nothing below is optional — Phases 2–8 all consume the contracts defined here.

---

## 1. Phase objective

The demo project (`PowerVision AI`) supports **9** axis-aligned fault classes on **one** UPS model. The production target must support **100+ specific fault types** across **multiple UPS system types** (online double-conversion, line-interactive, modular rack, transformer-based industrial UPS, etc.) with multi-modal evidence (RGB + thermal), multi-task labels (component bbox + fault class + anomaly mask), and a feedback loop from human review.

This phase delivers:

| Deliverable | Artifact |
|---|---|
| Canonical fault ontology (≥ 100 fault types) | `taxonomy/fault_taxonomy.yaml` |
| Component ontology (for YOLOv11 detector) | `taxonomy/component_taxonomy.yaml` |
| UPS system type registry | `taxonomy/ups_system_types.yaml` |
| Severity matrix and routing rules | `taxonomy/severity_matrix.yaml` |
| Canonical raw + curated dataset directory contract | This document, §6 |
| Multi-task labeling schema (detection + classification + anomaly + thermal) | `taxonomy/label_schema.json` (JSON Schema) |
| Dataset versioning policy (DVC + S3/GCS remote) | `dvc.yaml` + this document, §8 |
| Train/val/test split policy for highly imbalanced industrial data | This document, §9 |
| Verification tooling | `scripts/taxonomy/verify_taxonomy.py`, `scripts/taxonomy/verify_dataset_v2.py` |

Why it matters: every model trained in Phases 3–5 must reference these IDs. If the taxonomy changes after labeling has begun, **all** downstream artifacts (label files, memory banks, class-weighted losses, LLM prompt templates, dispatch routing) must be re-versioned. Lock this phase before any large-scale labeling job.

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Source | Format |
|---|---|---|
| Existing demo classes (9) | `dataset.yaml` (project root) | YAML list (`names: 0..8`) |
| SME knowledge (field engineers, OEM service manuals) | Manual + scanned PDFs | Free-form, distilled into YAML |
| Existing raw RGB images | `data/raw/Images/{train,val,test}` | JPG/PNG |
| Existing YOLO labels | `data/raw/labels/{train,val,test}` | `class_id cx cy w h` normalized |

### 2.2 Outputs (this phase)

```text
taxonomy/
├── fault_taxonomy.yaml            # 100+ faults, hierarchical
├── component_taxonomy.yaml        # ~30 components for YOLOv11
├── ups_system_types.yaml          # registry of UPS models
├── severity_matrix.yaml           # severity × urgency mapping
├── label_schema.json              # JSON Schema for multi-task labels
├── id_maps/
│   ├── fault_id_to_name.json      # int → str
│   ├── fault_name_to_id.json      # str → int
│   ├── component_id_to_name.json
│   └── component_name_to_id.json
└── CHANGELOG.md                   # taxonomy version history

dataset_v2/                        # canonical curated root (DVC-tracked)
├── manifest.parquet               # per-image record (see §6.4)
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── images/
│   ├── rgb/
│   ├── thermal/
│   └── fused/
├── labels/
│   ├── components/                # YOLO bbox (Phase 3 input)
│   ├── faults/                    # YOLO bbox + fault class (Phase 5 source crops)
│   ├── anomaly_masks/             # PNG masks (Phase 4 eval)
│   └── classification/            # JSONL per image (multi-label)
└── README.md
```

### 2.3 Format contracts (locked)

- **YOLO label line** (preserved from demo): `class_id cx cy w h` with all four geometric fields normalized to `[0, 1]`. **Do not change.**
- **Class IDs are stable forever.** Append-only. New faults receive the next free integer ID; deprecated faults are marked `status: deprecated` in `fault_taxonomy.yaml` and **never** reused.
- **String names** use `snake_case`, ASCII only, no spaces, max 48 chars.
- **Severity strings**: `info | low | medium | high | critical`.
- **Modality tags**: `rgb | thermal | fused`.

---

## 3. Fault taxonomy design (≥ 100 faults)

### 3.1 Hierarchy

A fault is a 4-tuple:

```
(ups_system_type, subsystem, component_group, fault_mode)
```

This four-axis design is the only way to scale to 100+ faults without name collisions and without breaking existing IDs when a new UPS model is onboarded.

| Axis | Examples | Stored in |
|---|---|---|
| `ups_system_type` | `online_double_conversion`, `line_interactive`, `modular_rack`, `transformer_industrial`, `dc_ups` | `ups_system_types.yaml` |
| `subsystem` | `input_stage`, `rectifier`, `inverter`, `battery_bank`, `bypass_static_switch`, `output_stage`, `control_pcb`, `cooling`, `enclosure_grounding` | enum in `fault_taxonomy.yaml` |
| `component_group` | `cable`, `terminal`, `screw`, `capacitor`, `battery_cell`, `relay`, `contactor`, `pcb_trace`, `fan`, `heatsink`, `bus_bar` | enum |
| `fault_mode` | `phase_reversed`, `loose`, `mismatch`, `bulge`, `leakage`, `corrosion`, `overheat`, `discoloration`, `crack`, `missing`, `burn_mark`, `arcing_residue` | enum |

A specific fault name is the canonical join: `<subsystem>__<component>__<mode>`, prefixed at runtime by `ups_system_type` when needed.

### 3.2 Master `fault_taxonomy.yaml` schema

```yaml
# taxonomy/fault_taxonomy.yaml
version: "2.0.0"
generated_at: "2026-05-24"
notes: |
  Append-only. Never reuse a retired ID. Class IDs 0-8 are inherited from the
  demo project (PowerVision AI v1) and remain pinned for backward compatibility
  with runs/phase_b/weights/best.pt.

# IDs 0-8 are LOCKED — they match the demo dataset.yaml exactly.
faults:
  - id: 0
    name: input_cable_fault
    subsystem: input_stage
    component_group: cable
    fault_mode: phase_reversed
    severity: high
    detectable_by: [yolo_fault, rule_engine]
    modality: [rgb]
    description: "Input cable phase order incorrect (e.g. L1/L2/L3 swapped at terminal)."
    visual_cues: ["cable color order does not match terminal label"]
    legacy_demo_id: 0
    status: active

  - id: 1
    name: loose_connection
    subsystem: input_stage
    component_group: terminal
    fault_mode: loose
    severity: high
    detectable_by: [yolo_fault, patchcore, rule_engine]
    modality: [rgb, thermal]
    description: "Wire not fully seated; visible gap or strands escaping clamp."
    visual_cues: ["protruding strands", "asymmetric clamp gap", "thermal hotspot >20°C above ambient"]
    legacy_demo_id: 1
    status: active

  # ... ids 2..8 mirror demo dataset.yaml ...

  # ===== NEW IDs (9+) — production-only =====
  - id: 9
    name: rectifier_capacitor_bulge
    subsystem: rectifier
    component_group: capacitor
    fault_mode: bulge
    severity: critical
    detectable_by: [yolo_fault, patchcore, efficientad, classifier]
    modality: [rgb]
    description: "Electrolytic capacitor top vent bulged outward; imminent failure."
    visual_cues: ["domed top", "vent rupture", "brown residue around base"]
    status: active
  - id: 10
    name: rectifier_capacitor_leakage
    subsystem: rectifier
    component_group: capacitor
    fault_mode: leakage
    severity: critical
    detectable_by: [yolo_fault, classifier]
    modality: [rgb]
    status: active
  - id: 11
    name: battery_terminal_corrosion
    subsystem: battery_bank
    component_group: terminal
    fault_mode: corrosion
    severity: high
    detectable_by: [yolo_fault, patchcore, classifier]
    modality: [rgb]
    status: active
  # ... continue through id: 119 ...
```

### 3.3 Canonical 120-row fault catalog (production target)

The catalog below is **the minimum viable production ontology**. Phase 1 is complete when this catalog (or an SME-reviewed superset) is fully populated in `fault_taxonomy.yaml`.

| ID range | Subsystem | Count | Representative faults |
|---|---|---|---|
| 0–8 | `input_stage` / `output_stage` / `control_pcb` | 9 | **Demo legacy — locked** (see existing `dataset.yaml`) |
| 9–24 | `rectifier` | 16 | capacitor_bulge, capacitor_leakage, igbt_burn, diode_burn, snubber_resistor_burn, dc_bus_arcing, pcb_track_burn, mov_burn, fuse_blown, fuse_missing, gate_driver_smoke, inductor_winding_burn, inductor_core_crack, thermal_paste_dryout, rectifier_overheating, rectifier_pcb_corrosion |
| 25–44 | `inverter` | 20 | igbt_module_crack, igbt_overheating, gate_resistor_burn, dc_link_cap_bulge, output_filter_inductor_burn, snubber_burn, busbar_discoloration, busbar_loose, busbar_arc_mark, pcb_solder_crack, pcb_track_corrosion, neutral_bar_loose, inverter_fan_missing, inverter_fan_blade_broken, heatsink_clogged, heatsink_warped, thermal_runaway_pattern, inverter_pcb_burn, gate_driver_pcb_burn, inverter_terminal_loose |
| 45–74 | `battery_bank` | 30 | battery_terminal_corrosion, battery_terminal_loose, battery_case_bulge, battery_case_crack, battery_electrolyte_leak, battery_strap_missing, battery_strap_corroded, battery_strap_loose, battery_polarity_reversed, battery_jumper_missing, battery_jumper_wrong_gauge, battery_label_missing, battery_label_unreadable, battery_vent_blocked, battery_terminal_grease_missing, battery_terminal_grease_excess, battery_string_imbalance_visual, battery_thermal_runaway_mark, battery_burn_mark, battery_cell_swollen, battery_tray_corrosion, battery_tray_loose_bolt, battery_disconnect_loose, battery_fuse_blown, battery_fuse_missing, battery_charger_cable_loose, battery_charger_cable_wrong_color, battery_temperature_sensor_missing, battery_temperature_sensor_dangling, battery_overheating |
| 75–84 | `bypass_static_switch` | 10 | scr_burn, scr_missing, bypass_relay_pitted, bypass_relay_stuck, bypass_terminal_loose, bypass_terminal_arc_mark, bypass_pcb_burn, bypass_fuse_blown, bypass_indicator_led_dead, bypass_overheating |
| 85–99 | `control_pcb` / `signal` | 15 | display_dead, display_cracked, control_pcb_burn, control_pcb_corrosion, control_pcb_loose_connector, ribbon_cable_torn, ribbon_cable_misseated, fiber_cable_bent, fiber_cable_disconnected, comms_port_damaged, eeprom_chip_missing, surface_mount_component_missing, surface_mount_solder_bridge, battery_temp_sensor_unplugged, fan_tach_unplugged |
| 100–109 | `cooling` | 10 | fan_blade_broken, fan_blade_missing, fan_grille_clogged, fan_cable_unplugged, fan_cable_wrong_polarity, fan_dust_buildup_severe, heatsink_fin_bent, heatsink_thermal_paste_dryout, intake_filter_missing, exhaust_obstruction |
| 110–119 | `enclosure_grounding` | 10 | ground_bar_loose, ground_cable_missing, ground_cable_corroded, ground_strap_missing, enclosure_door_open, enclosure_panel_missing, enclosure_corrosion, gland_plate_missing, gland_seal_torn, ip_rating_breach |

Total: **120** distinct fault types (ids `0–119`). New UPS models extend further via reserved blocks `120–199`, `200–299`, etc.

### 3.4 UPS system type registry

```yaml
# taxonomy/ups_system_types.yaml
version: "1.0.0"
ups_types:
  - id: ups_type_001
    name: online_double_conversion_10kva
    vendor: generic
    topology: online_double_conversion
    rated_kva: 10
    battery_topology: external_string_36s
    image_capture_views: [front_open, side_left, side_right, rear, battery_tray_top]
    applicable_fault_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "..."]
  - id: ups_type_002
    name: modular_rack_20kva
    topology: modular_rack
    rated_kva: 20
    image_capture_views: [front_module, rear_busbar, battery_shelf_front]
    applicable_fault_ids: [9, 10, 25, 26, "..."]
  - id: ups_type_003
    name: transformer_industrial_60kva
    topology: transformer_based
    rated_kva: 60
    image_capture_views: [front_open, transformer_top, output_busbar, control_pcb]
    applicable_fault_ids: [0, 1, 2, "...", 119]
```

A fault is applicable to an inspection only if its `id` ∈ `ups_types[*].applicable_fault_ids` for the device under test. Phase 6's rule engine uses this whitelist to suppress false positives from the fault classifier.

### 3.5 Severity matrix

```yaml
# taxonomy/severity_matrix.yaml
version: "1.0.0"
severity_levels:
  - level: info
    sla_hours: 720
    routing: [dashboard]
  - level: low
    sla_hours: 168
    routing: [dashboard, email_low_priority]
  - level: medium
    sla_hours: 48
    routing: [dashboard, email]
  - level: high
    sla_hours: 8
    routing: [dashboard, email, itsm_ticket]
  - level: critical
    sla_hours: 1
    routing: [dashboard, email, itsm_ticket, sms, pager]
overrides:
  # Specific fault IDs that escalate beyond their default severity
  battery_thermal_runaway_mark: critical
  enclosure_door_open:
    severity: medium
    note: "Safety hazard but rarely an electrical failure."
```

Phase 7 (LLM report dispatcher) consumes this verbatim; do not duplicate the routing logic anywhere else.

---

## 4. Component taxonomy (for YOLOv11 component detector — Phase 3)

The component detector is **a different model with a different class list** from the fault classifier. It localizes generic UPS components so anomaly models (PatchCore, EfficientAD) and the fault classifier can operate on tight ROI crops.

```yaml
# taxonomy/component_taxonomy.yaml
version: "1.0.0"
components:
  - {id: 0, name: input_terminal_block,    aliases: [input_tb]}
  - {id: 1, name: output_terminal_block,   aliases: [output_tb]}
  - {id: 2, name: battery_terminal_post}
  - {id: 3, name: battery_cell_case}
  - {id: 4, name: battery_strap}
  - {id: 5, name: cable_run_input}
  - {id: 6, name: cable_run_output}
  - {id: 7, name: cable_run_battery}
  - {id: 8, name: cable_run_signal}
  - {id: 9, name: ferrule_crimp}
  - {id: 10, name: screw_terminal}
  - {id: 11, name: bus_bar_segment}
  - {id: 12, name: electrolytic_capacitor}
  - {id: 13, name: film_capacitor}
  - {id: 14, name: igbt_module}
  - {id: 15, name: rectifier_diode_pack}
  - {id: 16, name: scr_thyristor}
  - {id: 17, name: relay_contactor}
  - {id: 18, name: fuse_holder}
  - {id: 19, name: mov_block}
  - {id: 20, name: inductor_core}
  - {id: 21, name: transformer_core}
  - {id: 22, name: heatsink_fin_block}
  - {id: 23, name: cooling_fan}
  - {id: 24, name: control_pcb}
  - {id: 25, name: ribbon_cable_connector}
  - {id: 26, name: display_panel}
  - {id: 27, name: ground_bar}
  - {id: 28, name: enclosure_panel}
  - {id: 29, name: cable_label_tag}
```

30 components → small enough for high mAP with the demo's existing two-phase YOLO recipe, large enough to cover all fault locations.

---

## 5. Multi-task labeling schema

Each curated image carries up to four parallel label artifacts. Phase 3 needs only the component bbox file; Phase 5 needs the fault bbox + crop classification; Phase 4 needs the anomaly mask (eval only); Phase 6 needs the structured JSON.

### 5.1 JSON Schema (`taxonomy/label_schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://powervision.ai/schemas/label_schema.json",
  "title": "PowerVision multi-task image label",
  "type": "object",
  "required": ["image_id", "ups_type_id", "modality", "components", "faults", "image_meta"],
  "properties": {
    "image_id": {"type": "string", "pattern": "^[a-z0-9_\\-]+$"},
    "ups_type_id": {"type": "string"},
    "modality": {"type": "string", "enum": ["rgb", "thermal", "fused"]},
    "capture_view": {"type": "string"},
    "device_id": {"type": "string"},
    "captured_at": {"type": "string", "format": "date-time"},
    "image_meta": {
      "type": "object",
      "required": ["width", "height", "ext"],
      "properties": {
        "width": {"type": "integer", "minimum": 32},
        "height": {"type": "integer", "minimum": 32},
        "ext": {"type": "string", "enum": ["jpg", "jpeg", "png", "tif", "tiff", "webp"]},
        "exif_iso": {"type": ["integer", "null"]},
        "exif_exposure_us": {"type": ["integer", "null"]},
        "thermal_min_c": {"type": ["number", "null"]},
        "thermal_max_c": {"type": ["number", "null"]}
      }
    },
    "components": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["component_id", "bbox_norm"],
        "properties": {
          "component_id": {"type": "integer", "minimum": 0},
          "bbox_norm": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 4, "maxItems": 4,
            "description": "[cx, cy, w, h] in normalized image coords"
          },
          "occlusion": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
          "truncation": {"type": "number", "minimum": 0, "maximum": 1, "default": 0}
        }
      }
    },
    "faults": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["fault_id", "bbox_norm"],
        "properties": {
          "fault_id": {"type": "integer", "minimum": 0},
          "bbox_norm": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 4, "maxItems": 4
          },
          "linked_component_idx": {
            "type": "integer",
            "description": "index into the `components` array of the parent component"
          },
          "anomaly_mask_path": {"type": ["string", "null"]},
          "annotator_confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "annotator_id": {"type": "string"},
          "annotation_session_id": {"type": "string"}
        }
      }
    },
    "image_level_labels": {
      "type": "object",
      "properties": {
        "has_any_fault": {"type": "boolean"},
        "is_blurry": {"type": "boolean"},
        "is_misframed": {"type": "boolean"},
        "is_low_light": {"type": "boolean"},
        "split_override": {"type": ["string", "null"], "enum": [null, "train", "val", "test"]}
      }
    },
    "provenance": {
      "type": "object",
      "required": ["source", "version"],
      "properties": {
        "source": {"type": "string", "enum": ["field_capture", "synthetic", "legacy_demo", "vendor_supplied"]},
        "version": {"type": "string"},
        "labeling_tool": {"type": "string", "enum": ["cvat", "labelimg", "roboflow", "scripted"]}
      }
    }
  }
}
```

### 5.2 On-disk representation per image

Given a curated image `dataset_v2/images/rgb/train/IMG_000123.jpg`:

| Companion file | Path | Format |
|---|---|---|
| Component bboxes (YOLO) | `dataset_v2/labels/components/train/IMG_000123.txt` | `class_id cx cy w h` (uses `component_taxonomy` IDs) |
| Fault bboxes (YOLO) | `dataset_v2/labels/faults/train/IMG_000123.txt` | `class_id cx cy w h` (uses `fault_taxonomy` IDs) |
| Anomaly mask | `dataset_v2/labels/anomaly_masks/train/IMG_000123.png` | 8-bit PNG, 0 = normal, 255 = anomaly |
| Structured JSON | `dataset_v2/labels/classification/train/IMG_000123.json` | Conforms to `label_schema.json` |

Empty `.txt` label files = background sample (preserved demo convention; see existing `bootstrap_raw_labels.py`).

---

## 6. Canonical dataset directory contract

### 6.1 Top-level layout

```text
dataset_v2/
├── manifest.parquet
├── splits/
│   ├── train.txt          # one image_id per line
│   ├── val.txt
│   └── test.txt
├── images/
│   ├── rgb/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── thermal/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── fused/
│       ├── train/
│       ├── val/
│       └── test/
├── labels/
│   ├── components/{train,val,test}/      # YOLO txt
│   ├── faults/{train,val,test}/          # YOLO txt
│   ├── anomaly_masks/{train,val,test}/   # PNG
│   └── classification/{train,val,test}/  # JSON
├── crops/                                # Phase 5 fault classifier source
│   ├── train/<fault_name>/<image_id>__<idx>.jpg
│   ├── val/<fault_name>/<image_id>__<idx>.jpg
│   └── test/<fault_name>/<image_id>__<idx>.jpg
├── normals_only/                         # Phase 4 anomaly training set
│   ├── <component_name>/train/<image_id>.jpg
│   └── <component_name>/val/<image_id>.jpg
├── dataset_v2.yaml                       # YOLO-style root descriptor (component detector)
├── dataset_v2_faults.yaml                # YOLO-style root descriptor (fault detector)
└── README.md
```

### 6.2 `dataset_v2.yaml` (component detector — Phase 3 input)

Preserves the Ultralytics convention from the existing demo `dataset.yaml`:

```yaml
# dataset_v2/dataset_v2.yaml
path: dataset_v2
train: images/rgb/train
val: images/rgb/val
test: images/rgb/test
task: detect
nc: 30
names:
  0: input_terminal_block
  1: output_terminal_block
  2: battery_terminal_post
  # ... full list from component_taxonomy.yaml ...
  29: cable_label_tag
```

> **Important:** the component detector resolves labels from `dataset_v2/labels/components/<split>/<stem>.txt` (Ultralytics' `images/...` → `labels/...` path-swap convention is preserved).

### 6.3 `dataset_v2_faults.yaml` (fault localization model used to bootstrap Phase 5 crops)

```yaml
path: dataset_v2
train: images/rgb/train
val: images/rgb/val
test: images/rgb/test
task: detect
nc: 120
names:
  0: input_cable_fault
  1: loose_connection
  # ... full list from fault_taxonomy.yaml, ids 0..119 ...
  119: ip_rating_breach
```

Labels for this model live at `dataset_v2/labels/faults/<split>/<stem>.txt`. To pick this label set Ultralytics needs a symlink (handled by `scripts/taxonomy/link_label_set.py`, see §10) since Ultralytics uses a fixed `images`→`labels` swap.

### 6.4 `manifest.parquet` (single source of truth)

Every image, every modality, every annotation is registered here. All split files, class-balance reports, and DVC pipelines read from this parquet — **never** glob the filesystem directly.

| Column | Type | Description |
|---|---|---|
| `image_id` | string | Stable, globally unique |
| `rgb_path` | string | Relative to `dataset_v2/` |
| `thermal_path` | string \| null | |
| `fused_path` | string \| null | |
| `ups_type_id` | string | FK → `ups_system_types.yaml` |
| `capture_view` | string | |
| `device_id` | string | |
| `captured_at` | timestamp | UTC ISO 8601 |
| `width` | int | RGB image |
| `height` | int | RGB image |
| `split` | string | `train` / `val` / `test` |
| `is_normal` | bool | True if `faults` array is empty |
| `fault_ids` | list\<int\> | All fault IDs present in image |
| `component_ids` | list\<int\> | All component IDs present |
| `quality_flags` | list\<string\> | `[blurry, misframed, low_light, overexposed]` |
| `provenance_source` | string | |
| `labeling_session_id` | string | |
| `label_schema_version` | string | |
| `taxonomy_version` | string | |
| `dvc_hash` | string | git-style content hash |

Generation script: `scripts/taxonomy/build_manifest.py` (see §10).

---

## 7. Labeling tooling and workflow

### 7.1 Tool selection

| Stage | Tool | Reason |
|---|---|---|
| Quick bbox-only bootstrap (legacy demo workflow) | **LabelImg** | Already used in `data/raw/`; preserve compatibility |
| Production multi-task (bbox + masks + image attributes + multi-annotator review) | **CVAT** (self-hosted) | Free, supports projects + tasks + tracks + segmentation + attribute schemas; exports YOLO + COCO + Datumaro |
| Optional fully-managed | **Roboflow** | Convenient but breaks air-gapped industrial deployments |

The CVAT project is configured to match `label_schema.json`:

```yaml
# taxonomy/cvat_project_spec.yaml
project_name: powervision_v2
labels:
  - name: input_terminal_block
    type: rectangle
    attributes:
      - name: occlusion
        input_type: number
        values: ["0", "0.25", "0.5", "0.75", "1"]
  # ... 30 components ...
  - name: rectifier_capacitor_bulge
    type: rectangle
    attributes:
      - name: annotator_confidence
        input_type: number
        values: ["0.5", "0.7", "0.9", "1.0"]
      - name: linked_component_idx
        input_type: number
  - name: anomaly_polygon
    type: polygon
    attributes:
      - name: fault_id
        input_type: number
image_level_attributes:
  - name: is_blurry
    input_type: checkbox
  - name: is_misframed
    input_type: checkbox
  - name: is_low_light
    input_type: checkbox
  - name: ups_type_id
    input_type: select
    values: ["ups_type_001", "ups_type_002", "ups_type_003"]
```

CVAT export → run `scripts/taxonomy/cvat_to_canonical.py` → produces YOLO `.txt`, anomaly PNG, and per-image JSON under `dataset_v2/`.

### 7.2 Quality assurance — double-blind annotation

| Phase | What | Pass criterion |
|---|---|---|
| 1st pass | Annotator A labels image | — |
| 2nd pass | Annotator B labels independently | — |
| Reconcile | Auto-compute IoU per bbox; if IoU ≥ 0.7 **and** same class → accept; else reviewer (senior engineer) resolves | ≥ 95% agreement on rare classes; ≥ 90% overall |
| Adjudicated | Final label written to `dataset_v2/` | — |

Reconciliation script: `scripts/taxonomy/reconcile_annotations.py`. Outputs `dataset_v2/qa/disagreement_report.json`.

### 7.3 Rare-class targeted labeling

When `fault_taxonomy.yaml` adds a new ID (e.g. id `42` `bypass_relay_pitted`), the labeling tool is filtered to surface only candidate images:

```bash
python scripts/taxonomy/build_labeling_queue.py \
  --fault-id 42 \
  --candidate-pool data/raw/Images/unlabeled \
  --hint-component scr_thyristor \
  --max-images 500
```

This script uses an existing component detector checkpoint (or, on the very first iteration, CLIP zero-shot) to pre-filter images to only those likely to contain the parent component, dramatically reducing labeling cost.

---

## 8. Dataset versioning and storage

### 8.1 DVC + cloud remote (S3 / GCS / Azure)

```text
.dvc/
├── config                # remote settings (no secrets — those go in env)
└── cache/                # local content-addressed store
dvc.yaml                  # pipeline stages (defined per phase)
dvc.lock
.dvcignore
```

`.dvc/config` (committed; credentials via env vars or `dvc remote modify --local`):

```ini
[core]
    remote = primary
['remote "primary"']
    url = s3://powervision-datasets/v2
    region = ap-south-1
['remote "warm_archive"']
    url = gs://powervision-archive/v2
```

### 8.2 Tracked artifacts

| Artifact | DVC-tracked | Git-tracked |
|---|---|---|
| `dataset_v2/images/**/*` | ✅ | ❌ |
| `dataset_v2/labels/**/*` | ✅ | ❌ |
| `dataset_v2/manifest.parquet` | ✅ | ❌ |
| `dataset_v2/dataset_v2*.yaml` | ❌ | ✅ |
| `taxonomy/*.yaml`, `taxonomy/*.json` | ❌ | ✅ |
| `taxonomy/CHANGELOG.md` | ❌ | ✅ |

### 8.3 Version bump rules

| Change | Bump |
|---|---|
| Add new fault ID, no schema change | **patch** (`2.0.0` → `2.0.1`) |
| Add new component ID | **patch** |
| Add new image-level attribute | **minor** (`2.0.x` → `2.1.0`) |
| Change `label_schema.json` shape | **major** (`2.x.y` → `3.0.0`); triggers **mandatory** re-export of CVAT and re-validation of all label files |
| Add new UPS system type | **minor** |
| Deprecate (never remove) a fault ID | **patch** (set `status: deprecated`) |

Each bump writes an entry to `taxonomy/CHANGELOG.md` and a new git tag `taxonomy-vX.Y.Z`. Models pin to a taxonomy version in their training config (Phase 3+).

### 8.4 Drive/Colab compatibility (preserved from demo)

The demo workflow mounts Google Drive in Colab and points `dataset.yaml`'s `path:` at the mounted root. The new pipeline preserves this:

```bash
# In Colab
from google.colab import drive
drive.mount('/content/drive')
dvc pull -r primary dataset_v2/   # OR rsync from /content/drive/MyDrive/powervision/dataset_v2
```

---

## 9. Train / val / test split policy

### 9.1 Constraints

1. **Device-disjoint:** images from the same physical UPS unit (`device_id`) must not appear in two splits. Prevents leakage from near-duplicate captures.
2. **Stratified by fault ID** *and* `ups_type_id`: every fault present in `train` must (where possible) also appear in `val`.
3. **Time-aware for test:** the most recent 15% of `captured_at` is reserved for `test` to simulate concept drift.
4. **Rare-class rule:** any fault ID with < 30 images in the entire corpus puts **2 images in val, 2 in test, remainder in train**; if fewer than 6 images exist, it is excluded from `test` (logged in `splits/excluded.txt`) and Phase 5 falls back to anomaly detection (Phase 4) for that fault.

### 9.2 Target ratios

| Split | Target | Hard floor per class |
|---|---|---|
| train | 70% | n ≥ 5 (or all if < 5) |
| val | 15% | n ≥ 2 |
| test | 15% | n ≥ 2 (waived for ultra-rare) |

### 9.3 Splitter implementation

```python
# scripts/taxonomy/build_splits.py (sketch)
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def build_splits(manifest_path: str, taxonomy_version: str, seed: int = 42):
    df = pd.read_parquet(manifest_path)
    df = df.sort_values("captured_at")

    # 1. Carve off time-aware test
    n_test_target = int(0.15 * len(df))
    test_pool = df.iloc[-n_test_target:].copy()
    rest = df.iloc[:-n_test_target].copy()

    # 2. Device-disjoint split for train/val on `rest`
    gss = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=seed)  # 0.176 ≈ 15/85
    train_idx, val_idx = next(gss.split(rest, groups=rest["device_id"]))
    rest.iloc[train_idx, rest.columns.get_loc("split")] = "train"
    rest.iloc[val_idx,   rest.columns.get_loc("split")] = "val"
    test_pool["split"] = "test"

    out = pd.concat([rest, test_pool], ignore_index=True)

    # 3. Promote rare-class images to val/test as needed
    out = _enforce_rare_class_floors(out, min_val=2, min_test=2)

    # 4. Sanity checks
    _assert_device_disjoint(out)
    _assert_no_classes_missing_from_train(out)

    out.to_parquet(manifest_path, index=False)
    for split in ("train", "val", "test"):
        (out.query("split == @split")["image_id"]
            .to_csv(f"dataset_v2/splits/{split}.txt", index=False, header=False))
```

### 9.4 Example distribution after split (target)

| Fault ID | Name | Train | Val | Test |
|---|---|---:|---:|---:|
| 9  | rectifier_capacitor_bulge | 280 | 60 | 60 |
| 11 | battery_terminal_corrosion | 410 | 90 | 90 |
| 45 | battery_terminal_loose | 320 | 70 | 70 |
| 100 | fan_blade_broken | 25 | 3 | 2 |
| 119 | ip_rating_breach | 6 | 2 | 0 (excluded → Phase 4 fallback) |

---

## 10. Code structure for Phase 1

```text
scripts/taxonomy/
├── verify_taxonomy.py             # YAML/JSON Schema validation of fault + component taxonomies
├── build_id_maps.py               # Regenerates taxonomy/id_maps/*.json
├── verify_dataset_v2.py           # Replacement for legacy verify_dataset.py
├── build_manifest.py              # Walks dataset_v2/ + reads CVAT exports → manifest.parquet
├── build_splits.py                # See §9.3
├── enforce_rare_class_floors.py
├── cvat_to_canonical.py           # CVAT export → dataset_v2/{images,labels,classification}
├── link_label_set.py              # Symlinks dataset_v2/labels/<set>/ ↔ dataset_v2/labels/
├── build_labeling_queue.py        # See §7.3
└── reconcile_annotations.py       # Double-blind QA, §7.2
```

### 10.1 `scripts/taxonomy/verify_taxonomy.py` (essential checks)

```python
"""Validate taxonomy/ YAMLs against invariants. Exit non-zero on failure."""
from pathlib import Path
import yaml, json, sys
from jsonschema import Draft202012Validator

TAX = Path("taxonomy")

def main():
    faults = yaml.safe_load((TAX / "fault_taxonomy.yaml").read_text())["faults"]
    comps  = yaml.safe_load((TAX / "component_taxonomy.yaml").read_text())["components"]
    ups    = yaml.safe_load((TAX / "ups_system_types.yaml").read_text())["ups_types"]
    sev    = yaml.safe_load((TAX / "severity_matrix.yaml").read_text())

    errors: list[str] = []

    # 1. Unique IDs, contiguous block, no gaps within active range
    ids = [f["id"] for f in faults]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate fault IDs.")
    if sorted(ids) != ids:
        errors.append("Fault IDs are not in ascending order.")

    # 2. Demo legacy IDs (0..8) preserved
    legacy = {
        0: "input_cable_fault", 1: "loose_connection", 2: "output_cable_fault",
        3: "ri_cable_mismatch", 4: "screw_faults", 5: "signal_cable_mismatch",
        6: "J14_cable_mismatch", 7: "red_white_mismatch", 8: "ferrule_mismatch",
    }
    for fid, fname in legacy.items():
        if not any(f["id"] == fid and f["name"] == fname for f in faults):
            errors.append(f"Legacy demo fault id={fid} ({fname}) drifted.")

    # 3. Severity values valid
    valid_sev = {s["level"] for s in sev["severity_levels"]}
    for f in faults:
        if f["severity"] not in valid_sev:
            errors.append(f"Fault {f['id']} has unknown severity {f['severity']}.")

    # 4. UPS-type fault whitelists reference real ids
    fault_ids_set = set(ids)
    for u in ups:
        bad = set(u["applicable_fault_ids"]) - fault_ids_set
        if bad:
            errors.append(f"UPS type {u['id']} references unknown fault ids {bad}.")

    # 5. Component IDs unique
    cids = [c["id"] for c in comps]
    if len(cids) != len(set(cids)):
        errors.append("Duplicate component IDs.")

    # 6. JSON Schema for label_schema.json is itself valid
    schema = json.loads((TAX / "label_schema.json").read_text())
    Draft202012Validator.check_schema(schema)

    if errors:
        print("FAIL — taxonomy invariants violated:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"PASS — {len(faults)} faults, {len(comps)} components, {len(ups)} UPS types.")

if __name__ == "__main__":
    main()
```

### 10.2 `scripts/taxonomy/build_manifest.py` (sketch)

```python
"""Walk dataset_v2/images/rgb/<split>/ + read companion labels + per-image JSON,
emit dataset_v2/manifest.parquet."""
from pathlib import Path
import json, pandas as pd, hashlib
from PIL import Image

ROOT = Path("dataset_v2")
SPLITS = ("train", "val", "test")

def _dvc_hash(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_yolo(p: Path) -> list[int]:
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [int(line.split()[0]) for line in p.read_text().splitlines() if line.strip()]

def main():
    rows = []
    for split in SPLITS:
        for img_path in sorted((ROOT / "images" / "rgb" / split).glob("*")):
            stem = img_path.stem
            cls_json = ROOT / "labels" / "classification" / split / f"{stem}.json"
            comp_txt = ROOT / "labels" / "components"     / split / f"{stem}.txt"
            flt_txt  = ROOT / "labels" / "faults"         / split / f"{stem}.txt"

            meta = json.loads(cls_json.read_text()) if cls_json.exists() else {}
            with Image.open(img_path) as im:
                w, h = im.size

            fault_ids = _read_yolo(flt_txt)
            comp_ids  = _read_yolo(comp_txt)

            rows.append(dict(
                image_id=stem,
                rgb_path=str(img_path.relative_to(ROOT)),
                thermal_path=meta.get("thermal_path"),
                fused_path=meta.get("fused_path"),
                ups_type_id=meta.get("ups_type_id", "unknown"),
                capture_view=meta.get("capture_view", "unknown"),
                device_id=meta.get("device_id", "unknown"),
                captured_at=meta.get("captured_at"),
                width=w, height=h,
                split=split,
                is_normal=(len(fault_ids) == 0),
                fault_ids=fault_ids,
                component_ids=comp_ids,
                quality_flags=meta.get("image_level_labels", {}).get("quality_flags", []),
                provenance_source=meta.get("provenance", {}).get("source", "unknown"),
                labeling_session_id=meta.get("provenance", {}).get("session_id", ""),
                label_schema_version=meta.get("provenance", {}).get("schema_version", ""),
                taxonomy_version=meta.get("provenance", {}).get("taxonomy_version", ""),
                dvc_hash=_dvc_hash(img_path),
            ))

    pd.DataFrame(rows).to_parquet(ROOT / "manifest.parquet", index=False)

if __name__ == "__main__":
    main()
```

### 10.3 `dvc.yaml` (Phase 1 stages only — extended in later phases)

```yaml
stages:
  taxonomy_verify:
    cmd: python scripts/taxonomy/verify_taxonomy.py
    deps:
      - taxonomy/fault_taxonomy.yaml
      - taxonomy/component_taxonomy.yaml
      - taxonomy/ups_system_types.yaml
      - taxonomy/severity_matrix.yaml
      - taxonomy/label_schema.json
      - scripts/taxonomy/verify_taxonomy.py
    always_changed: false

  id_maps:
    cmd: python scripts/taxonomy/build_id_maps.py
    deps:
      - taxonomy/fault_taxonomy.yaml
      - taxonomy/component_taxonomy.yaml
    outs:
      - taxonomy/id_maps/fault_id_to_name.json
      - taxonomy/id_maps/fault_name_to_id.json
      - taxonomy/id_maps/component_id_to_name.json
      - taxonomy/id_maps/component_name_to_id.json

  manifest:
    cmd: python scripts/taxonomy/build_manifest.py
    deps:
      - dataset_v2/images
      - dataset_v2/labels
      - scripts/taxonomy/build_manifest.py
    outs:
      - dataset_v2/manifest.parquet

  splits:
    cmd: python scripts/taxonomy/build_splits.py
    deps:
      - dataset_v2/manifest.parquet
      - scripts/taxonomy/build_splits.py
    outs:
      - dataset_v2/splits/train.txt
      - dataset_v2/splits/val.txt
      - dataset_v2/splits/test.txt

  verify_dataset:
    cmd: python scripts/taxonomy/verify_dataset_v2.py
    deps:
      - dataset_v2/manifest.parquet
      - dataset_v2/splits
      - taxonomy/label_schema.json
    metrics:
      - dataset_v2/qa/verification_report.json
```

---

## 11. Integration contracts (handed to later phases)

| Consumer phase | Contract surface | Provided by Phase 1 |
|---|---|---|
| **Phase 2** (preprocessing) | `manifest.parquet` columns (`rgb_path`, `thermal_path`, `captured_at`, `device_id`, `quality_flags`) | Built here |
| **Phase 3** (YOLO component) | `dataset_v2/dataset_v2.yaml`, `taxonomy/component_taxonomy.yaml`, `dataset_v2/labels/components/<split>/*.txt` | Built here |
| **Phase 4** (PatchCore/EfficientAD) | `dataset_v2/normals_only/<component_name>/{train,val}/*.jpg` populated from `is_normal == True` rows of manifest | Filesystem layout reserved here; population script lives in Phase 4 |
| **Phase 5** (fault classifier) | `dataset_v2/crops/<split>/<fault_name>/*.jpg` derived from `labels/faults/`; `taxonomy/fault_taxonomy.yaml` for class list | Layout reserved here |
| **Phase 6** (rule engine) | `taxonomy/ups_system_types.yaml` (`applicable_fault_ids`), `taxonomy/severity_matrix.yaml`, `taxonomy/fault_taxonomy.yaml` (`detectable_by`, `modality`) | Built here |
| **Phase 7** (LLM report) | `taxonomy/fault_taxonomy.yaml` (`description`, `visual_cues`), `taxonomy/severity_matrix.yaml` (`routing`) | Built here |
| **Phase 8** (HITL retraining) | `manifest.parquet` (`provenance_source`, `labeling_session_id`, `taxonomy_version`); CVAT project spec for re-labeling | Built here |

### 11.1 Backward compatibility with the demo

| Demo artifact | Behavior |
|---|---|
| `dataset.yaml` (root) | Kept untouched; remains the input for the demo two-phase YOLO pipeline. |
| Fault IDs 0–8 | Identical name + index in `fault_taxonomy.yaml`. `runs/phase_b/weights/best.pt` still infers without re-mapping. |
| YOLO label format | Unchanged. |
| `bootstrap_raw_labels.py` | Still works against `data/raw/`. New equivalent for `dataset_v2/` lives in `scripts/taxonomy/`. |
| Inference (`api/main.py`, `api/streamlit_app.py`) | Unchanged until Phase 3 ships a new component detector; even then, the legacy fault-detector weights remain a valid fallback (gated via env var `POWERVISION_USE_LEGACY_MODEL=1`). |

---

## 12. Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| Class ID drift after CVAT re-export | Phase 3 trains with shuffled class names | `verify_taxonomy.py` checks legacy IDs (§10.1); CI runs this on every PR touching `taxonomy/` |
| Annotator confuses two visually similar faults (e.g. `capacitor_bulge` vs `capacitor_leakage`) | High val mAP but low test mAP on confused classes | Double-blind QA (§7.2); confusion matrix on val every retraining cycle; SME monthly review of `disagreement_report.json` |
| Device leakage (same UPS in train + test) | Suspiciously high test mAP, drops in production | `GroupShuffleSplit` on `device_id`; `verify_dataset_v2.py` asserts disjointness |
| Rare-class catastrophe (< 5 examples globally) | Phase 5 cannot train; classifier returns no probability mass | Rare-class fallback to Phase 4 anomaly model (declared via `detectable_by` field); `build_labeling_queue.py` runs in HITL loop (Phase 8) |
| Thermal/RGB misalignment in `fused` modality | Anomaly heatmaps shifted from RGB ROI | Phase 2 owns calibration; Phase 1 ensures `thermal_path` is **optional**, never assumed |
| Taxonomy version mismatch between training data and inference | Inference returns class index that no longer exists | Every checkpoint stores `taxonomy_version` in its metadata (Phase 3+); Phase 6 confidence gate refuses serving if mismatch |
| DVC remote outage during Colab session | Training cannot fetch images | `dvc pull` failures fall back to `rsync` from a Drive mirror (warm archive); manifest hashes verify integrity |
| CVAT export drift (CVAT bumps internal field names) | `cvat_to_canonical.py` silently drops attributes | Schema validation against `label_schema.json` is mandatory at the end of `cvat_to_canonical.py`; failures exit non-zero |
| Ontology bloat (SMEs keep adding marginal faults) | 200+ faults, most with < 5 examples | Quarterly taxonomy review: faults with `< 10` examples after 6 months get `status: deprecated` and merged into a coarser parent fault |
| Mixed-tenancy data leakage (customer A's images visible to model trained for customer B) | Compliance breach | `provenance.source` and `device_id` carry tenant prefix (`tenantA_devid_001`); `build_splits.py` enforces tenant whitelist when present |

---

## 13. Phase 1 exit checklist

- [ ] `taxonomy/fault_taxonomy.yaml` has ≥ 100 entries, validated by `verify_taxonomy.py`.
- [ ] `taxonomy/component_taxonomy.yaml` has ≥ 25 entries.
- [ ] `taxonomy/ups_system_types.yaml` has ≥ 1 UPS type fully spec'd.
- [ ] `taxonomy/label_schema.json` validates against Draft 2020-12.
- [ ] `dataset_v2/` directory tree created (may be empty; populated in Phase 2).
- [ ] `dvc.yaml` Phase 1 stages run clean: `dvc repro taxonomy_verify id_maps`.
- [ ] CVAT project provisioned with labels matching `cvat_project_spec.yaml`.
- [ ] CI workflow `.github/workflows/taxonomy.yml` runs `verify_taxonomy.py` on every PR touching `taxonomy/**`.
- [ ] Git tag `taxonomy-v2.0.0` pushed.
- [ ] Legacy demo (`api/main.py`, `runs/phase_b/weights/best.pt`) still serves inference unchanged.

Phase 1 is **frozen** when all boxes are checked. Phase 2 begins.
