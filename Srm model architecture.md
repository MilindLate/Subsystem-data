# SRM Structural Defect Detection — Model Architecture & I/O Reference
**Companion to `ISHM_SRM_DefectDetection_Colab.ipynb`.**

---

## 1. Architecture overview

```
                    ┌───────────────────────────────┐
                    │   RAW TELEMETRY (10 channels)  │
                    │   pressure, bond stress, temp,  │
                    │   strain (axial+hoop), etc.     │
                    │   all at 2kHz (see SRM dict)     │
                    └───────────────┬─────────────────┘
                                   │
             ┌─────────────────────┴──────────────────────┐
             │                                             │
             ▼                                             ▼
┌─────────────────────────────┐          ┌────────────────────────────────┐
│ PATH A: DBST threshold       │          │ PATH B: DNN on embedded strain │
│ monitoring (cheap, flight-   │          │ data (1D CNN)                  │
│ proven, no training needed)  │          │                                 │
│                               │          │ 1. Window strain_axial_ue +    │
│ 1. Read bond_stress_mpa +     │          │    strain_hoop_ue (2-channel,  │
│    case_temp_c (whole burn)   │          │    10% window, 50% overlap)    │
│ 2. Compare against calibrated │          │ 2. Normalize (global mean/std) │
│    warning/critical bands     │          │ 3. 3-layer 1D CNN + GAP head   │
│ 3. Flag if either exceeds     │          │    -> softmax over 6 classes   │
│    band for a sustained       │          │ 4. Majority-vote windows ->    │
│    fraction of the burn       │          │    instance-level prediction   │
└───────────────┬───────────────┘          └────────────────┬────────────────┘
                │                                            │
          dbst_flag (0/1)                            cnn_flag (0/1) + fault_type
                │                                            │
                └──────────────────┬─────────────────────────┘
                                   ▼
                    hybrid_flag = dbst_flag OR cnn_flag
                                   │
                                   ▼
              OUTPUT: fault_type, hybrid_flag, dbst_severity
```

**Why DBST first, DNN layered on top (not the reverse):** DBST is cheap, explainable, and flight-proven with no training data required — it should be the always-on primary layer. The CNN adds coverage for fault modes DBST's two channels can't see (e.g. `ignition_fault`, `nozzle_erosion_blockage`) and finer discrimination of *which* strain-related fault occurred, not just that bond stress/temperature crossed a line.

---

## 2. Inputs — exact schema

### Path A (DBST) reads only:
| Column | Unit | Role |
|---|---|---|
| `bond_stress_mpa` | MPa | Compared against 8/12 MPa warning/critical bands |
| `case_temp_c` | °C | Compared against 180/230°C warning/critical bands |

### Path B (CNN) reads only:
| Column | Unit | Role |
|---|---|---|
| `strain_axial_ue` | µε | Channel 1 of the 2-channel CNN input |
| `strain_hoop_ue` | µε | Channel 2 of the 2-channel CNN input |

Both paths ignore the other 8 channels in the telemetry dictionary (`chamber_pressure_psi`, `nozzle_erosion_mm_s`, `igniter_continuity`, `accel_axial_g`, `regression_rate_mm_s`, `storage_temp_c`) entirely for **this** model — those remain available for future subsystem-level aggregation (Phase 6) but aren't consumed by either detector here.

---

## 3. Outputs — exact schema

| Output | Type | Produced by | Meaning |
|---|---|---|---|
| `dbst_flag` | 0/1 | Path A | 1 if bond stress or case temp sustained-exceeded the warning band |
| `dbst_severity` | string | Path A | `"normal"` / `"warning"` / `"critical"` |
| `cnn_pred` | string (one of 6 fault_type labels) | Path B | Per-window prediction; instance-level = majority vote |
| `cnn_flag` | 0/1 | Derived | 1 if `cnn_pred != "normal"` |
| `hybrid_flag` | 0/1 | Fusion | `dbst_flag OR cnn_flag` — the final decision |

6 possible `fault_type` values: `normal`, `case_bond_degradation`, `grain_crack`, `nozzle_erosion_blockage`, `ignition_fault`, `combustion_instability` — defined in `SRM_Telemetry_Dictionary.md`.

---

## 4. Which parameters live in which file

| File | What it defines | Edit when... |
|---|---|---|
| `SRM_Telemetry_Dictionary.md` | Authoritative parameter ranges, sensor types, fault definitions | Changing what a fault means physically |
| `generate_srm_data.py` (data-collection deliverable) | `CHANNELS`, `baseline_channel()`, `apply_fault()` | Regenerating training data / changing burn assumptions |
| **Notebook cell 1** | Same generator, self-contained copy for Colab | Keep in sync with the standalone `.py` manually |
| **Notebook cell 2** | `N_PER_CLASS` (now 100 by default — a real whole-dataset scale, not a demo), `BOND_STRESS_WARNING_MPA`/`CRITICAL_MPA`, `CASE_TEMP_WARNING_C`/`CRITICAL_C`, `sustained_fraction`, `WINDOW_FRAC`, `OVERLAP`, `STRAIN_COLS` — dataset generation, DBST, and strain windowing are now one streaming cell (see §7 memory note) | **DBST thresholds must be re-validated against real motor qualification data before flight use**; dataset-size/windowing changes also go here |
| **Notebook cell 5** | CNN architecture (`build_cnn`: filter counts, kernel sizes, dropout) | Tuning model capacity |
| **Notebook cell 6** | Training config (`epochs`, `batch_size`, `EarlyStopping` patience) | Tuning training regime |

---

## 5. Using this in real time

### Must change before real use
1. **DBST thresholds (cell 2)** are grounded in literature-survey ranges, not a specific certified motor's qualification limits — re-derive from your actual motor's test data before flight use.
2. **Retrain the CNN** on real or FEA-simulated strain data once available, replacing the synthetic generator's output — same schema, same windowing, no architecture change needed.
3. **Re-check the window length assumption** — `WINDOW_FRAC=0.10` was tuned against this notebook's 20-second synthetic burn; a different motor's burn duration changes the absolute window length in samples, which changes the CNN's effective receptive field. Re-tune if your real motor's burn time differs substantially from 20s.

### Real-time inference loop (conceptual)
```
continuously (pre-ignition):
    monitor bond_stress_mpa, case_temp_c -> dbst_flag  (this is the only
    check that matters before ignition, since strain-based CNN needs an
    active burn signal to be meaningful)

during burn, every WINDOW_STEP:
    1. pull latest strain_axial_ue + strain_hoop_ue window
    2. normalize with the SAME mean_/std_ saved in
       cnn_normalization_and_classes.json (do not recompute per-window)
    3. cnn.predict(window) -> fault_type for this window
    4. hybrid_flag = dbst_flag OR (cnn_pred != "normal")
    5. if hybrid_flag: raise alert with fault_type + dbst_severity to
       the FDIR Level 1/2 aggregation layer
```

---

## 6. Known limitations, stated plainly

- No real SRM data exists anywhere to validate against — this model's real-world accuracy is genuinely unknown until tested against actual instrumented static-fire or FEA-simulated data. Treat reported metrics as a pipeline sanity check.
- The CNN was trained and evaluated on the same synthetic generator used to create it — there's no independent real-world holdout, which is a materially weaker validation than the liquid propulsion model has (that one at least blends in two real datasets).
- DBST bands are literature-derived estimates, explicitly flagged in the saved `dbst_thresholds.json` as needing re-validation.
- Fusion is the simplest OR rule — same caveat as liquid propulsion, a confidence-weighted fusion is the natural upgrade once real data exists to calibrate it.

---

## 7. Memory-safety note (why generation/DBST/windowing are one cell)

An earlier version of this notebook generated all `N_PER_CLASS` × 6 instances into one list before processing — which meant holding every instance's full-resolution DataFrame in memory at once. At a real "whole data" scale (`N_PER_CLASS=100`, 600 instances), that peaked at **2.6GB** in testing, before the CNN's strain-window tensor was even built, and reliably crashed the kernel — the same category of mistake that crashed the liquid propulsion notebook's feature-extraction step, just triggered by a different mechanism (accumulating raw DataFrames vs. building an oversized long-format table). Cell 2 now processes one instance at a time — generate, run DBST, extract strain windows as `float32`, discard the full-resolution DataFrame — reducing peak memory to ~524MB for the same 600-instance run, verified by rerunning it before delivery. Only one example DataFrame per fault class is retained (`example_by_fault`), used solely by the final visualization cell.
