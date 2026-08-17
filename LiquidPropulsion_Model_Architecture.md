# Liquid Propulsion Fault Detection — Model Architecture & I/O Reference
**Companion to `ISHM_LiquidPropulsion_FaultDetection_Colab.ipynb`. Explains what goes in, what comes out, how it works internally, which parameters live in which file, and how to actually run this against real-time telemetry.**

---

## 1. Architecture overview

Two independent detectors, fused by a simple OR rule:

```
                         ┌─────────────────────────────┐
                         │   RAW TELEMETRY (13 chan)    │
                         │  10 medium-rate (1kHz) +     │
                         │  1 high-rate vibration(20kHz)│
                         └───────────────┬──────────────┘
                                         │
                     ┌───────────────────┴────────────────────┐
                     │                                        │
                     ▼                                        ▼
      ┌──────────────────────────────┐        ┌───────────────────────────────┐
      │  PATH A: Data-driven (ML)    │        │  PATH B: Physics-based        │
      │                              │        │                                │
      │  1. Window the 10 medium-rate│        │  1. Compute expected chamber   │
      │     channels (10% length,    │        │     pressure from commanded    │
      │     50% overlap)             │        │     throttle curve             │
      │  2. tsfresh feature          │        │  2. residual = |actual -       │
      │     extraction (Efficient    │        │     expected| / P_MAX          │
      │     FC parameters)           │        │  3. flag if residual exceeds   │
      │  3. Benjamini-Hochberg       │        │     threshold for a sustained  │
      │     significance selection   │        │     fraction of the window     │
      │  4. Hand-picked vibration    │        │                                │
      │     features (RMS, kurtosis, │        │  No training data needed --   │
      │     dominant freq, etc.)     │        │  first-principles check        │
      │  5. Random Forest classifier │        └────────────────┬───────────────┘
      │     -> per-window fault_type │                         │
      └───────────────┬───────────────┘                         │
                      │                                        │
                      ▼                                        ▼
              rf_flag (0/1)                          physics_flag (0/1)
                      │                                        │
                      └──────────────────┬─────────────────────┘
                                         ▼
                         hybrid_flag = rf_flag OR physics_flag
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │  OUTPUT: fault_type, confidence,        │
                    │  hybrid_flag, physics_residual value    │
                    └────────────────────────────────────────┘
```

**Why two independent paths, not one:** if the Random Forest misclassifies (e.g., a fault pattern it wasn't trained on), the physics check can still catch anything that visibly perturbs chamber pressure vs. the commanded throttle curve, since it doesn't depend on training data at all. This is the hybrid-AI requirement from your project's AI Design Principles — redundant detectors that can fail in different, independent ways, not two versions of the same idea.

---

## 2. Inputs — exact schema

### 2.1 Medium-rate channels (10 channels, 1kHz, fed to Path A's tsfresh pipeline)

| Column name | Unit | Used by |
|---|---|---|
| `chamber_pressure_psi` | psi | Path A (features) + **Path B (the only channel Path B reads)** |
| `injector_pressure_fuel_psi` | psi | Path A |
| `injector_pressure_ox_psi` | psi | Path A |
| `feedline_pressure_fuel_psi` | psi | Path A |
| `feedline_pressure_ox_psi` | psi | Path A |
| `flow_rate_fuel_kgs` | kg/s | Path A |
| `flow_rate_ox_kgs` | kg/s | Path A |
| `turbopump_shaft_rpm` | RPM | Path A |
| `wall_temp_c` | °C | Path A |
| `cryo_tank_temp_k` | K | Path A |
| `valve_position_fuel_pct` | % | Path A |
| `valve_position_ox_pct` | % | Path A |
| `plume_asymmetry_idx` | dimensionless | Path A |

Plus `t_s` (elapsed mission/burn time, seconds) — required for both windowing (Path A) and the throttle-curve lookup (Path B).

### 2.2 High-rate channel (1 channel, 20kHz, fed to Path A's hand-picked vibration features only)

| Column name | Unit | Used by |
|---|---|---|
| `turbopump_vibration_g` | g | Path A only (RMS, kurtosis, peak-to-peak, dominant frequency, dominant-frequency energy fraction — 5 derived features, not full tsfresh, for compute efficiency) |

### 2.3 What Path B (physics) does NOT use
Path B reads **only** `chamber_pressure_psi` and `t_s`. It ignores all 11 other channels entirely — this is intentional (see the notebook's section 7 scope note) and is why its standalone recall is narrow: it's built to catch combustion-instability-type faults specifically, not a general-purpose detector.

---

## 3. Outputs — exact schema

| Output | Type | Produced by | Meaning |
|---|---|---|---|
| `rf_pred` | string (one of the 7 fault_type labels) | Path A (Random Forest) | Per-window predicted class; instance-level is the majority vote across that instance's windows |
| `rf_flag` | 0/1 | Derived from `rf_pred` | 1 if `rf_pred != "normal"` |
| `physics_flag` | 0/1 | Path B | 1 if the sustained chamber-pressure residual exceeds threshold |
| `max_residual_frac` | float (0-1+) | Path B | The actual residual magnitude, for logging/trending even when below the flag threshold |
| `hybrid_flag` | 0/1 | Fusion | `rf_flag OR physics_flag` — the final decision this model should be judged on |

The 7 possible `fault_type` values: `normal`, `bearing_wear`, `combustion_instability`, `injector_blockage`, `feed_cavitation`, `progressive_degradation`, `valve_fault` — defined and grounded in `LiquidPropulsion_Telemetry_Dictionary.md`.

---

## 4. Which parameters live in which file — don't edit the wrong one

| File | What it defines | Edit this file when... |
|---|---|---|
| `LiquidPropulsion_Telemetry_Dictionary.md` | The authoritative parameter list, units, normal/warning/critical ranges, sensor types, and fault-class definitions | You're changing what a fault *means* physically, or adding/removing a monitored parameter for the vehicle |
| `generate_liquid_propulsion_data.py` (data-collection phase deliverable) | `CHANNELS` list, `baseline_channel()` per-channel design curves, `apply_fault()` per-fault-class signal perturbations, real-data hybridization hooks (`--bearing-data-dir`, `--cmapss-train-file`) | You're regenerating training data, changing burn-profile assumptions, or wiring in a different real dataset |
| `ISHM_LiquidPropulsion_FaultDetection_Colab.ipynb`, **cell 2** (generator defs) | A copy of the same generator logic, self-contained for Colab | You're iterating on the model in Colab and need to also tweak the generator — keep this in sync with the standalone `.py` file manually, they're not automatically linked |
| **Notebook cell 3** | `N_PER_CLASS` (dataset size), `bearing_dir`/`cmapss_file` real-data paths | Changing how much data to generate, or where the real datasets are cloned to |
| **Notebook cell 4** | `WINDOW_FRAC`, `OVERLAP`, `EfficientFCParameters()` choice, vibration hand-picked feature definitions (`vibration_features()`) | Changing windowing granularity or the tsfresh feature set's breadth/cost tradeoff |
| **Notebook cell 6** | Random Forest hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`) | Tuning the classifier itself |
| **Notebook cell 9** | `expected_chamber_pressure()` (the design throttle curve), `threshold_frac`, `sustained_fraction` | **This is the one that MUST change before real-flight use** — see section 5 |
| **Notebook cell 12** | Output artifact filenames, what gets bundled into the downloadable zip | Changing what ships in the trained-model package |

---

## 5. Using this in real time — what actually changes vs. what stays the same

### What has to change before this touches real telemetry
1. **`expected_chamber_pressure()` in cell 9 is currently the notebook's own synthetic design curve.** Before using Path B on a real engine, replace this function's body with your actual engine's real commanded-throttle-to-pressure design map (from your propulsion design data, not this notebook). Right now it will produce meaningless residuals against real telemetry — the notebook's docstring flags this explicitly, and `physics_detector_params.json` in the saved bundle carries a warning note to the same effect.
2. **Retrain the Random Forest on real telemetry once you have it**, replacing the synthetic+hybrid training data. The feature-extraction and training code doesn't need to change — only the data source (swap the generator's output for real logged telemetry in the same CSV schema).
3. **Recalibrate `threshold_frac`/`sustained_fraction`** (cell 9) against real engine noise characteristics — the current values (0.15, 0.1) were tuned against this notebook's synthetic noise level, not a real sensor's actual noise floor.

### What stays the same
- The windowing scheme (10% window length, 50% overlap).
- The tsfresh + Benjamini-Hochberg feature-selection methodology.
- The hybrid OR-fusion logic.
- The 13-channel input schema (assuming your real telemetry maps to the same parameter set — if it doesn't, check `LiquidPropulsion_Telemetry_Dictionary.md` first, since that's the source of truth for what should be monitored).

### Real-time inference loop (conceptual — not yet built as a deployable script)
```
every WINDOW_STEP seconds:
    1. pull the last WINDOW_LEN seconds of the 10 medium-rate channels + vibration
    2. extract tsfresh features on the medium-rate window (using the SAME
       selected_features.json list saved in the model bundle -- do not
       re-run select_features() at inference time, only extract the
       already-selected feature names)
    3. extract the 5 hand-picked vibration features
    4. rf.predict(features) -> fault_type for this window
    5. physics_residual_flag(latest chamber_pressure window) -> physics_flag
    6. hybrid_flag = (rf_pred != "normal") OR physics_flag
    7. if hybrid_flag: raise alert with fault_type, residual value, and
       confidence (rf.predict_proba) to the FDIR Level 1/2 aggregation layer
```

This loop isn't in the notebook yet (the notebook trains and evaluates offline) — it's the natural next artifact once you're ready to wire this into the actual data-ingestion pipeline from Phase 4 of the master project plan, and it belongs in your streaming/inference service, not in the training notebook itself.

---

## 6. Known limitations, stated plainly

- The Random Forest is trained entirely on synthetic + hybridized data. Its real-world accuracy is unknown until validated against actual engine telemetry — treat the notebook's reported metrics as a pipeline sanity check, not a real performance number.
- The physics detector's design curve is a placeholder shape, not a real engine's throttle map (see section 5.1).
- Vibration features are a compact hand-picked set (5 features), not a full tsfresh extraction, traded off deliberately for compute efficiency — if bearing-fault detection accuracy proves insufficient in practice, revisit this trade-off first.
- The hybrid fusion rule is the simplest possible (OR) — a confidence-weighted fusion using `rf.predict_proba()` alongside the physics residual magnitude is a documented, natural upgrade once real operational data exists to calibrate relative trust between the two detectors.
