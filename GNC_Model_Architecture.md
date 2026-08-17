# GNC/AOCS Sensor Fault Detection — Model Architecture & I/O Reference
**Companion to `ISHM_GNC_FaultDetection_Colab.ipynb`.**

---

## 1. Architecture overview

```
              ┌───────────────────────────────┐
              │  RAW TELEMETRY (13 channels)   │
              │  IMU (accel+gyro), GPS (pos+   │
              │  vel), actuator position         │
              └───────────────┬─────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        ▼                                             ▼
┌──────────────────────────────┐      ┌────────────────────────────────┐
│ PRIMARY: UKF analytical       │      │ SECONDARY: classifier on UKF   │
│ redundancy (no training data) │      │ residual features               │
│                                │      │                                  │
│ 1. Predict state from IMU     │      │ 1. tsfresh features on the       │
│    accel (6-state: pos+vel)   │      │    UKF's innovation sequence     │
│ 2. Update from GPS pos+vel     │      │    (NOT raw sensor data)         │
│ 3. Compute NIS (chi-square      │      │ 2. Benjamini-Hochberg selection  │
│    consistency test)            │      │ 3. Random Forest -> WHICH fault  │
│ 4. Flag if NIS sustained-        │      │    occurred (6 classes)           │
│    exceeds threshold             │      └────────────────┬─────────────────┘
└───────────────┬────────────────┘                        │
             ukf_flag (0/1)                          rf_flag (0/1) + fault_type
                │                                            │
                └──────────────────┬─────────────────────────┘
                                   ▼
                    hybrid_flag = ukf_flag OR rf_flag
                                   │
                                   ▼
                OUTPUT: fault_type, hybrid_flag, NIS trace
```

**Why UKF first:** analytical redundancy (comparing what the IMU predicts against what the GPS reports) is standard, certifiable aerospace practice — it needs no training data and is directly explainable. The classifier adds *which fault* on top of *whether something's wrong*.

---

## 2. Inputs — exact schema

| Column | Unit | Used by |
|---|---|---|
| `accel_x/y/z_ms2` | m/s² | UKF predict step |
| `gps_lat_deg`, `gps_lon_deg`, `gps_alt_m` | deg, deg, m | UKF update step (converted to local meters first) |
| `gps_vel_n/e/d_ms` | m/s | UKF update step |
| `gyro_x/y/z_rads`, `actuator_pos_deg` | rad/s, deg | **Not used by this model** — generated for completeness/future subsystems, ignored by both the UKF (translational-only, see honest simplification below) and the current classifier |

---

## 3. Outputs — exact schema

| Output | Type | Meaning |
|---|---|---|
| `ukf_flag` | 0/1 | 1 if NIS sustained-exceeds the chi-square(6, 99.9%) threshold (~22.46) |
| `mean_nis`, `max_nis` | float | Raw NIS statistics, for trending/logging even below threshold |
| `rf_pred` | string | One of 6 fault_type labels, predicted from residual features |
| `rf_flag` | 0/1 | 1 if `rf_pred != "normal"` |
| `hybrid_flag` | 0/1 | `ukf_flag OR rf_flag` — the final decision |

6 fault classes: `normal`, `gps_spoofing`, `gps_degradation`, `imu_bias_drift`, `combined_fault`, `actuator_fault`.

**Important, verified behavior:** `actuator_fault` never triggers `ukf_flag` (by design — the UKF doesn't watch the actuator channel at all) and is caught entirely by the classifier. This is the concrete example of "primary + secondary, not primary alone" actually mattering, not just a design principle stated in the abstract.

---

## 4. How this notebook actually came together — a fuller account than usual

This model went through an unusually long debugging cycle, and it's worth documenting honestly rather than presenting the final version as if it worked on the first try. Building a UKF-based consistency check is fundamentally different from the other subsystems' models: it depends on IMU and GPS being *physically consistent* with each other in the normal case, which none of the other models required. That single requirement surfaced a chain of real bugs, each only visible once the previous one was fixed and the numbers were actually checked against physical expectation (chi-square(6) ≈ 6 for a well-tuned filter) rather than just "the code ran without an error":

1. **Independently-generated IMU and GPS channels.** The original generator synthesized each channel from its own formula, with no physical relationship between them. A UKF comparing "what IMU integration predicts" against "what GPS reports" saw enormous, universal mismatches — on *every* instance, fault or not — because there was nothing for the two to be consistent *with*. Fixed by deriving GPS position/velocity from actually integrating the accelerometer channel.
2. **Windowed FFT perturbation on a doubly-integrated channel.** The synthetic-augmentation method used elsewhere in this project (perturb a window's FFT coefficients) can inject near-DC spectral artifacts that are invisible in the raw signal but become enormous spurious drift once double-integrated — which is exactly what the UKF does to the accelerometer channel. Fixed by giving accelerometer channels bounded additive noise instead.
3. **The same bug again, via GPS's large constant offset.** `gps_lon_deg` carries a huge constant reference term (e.g. -118°); perturbing it directly multiplies near-DC spectral energy by that constant, turning a "2% perturbation" into a multi-kilometer spurious jump once converted to meters. Confirmed directly: longitude jumped -118.0 → -119.3 in one second with no physical cause. Fixed by adding bounded noise to position *in meters*, before the degrees conversion.
4. **A zero-padding edge artifact in trend estimation.** The `smoothed_trend()` helper (used to extract "noise residual" from real data and to calibrate blend scales) used zero-padded convolution, which drags the trend estimate toward zero at the array boundaries. Confirmed directly: a signal sitting at ~34 showed an edge trend value of ~17.7 — almost exactly what zero-padding against half-real/half-phantom-zero values predicts. Those ~24 corrupted samples then dominated every downstream variance calculation. Fixed with a proper edge-aware rolling mean.
5. **Real-data blending that erased the fault signal it was supposed to preserve.** Rescaling each fault class's real residual to match a fixed target noise level meant *every* class — baseline and spoofing alike — ended up at the same injected magnitude by construction, regardless of how different they actually were in the source data. Fixed by calibrating a single scale factor from real baseline data only, then applying that same factor to every fault class, which preserves whatever relative severity the real data actually contains.
6. **Spoofing/bias-drift are trend-level effects, not noise-level effects.** Once the above was fixed, `gps_spoofing` and `imu_bias_drift` still showed no separation from normal — because these fault types manifest in the real data as a slow divergence in the underlying trend, and detrending (step 5's whole mechanism) specifically *removes* trends by design. Fixed by adding a second blending term that injects the deviation between each fault class's real trend and baseline's real trend, calibrated the same baseline-first way.
7. **Most files in the real dataset's fault folders are near-identical controls.** The dataset has 17 recordings per condition folder; picking the alphabetically-first file (as the code did) mostly landed on recordings with negligible injected fault content (~50m of GPS noise, not a real spoofing event). One specific file was confirmed to contain genuine, substantial fault signatures (~45km position divergence for hijack vs. baseline) across all conditions and is now used explicitly.

**Verified result after all seven fixes:** `normal` NIS sits at ~4.8 (near the chi-square(6) theoretical expectation of 6); every real fault class shows 10-25x elevation; `actuator_fault` correctly stays low since the UKF doesn't watch that channel. This was confirmed by rerunning the notebook, not assumed from the code looking reasonable.

---

## 5. Which parameters live in which file

| File | What it defines | Edit when... |
|---|---|---|
| `GNC_Telemetry_Dictionary.md` | Parameter ranges, fault-class definitions | Changing what a fault means physically |
| **Notebook cell 2** | Generator (trajectory, noise model, blending, `PREFERRED_REAL_FILE`) | Regenerating data or fixing a data-quality issue in a future dataset version |
| **Notebook cell 3** | `N_PER_CLASS` | Dataset size |
| **Notebook cell 4** | UKF class (`q_c`, `r_std_pos`, `r_std_vel`, process/measurement model) | **Must be re-tuned against real IMU/GPS noise specs before flight use** |
| **Notebook cell 5** | `NIS_THRESHOLD`, `sustained_fraction` | Adjusting detection sensitivity/false-alarm tradeoff |
| **Notebook cell 10** | Random Forest hyperparameters | Tuning the secondary classifier |

---

## 6. Known limitations, stated plainly

- The UKF is translational-only (position/velocity), not a full strapdown INS with attitude mechanization — stated in the notebook intro, not hidden.
- `q_c`/`r_std_pos`/`r_std_vel` are tuned against this synthetic+real-hybrid dataset's noise characteristics, not a specific real IMU/GPS pair's datasheet values.
- The real dataset's fault content is concentrated in one specific recording out of 17 per folder; broader validation across more of the real files would strengthen confidence before relying on this for anything beyond a pipeline demonstration.
- Reported classifier accuracy at small `N_PER_CLASS` (this doc's own testing used N=5 for speed) is not representative of real-world performance — rerun at the notebook's default N=20+ and treat results as a pipeline sanity check, consistent with every other subsystem model in this project.
