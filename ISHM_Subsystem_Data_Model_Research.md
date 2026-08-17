# Deep Research: Subsystems, Data Types, Parameter Counts, and Model Selection
**Companion to the Master Project Plan — answers: which subsystems, what data, how many parameters, which model**

---

## 1. Subsystem taxonomy actually used in rocket/missile ISHM

Not every vehicle has every subsystem below — a missile and an orbital launch vehicle diverge (missiles usually have no cryogenic tanks; launchers usually have no seeker/warhead). Treat this as the superset to select from for Phase 2.

| # | Subsystem | Present in | Primary failure concern |
|---|---|---|---|
| 1 | Propulsion — liquid engine | Launch vehicles, some missiles | Combustion instability, turbopump bearing wear, valve/feed-system degradation |
| 2 | Propulsion — solid rocket motor (SRM) | Missiles, boosters | Case-bond debonding, propellant grain cracking, nozzle erosion |
| 3 | Guidance, Navigation & Control (GNC) / AOCS | All | IMU drift, GPS/GNSS spoofing or dropout, actuator/servo degradation |
| 4 | Structures & Thermal Protection System (TPS) | All | Fatigue cracking, thermal-induced delamination, MMOD impact |
| 5 | Avionics / on-board electronics | All | Solder-joint fatigue from thermal cycling, connector degradation |
| 6 | Power (battery + distribution) | All | State-of-health fade, thermal runaway precursors, bus voltage sag |
| 7 | Telemetry & communications | All | Link dropout, antenna/connector degradation |
| 8 | Stage separation / pyrotechnics | Multi-stage vehicles | Separation timing faults, electronic initiator failure |
| 9 | Seeker / terminal guidance | Missiles only | Sensor blinding, tracking loss |
| 10 | Cryogenic tanks | Launch vehicles (LOX/LH2) | Insulation icing, boil-off, structural stress from thermal gradient |

Your own project already has a working reference for #3-style parameter classification (the DRDO angular-rate paper) — that's your fastest subsystem to prototype first.

---

## 2. Data/sensor types gathered, per subsystem

| Subsystem | Sensor types | Signal characteristics |
|---|---|---|
| **Liquid propulsion** | High-temp piezoelectric pressure transducers (chamber, injector, feed lines — rated to ~550°C near-chamber, cryogenic-capable down to -196°C), piezoelectric accelerometers, thermocouples/RTDs, flow meters, electrostatic sensors for exhaust asymmetry, strain gauges on the pressure vessel wall | Chamber pressure and accelerometer channels are typically sampled in the kHz range to resolve combustion instability frequencies; temperature and flow are low-rate (Hz range) |
| **Solid rocket motor** | Dual bond-stress/temperature (DBST) sensors at the case wall, embedded fiber Bragg grating (FBG) strain sensors, polymer optical fiber (POF) sensors bonded into the propellant itself, ultrasonic/X-ray for ground inspection | Strain and temperature at case-bond interface, continuous during storage and firing; POF can also serve as a plastic-deformation event logger for stress history |
| **GNC/AOCS** | MEMS or ring-laser IMU (accelerometers + gyros), GNSS/GPS receiver, magnetometer, star tracker/sun sensor (launch vehicles), radar or infrared seeker (missiles), actuator position feedback | IMU at 100+ Hz, GPS typically 1-10 Hz (rate mismatch is itself a fusion design constraint), seeker frame rate depends on sensor type |
| **Structures/TPS** | Strain gauges, accelerometers (vibration/shock), thermocouples, piezoelectric wafer active sensors (PWAS) for guided-wave damage detection, fiber-optic distributed sensing | Vibration/shock at kHz, thermal gradients at Hz, PWAS excitation/response in the ultrasonic range |
| **Avionics/electronics** | Thermocouples on PCB/component level, current/voltage monitors, in-situ solder-joint resistance monitoring, vibration sensors on chassis | Thermal cycling (steady-state temp, ramp rate, dwell time) is the dominant driver — sampled continuously but analyzed over mission/cycle timescales, not instantaneously |
| **Power** | Cell voltage, current, temperature per cell/module, internal impedance (via signal injection), sometimes ultrasonic SOC/SOH sensing | Voltage/current/temperature at Hz-range; impedance spectroscopy is a periodic diagnostic event, not continuous |
| **Telemetry/comms** | Signal strength (RSSI), bit-error rate, antenna temperature/position, transmitter power draw | Low-rate housekeeping channel |
| **Stage separation** | Position sensors, sometimes temperature, magnetometers, initiator continuity/current monitors | Event-triggered, high-rate burst around the separation event, otherwise idle |

---

## 3. How many parameters is "enough"? — quantitative guidance

There's a real, published answer to this rather than a rule of thumb, and it comes directly from the two most relevant papers:

**From the DRDO/MVSR paper (your closest working reference):** they modeled a subsystem using **5 parameters** (angular rates AR1-AR5). For each single parameter, `tsfresh` generated **1558 candidate features** (16 statistical + 8 spectral + 7 temporal families expanded across window statistics). After Benjamini-Hochberg significance filtering, this was reduced to **31 selected features per parameter** — and critically, their ablation (Table IX) showed that going from 31 to 8 hand-picked features per parameter cost only 3-7 percentage points of accuracy (99.9% → 96.8% for Rule I; 76.3% → 69.0% for Rule II). **Implication for your build:** don't hand-tune features from scratch — run the full 1558-feature `tsfresh` extraction on each raw parameter, apply automatic statistical feature selection (Benjamini-Hochberg or similar), and only manually curate features if you specifically need to shrink model size or shorten training-model file length (their file-length column ranged from 154,300 to 4,783,300 samples depending on feature count).

**Parameter-count budget by subsystem** (informed by the sensor catalogues above — this is a *starting* count, not a hard limit; expand once real telemetry access is validated):

| Subsystem | Minimum viable parameter count | Rationale |
|---|---|---|
| Liquid propulsion (per engine) | 8-12 | Chamber pressure, 2 injector pressures, 2 feed-line pressures/flows, 1-2 accelerometer axes, 2-3 temperatures — matches published LRE fault-detection studies using pressure + temperature + flow + vibration as the core signal set |
| Solid rocket motor | 4-6 | Case-bond stress, case-bond temperature, 1-2 embedded strain channels — SRMs have far fewer real-time channels than liquid engines since most SHM there is ground-based/destructive testing today |
| GNC/AOCS | 6-9 | 3-axis accelerometer + 3-axis gyro (IMU core) + GPS position/velocity residual + actuator position feedback |
| Structures/TPS | 5-10 per monitored zone | Multiply by number of critical zones (typically 3-6 zones: interstage, TPS panels, high-stress joints) |
| Avionics | 3-5 per board/module | Board temperature, current draw, voltage rail(s), sometimes a vibration channel |
| Power | 4-6 per battery module | Cell voltage, current, temperature, and a periodic impedance/SOH estimate |
| Telemetry | 3-4 | RSSI, BER, transmitter power, antenna temperature |
| Stage separation | 2-4 | Position/displacement, initiator continuity/current, sometimes temperature |

**Total system-level count** for a 5-subsystem missile-class vehicle: roughly **35-55 raw parameters**, each expanded via feature engineering (tsfresh-style) into 15-30+ derived features before subsystem-level classification — consistent with the scale the DRDO paper worked at, just multiplied across more subsystems than their single angular-rate example.

---

## 4. Model selection, per subsystem — compare at least three approaches

Per your own project instructions, no single model per subsystem — here's the comparison, grounded in what's actually published for each subsystem type.

### 4.1 Liquid propulsion — fault/anomaly detection

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| Classical ML on statistical features (tsfresh + LR/Decision Tree/Random Forest) | Same pipeline as your DRDO reference, applied to pressure/temperature/flow instead of angular rate | Fast to train, interpretable, small footprint — good first baseline | Loses fine-grained combustion-instability frequency content unless spectral features are included |
| Least Squares Support Vector Regression (LSSVR) | Published real-time fault-detection approach specifically for hydrogen-oxygen liquid rocket engines, using vibration, pressure, temperature, and flow-rate signals together | Designed for real-time onboard use; handles the multi-signal-type fusion this subsystem needs | Requires careful kernel/hyperparameter tuning; less interpretable than tree-based methods |
| Physics-based / analytical redundancy (mathematical combustion + feed-system model, residual generation) | Model expected pressure/flow relationships from first-principles propulsion equations; flag deviations as residuals | No training data needed; directly explainable in propulsion-engineering terms | Requires an accurate engine model per configuration — expensive to build and maintain per vehicle variant |

**Recommendation for your first build:** start with tsfresh + Random Forest (reuses your validated pipeline), add a physics-based residual check on chamber pressure vs. commanded throttle as a second, independent detector — this gives you the hybrid architecture your project instructions require without committing to LSSVR complexity on day one.

### 4.2 Solid rocket motor / structural elements — defect detection

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| Deep neural network on embedded fiber-optic strain data | A published approach trains DNNs on FEA-simulated strain patterns from embedded optical sensors, achieving over 98% accuracy discriminating healthy vs. damaged case-bond conditions, and predicting crack depth to ~2.3mm and delamination angle to ~1.6° | Very high accuracy demonstrated; works from non-destructive embedded sensors rather than ground-based destructive testing | Needs FEA-simulated training data since real damaged-motor examples are scarce and expensive |
| Piezoelectric guided-wave analysis (PWAS pulse-echo / phased array) | Impulse-and-detect approach for crack/delamination location in structural elements more broadly | Physically interpretable — directly measures wave reflection from a defect | Signal degradation and attenuation issues in foam/composite materials; more complex excitation electronics |
| Temperature-gradient / bond-stress threshold monitoring (DBST sensors) | Traditional dual bond-stress-temperature sensors embedded at the case wall, simple threshold-based alerting | Simple, low computational cost, already flight-proven on many SRM programs | Limited multiplexing (few measurement points), reactive rather than predictive — detects damage after a stress event, not degradation trend |

**Recommendation:** start with DBST threshold monitoring (cheapest, flight-proven), layer the DNN-on-embedded-strain-data approach on top once you have (or simulate via FEA) enough labeled healthy/damaged examples — this is a case where synthetic/simulated training data genuinely substitutes for scarce real failure data.

### 4.3 GNC/AOCS — sensor fault detection and fusion

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| Extended Kalman Filter (EKF) bank with Normalized Solution Separation | Multiple EKFs fuse IMU + GNSS + a vehicle dynamics model; fault detection compares filter outputs to isolate a faulty sensor | Proven aerospace-standard approach; explainable, real-time-capable | Assumes reasonably linearizable dynamics; performance degrades under highly nonlinear maneuvers (e.g. missile terminal phase) |
| Unscented Kalman Filter (UKF) fusion | Same fusion concept as EKF but handles nonlinear dynamics without linearization, shown to reduce GNSS-only position RMSE by roughly 3-60x across axes in published autonomous-vehicle fusion work | Better nonlinear performance than EKF; still a well-understood, explainable Bayesian method | Higher computational cost per update than EKF |
| Machine-learned residual classifier (deep learning fusion, e.g. LSTM on IMU+GNSS residual sequences) | Learns fault signatures directly from labeled fault-injected training data instead of hand-derived residual thresholds | Can catch fault patterns a hand-built residual test misses | Needs substantial labeled fault data (usually simulation-injected); less certifiable/explainable than Kalman-filter residuals |

**Recommendation:** EKF/UKF-based analytical redundancy should be your primary GNC health-monitor (this is standard, certifiable practice per the FDIR paper's analytical-redundancy discussion) — reserve the learned classifier as a secondary anomaly detector layered on top of the Kalman residuals, not a replacement for them.

### 4.4 Avionics/electronics — degradation and RUL

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| Physics-based fatigue/damage modeling (thermal-mechanical stress analysis + cumulative damage models, e.g. Miner's rule variants or total strain-energy-density models) | Uses actual mission thermal-cycle history (steady-state temp, ramp rate, dwell time) to compute accumulated solder-joint fatigue damage and predict RUL | Grounded in decades of electronics-reliability physics; directly explainable to a reliability engineer | Needs an accurate thermal/mechanical model per board design — significant upfront modeling effort |
| Data-driven RUL regression (regression trees, ensemble RNNs) on in-situ thermal/vibration monitoring | Learn degradation trends directly from monitored temperature/vibration time series | Adapts to actual operating conditions rather than assumed mission profile | Needs run-to-failure or accelerated-life training data, which is scarce for flight electronics specifically |
| Threshold/rule-based monitoring (steady-state temp, ramp rate limits) | Simple bounds-checking against known-safe thermal cycling limits | Cheapest to implement, matches current industrial avionics-health practice | No prognostic capability — only tells you a limit was exceeded, not how much life remains |

**Recommendation:** physics-based cumulative damage modeling is the right primary approach here — this field is unusually mature on the physics side (this is literally what "physics-informed" was invented for in your own AI Design Principles), with data-driven RUL as a secondary cross-check once you accumulate enough operational history.

### 4.5 Power (battery) — SOH/SOC estimation

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| Electrochemical model + differential voltage/capacity analysis | Tracks how measured voltage-vs-capacity curves deviate from anode/cathode reference curves to isolate degradation mode | Physically grounded, can distinguish *which* electrode is degrading | Requires cell-chemistry-specific reference curves |
| Equivalent-circuit + Kalman filter (impedance-based SOC/SOH tracking) | Models the battery as an equivalent circuit, uses filtering to track internal resistance growth over time | Standard, computationally cheap, works online during flight | Equivalent-circuit parameters drift and need periodic recalibration |
| Data-driven regression/classifier on voltage-current-temperature history | Learns SOH directly from historical charge/discharge curves | Captures real degradation behavior without needing a chemistry model | Needs substantial historical cycling data per battery type/lot |

**Recommendation:** equivalent-circuit + Kalman filter is the standard, flight-proven approach and should be your baseline; add differential-capacity analysis as a secondary check specifically before flight (ground diagnostic), since it needs a slower, more controlled charge/discharge cycle to compute cleanly.

---

## 5. Consolidated model-per-subsystem summary table

| Subsystem | Primary model | Secondary/cross-check model | Data need |
|---|---|---|---|
| Liquid propulsion | tsfresh features + Random Forest/LSSVR | Physics-based residual (expected pressure vs. throttle command) | Pressure, temperature, flow, vibration |
| Solid rocket motor | DBST threshold monitoring | DNN on embedded FBG/POF strain (FEA-trained) | Bond stress, temperature, strain |
| GNC/AOCS | EKF/UKF sensor fusion + residual fault detection | Learned residual classifier (secondary anomaly layer) | IMU, GNSS, actuator feedback |
| Structures/TPS | PWAS guided-wave + threshold | CNN/DNN defect classifier once labeled data exists | Strain, vibration, temperature |
| Avionics | Physics-based cumulative damage (thermal cycling) | Data-driven RUL regression once operational history accumulates | Temperature, current/voltage, vibration |
| Power | Equivalent-circuit Kalman filter | Differential-capacity ground diagnostic | Voltage, current, temperature |
| Telemetry/comms | Threshold/rule-based | — (low complexity subsystem, doesn't need a hybrid layer yet) | RSSI, BER, power draw |
| Stage separation | Threshold/rule-based (Rule I/II from your DRDO reference) | Event-window classifier on initiator current signature | Position, initiator current/continuity |

This table is your Phase 6 starting point — one row per subsystem, each already assigned a primary+secondary model pair with a documented reason, which satisfies your own project instructions' requirement to never pick a single model without comparing alternatives.
