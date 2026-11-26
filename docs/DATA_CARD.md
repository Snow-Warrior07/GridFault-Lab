# Data card: GridFault v0.1

**Dataset type:** generated phase-domain, fundamental-frequency electrical simulations. **No measured utility data, no instructor dataset, and no EMT waveform dataset are included.**

## Electrical source and assumptions

The positive-sequence branch R, X, shunt B and nominal load data use MATPOWER case9 on a 100 MVA, 345 kV equivalent base. Source: https://github.com/MATPOWER/matpower/blob/master/data/case9.m (accessed 26 November 2026). The code preserves the numerical constants in `config.py` and records provenance in `data/manifest.json`.

MATPOWER bus numbering maps to the review's diagram through `{1:1, 2:2, 3:3, 4:4, 5:6, 6:9, 7:8, 8:7, 9:5}`. Thus the monitored circuit is 7-8 in the adapted diagram. This circuit's positive-sequence impedance is 0.0085 + j0.072 pu. Its length is **assumed to be 120 km**; that length is not supplied by MATPOWER.

The monitored circuit uses a series-only model so that the implemented two-ended KVL locator has the same explicitly defined electrical assumptions. Its MATPOWER shunt susceptance is intentionally omitted; other network branches retain pi shunts. This should not be described as an unchanged MATPOWER benchmark or a complete replication of the paper's simulation.

Negative-sequence impedance equals positive-sequence impedance. Zero-sequence impedance is assumed three times positive for branches; generators use separate assumed grounded Norton equivalents. Detailed transformer winding connections, magnetic saturation, frequency dependence, and zero-sequence blocking are not modelled. Loads are constant admittances. Generator excitation/rotor dynamics are absent. This is a transparent linear circuit, not a converged constant-power dynamic generator study.

## Scenario generation

| Variable | Range or rule |
|---|---|
| Independent condition groups | 800 |
| Cases per group | 10 internal fault classes, 1 healthy case, 2 external faults |
| Physical scenarios | 10,400 |
| Observations per scenario | One clean and one noisy measurement record |
| Total records | 20,800 |
| Internal classes | AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC |
| Fault resistance | Log-uniform 0.5-80 ohm |
| Position | Continuous 5-95% of monitored line |
| Held-out distance bands | 20-25% and 70-75%; excluded from train/validation/calibration |
| Load scale | Uniform 0.65-1.35 |
| Source impedance scale | Uniform 0.75-1.25 |
| Source relative angle | Uniform 2-9 degrees for generator 2; generator 3 uses 0.6 times this |
| Source voltage scale | Uniform 0.97-1.05 |
| Line R and X scale | Independently uniform 0.95-1.05 |
| Load phase unbalance | Uniform -3% to +3%, with opposed A/B load factors |
| External faults | At buses 9 and 6; their fault classes are randomly sampled |
| Noise | Independent zero-mean Gaussian real/imaginary phasor perturbations at 20, 30, 40, or 50 dB |

AG faults connect one phase to ground through Rf; LL faults connect a phase pair through Rf. DLG faults use two Rf/2 phase arms and a shared Rf/2 ground-return arm. ABC uses three Rf arms to a floating common junction. These explicit resistor networks define the meaning of Rf. Balanced ABCG is not presented as a separately identifiable class.

For each voltage/current three-phase vector, noise variance uses the vector's RMS magnitude. SNR is defined in the **complex-phasor domain**, not for sampled waveforms. It is inappropriate to equate these noise levels with the reviews' raw-waveform SNR experiments. There are no independently generated fault-inception angles or transient windows in this model.

## Split and independence

| Partition | Condition groups | Measurement records | Purpose |
|---|---:|---:|---|
| Train | 480 | 12,480 | Fit forests, ANFIS, scaling, clustering and support bounds |
| Validation | 120 | 3,120 | Fit class-score temperature parameters |
| Calibration | 80 | 2,080 | Estimate accepted-fault location-error bands |
| Test | 120 | 3,120 | Final reported evaluation |

All thirteen event cases of a group and both noise variants remain together. `group_id` and `scenario_id` overlaps across partitions are tested. Continuous random positions in testing are supplemented by the explicitly held-out distance bands. Fixed RNG seed: **20261126**. Different seed/group-count configurations create different experiments.

Calibration/test records are correlated within a physical group. The location band is a finite-sample absolute-error quantile inspired by split conformal calibration, but it is reported as an **empirically assessed 90% target band**, not an unconditional distribution-free guarantee. Distance-band shift is an additional reason to report measured coverage.

## Stored features and targets

`data/dataset.csv.gz` is the complete feature/label dataset. It stores metadata, the twelve RMS channel magnitudes, real/imaginary phasors, twelve sequence magnitudes, differential-current magnitudes and a differential ratio. `data/manifest.json` records the checksum, provenance, split counts, features and excluded labels.

Prediction targets are `event`, `fault_type`, and normalized `location_pu`. Healthy and external cases have a null monitored-line position; there is no artificial location label for healthy operation. External fault class/bus information is recorded as ground truth but never used as input. True resistance, true line parameter scales, source/load conditions and noise settings also remain simulator metadata.

Not every metadata parameter can be predicted merely because it is stored as a label. This version does not train a fault-resistance, true-SNR, R/L/C, fault-inception-time, or control-action model.

## Measurement conventions

Voltages are **phase-to-neutral RMS phasors**. `base_kv=345` refers to the line-to-line base, so the phase voltage base is 345/sqrt(3) kV. The current base is 100e6/(sqrt(3)*345e3) A. Both terminal currents point **into the monitored line**. Phasors must share a reference for double-ended KVL. Metadata must explicitly declare synchronization; the software does not infer clock synchronization from equal array lengths or from a checkbox's presence.

## Appropriate interpretation

These data are suitable for studying input availability, a first ANFIS baseline, classical ML, physical location equations, error reporting, and reproducible deployment. They do not establish generalization to other lines, grounding systems, generator technologies, source/operating ranges, raw transient recordings or actual protective relay events. Future data should receive independent validation partitions and an updated model/data card.

