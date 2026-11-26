# Model card: GridFault v0.1

## Tasks and routing

An explicit input validator first checks the registered benchmark, line, bases, current direction contract, numeric finiteness, available channels and declared synchronization. The automatic router prefers synchronized two-ended phasors, then both-ended RMS, then whichever supported single end is present. It never manufactures phase angles or fills a missing terminal with zeros.

Four profiles are separately trained: six RMS features at bus 7; six at bus 8; twelve RMS features across both ends; and 52 features for synchronized phasors (12 magnitudes, 24 real/imaginary components, 12 sequence magnitudes, and 4 differential-current features).

## Random Forest models

Each profile has an event classifier for healthy/internal/external events and a type classifier for ten internal fault classes. Type training uses internal-fault records only. Forests have 40 trees, maximum depth 13, minimum leaf size 3, fixed seed, and balanced-subsample class weights for classification. Bootstrap sampling is used by scikit-learn's Random Forest implementation.

RMS profiles also have a Random Forest regressor for d/L, trained on internal faults only. The phasor profile uses a physical distance equation instead of a learned location regressor. Thus the name `forest` in that profile denotes the event/type models plus the explicit electrical locator.

Trees receive consistent per-unit inputs but do not require z-score scaling. Simulator labels/nuisance settings are excluded. Entire operating groups are held out before any noisy variants are generated. Full parameters and all decision-tree nodes are exported in `site/assets/model-*.json`.

## ANFIS reference

The sending-end six-RMS-input profile includes three separate first-order Gaussian Sugeno models for event, type and distance. Each has 100 rules. Training-only mean/std scaling is followed by training-only k-means premise initialization. Gaussian firing strengths are normalized with a stable softmax of log firing strengths. A rule's consequent is an affine function of the six standardized features. The model output is the weighted sum of these local functions.

The hybrid optimizer fits ridge least-squares consequents (alpha 0.03), updates Gaussian centers and log widths with exact squared-error gradients, and refits consequents. The release performs two premise-update epochs followed by a final consequent fit. Gaussian widths are bounded for numerical stability. Multi-output one-hot targets train the classifiers; categorical metrics evaluate decoded classes rather than using MSE of arbitrary integer class codes.

This implementation is aligned with the supplied paper's six-input, Gaussian/Sugeno, 100-cluster idea. It is not MATLAB's `anfis` function, not a claim to reproduce the paper's undocumented training procedure, and not a claim to reproduce its numerical error values.

## Electrical locator

With phase-domain line matrix Z and both currents oriented into the line:

```
Vf = Vs - m Z Is = Vr - (1-m) Z Ir
a = Z (Is + Ir)
b = Vs - Vr + Z Ir
m_hat = Re(a^H b) / (a^H a)
distance_km = 120 m_hat
```

The least-squares estimate uses all phases with a real-valued distance. It does not require the true fault class or fault resistance for the series-only monitored-line model. It uses nominal registered impedance, while generation varies actual R and X by ±5%. A normalized residual norm(b-m*a)/(norm(b)+epsilon) above 0.35 withholds location. A near-zero denominator or m outside [0,1] also withholds location. Points are not silently clipped into the line.

The single-ended reactance-loop comparator uses the true class for loop selection and is explicitly labelled an oracle-type baseline. It remains affected by resistance/remote infeed. Neither comparator is a universal locator for charged/distributed/compensated/branched lines without adapting its equations.

## Probability calibration and abstention

Temperature scaling minimizes negative log likelihood on validation data only. For forests, inputs are log vote/probability values; ANFIS uses its learned output scores. Scores below 0.65 withhold an event diagnosis; type scores below 0.55 withhold a type label. The distance estimator is independent of the type output, so a supported internal fault can still receive distance when its type is uncertain.

An axis-aligned feature-support screen uses training minima/maxima extended by 15% of range (with a small numerical margin). It is a useful heuristic, **not a complete out-of-distribution detector**. It cannot identify every unseen circuit or measurement error. Registered-line and units checks are therefore mandatory.

Healthy/external event predictions receive no in-line location. Actual mistakes can still occur when external cases are misclassified as internal, particularly with RMS-only measurements. Their false-location counts are explicitly reported.

## Error bands

After all model parameters and decision gates are fixed, accepted internal calibration cases determine the absolute location-error quantile at finite-sample rank ceil((n+1)*0.9). A symmetric band uses this radius; interval endpoints are intersected with [0,120]. The point itself is never clipped. Calibration target is 90%; actual test coverage is reported separately for each profile.

Correlated variants, held-out distance bands, and future domain changes prevent interpreting this as an unconditional exact 90% guarantee. Never substitute model confidence for a validated uncertainty statement.

## Evaluation and limitations

See [actual results](../results/SUMMARY.md) and `results/evaluation.json`. Event/type classification, location errors, location availability, false non-internal locations, calibration error, and interval coverage have separate denominators. Type confusion matrices use oracle membership in the internal-fault subset; final-pipeline class coverage also counts missed or withheld events.

The strongest two-ended result reflects the simplicity and consistency of this linear synthetic benchmark, including omission of monitored-line charging. Field data, EMT transients, CT/CVT distortion, inverter limiting, clock drift, detailed transformer grounding, distributed capacitance, mutual parallel-line effects and other topologies require further experiments and potentially new models.

The browser model accepts JSON numbers; it does not deserialize uploaded executables or pickle models. No upload is transmitted to a prediction server. Browser/Python equality is checked on 120 held-out fixtures, including ANFIS and each measurement profile.

