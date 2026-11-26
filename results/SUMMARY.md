# Held-out results

| Profile | Family | Event macro-F1 | Type macro-F1 (true internal) | MAE km | RMSE km | Internal location availability | Interval test coverage | False non-internal locations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rms_single | forest | 0.9962 | 0.9992 | 9.141 | 13.076 | 99.96% | 92.25% | 4 / 720 |
| rms_single | anfis | 0.9985 | 1.0000 | 7.807 | 11.018 | 99.67% | 91.01% | 0 / 720 |
| rms_receiving | forest | 0.9647 | 1.0000 | 14.344 | 18.799 | 96.50% | 88.77% | 13 / 720 |
| rms_double | forest | 0.9987 | 0.9996 | 3.657 | 5.295 | 100.00% | 89.54% | 3 / 720 |
| phasor_double | forest | 1.0000 | 1.0000 | 1.377 | 2.770 | 91.88% | 90.16% | 0 / 720 |

Synthetic fundamental-frequency benchmark only. Location errors are conditional on accepted internal faults; availability and false locations are shown to prevent selective reporting. Type scores use oracle internal-event membership. No field/EMT performance claim. Calibration intervals target 90%; reported coverage is empirical.
