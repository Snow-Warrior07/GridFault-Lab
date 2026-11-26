# From the supplied papers to version 0.1

Page numbers refer to PDF positions, beginning at 1. The instructor PDFs are not included in the repository.

| Review | Sections used | Implemented decision |
|---|---|---|
| Stefanidou-Voziki et al., *A review of fault location and classification methods in distribution grids* (2022), DOI 10.1016/j.epsr.2022.108031 | Definitions pp. 1-2; Sections 2.3-2.5 pp. 7-11; classification pp. 12-14; gaps pp. 15-17 | Separate targets, measurement profiles, hybrid comparison, classification-specific metrics, missing-data and generalization limitations |
| Kanwal and Jiriwibhakorn, *Artificial Intelligence Based Faults Identification, Classification, and Localization Techniques in Transmission Lines-A Review* (2023), IEEE article 10305233 | II-B(1) pp. 4-5; Figures 4/6 pp. 5/8; Tables III/IV pp. 12-13 | Adapted IEEE 9-bus 7-8 circuit; six sending-end RMS inputs; separate 100-rule ANFIS models; explicit reference model rather than invented reported results |
| Panahi et al., *Advances in Transmission Network Fault Location in Modern Power Systems: Review, Outlook and Future Works* (2021), DOI 10.1109/ACCESS.2021.3129838 | VIII-A/B pp. 7-9; X-B p. 12; X-C(9)/(11) p. 14 | Single-/double-ended impedance baselines; synchronization gate; declared line-parameter assumptions; missing-terminal fallback |

## Deliberate corrections and adaptations

The second review reports 1,840 records from 40 cases and four sample positions, whereas this version generates a larger, independently partitioned continuous-position dataset. It uses explicit network/grounding/source assumptions because the paper does not supply a complete executable simulator and dataset. The result is an adapted reproduction, not an exact one.

The paper's sample table associates location values with healthy operation. This version sets healthy and external monitored-line location targets to null and excludes them from distance fitting. A healthy line has no fault position.

The reported testing location MSE 5.32 has units km squared if the target is in km; its reported RMSE is 2.31 km. Our errors are computed from our own generated test data and are not presented as the paper's results. Classification uses categorical metrics instead of MSE of arbitrary numerical class IDs.

The first review's distribution-specific conclusions are used as methodological guidance, not as proof that a transmission model generalizes. The third review's traveling-wave notation/sign and timing sensitivity need independent checks before any implementation. This release uses a clearly derived series-line KVL baseline instead of traveling-wave location.

## Deferred work

EMT-generated time windows, DWT/LSTM comparisons, realistic CT/CVT errors, detailed transformer zero-sequence paths, broader line/topology validation, pre-fault parameter identification, and simulated breaker control remain future work. The presence of a MATLAB algebra check does not mean those studies have been completed. Historical review gaps are not treated as proof of novelty in 2026.

