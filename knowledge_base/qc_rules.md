# ECG QC Rules

## Goal
Provide deterministic quality-control rules for ECG signals before feature extraction.

## Required outputs
- `ecg_qc.parquet`
- Required fields:
  - `record_id`
  - `qc_pass` (bool)
  - `qc_reasons` (list[string] or string)

## Rule categories
- Signal availability:
  - Missing waveform data => fail reason: `missing_signal`.
- Flatline detection:
  - Extremely low variance for sustained window => fail reason: `flatline`.
- Clipping detection:
  - Saturation ratio above threshold => fail reason: `clipping`.
- NaN ratio:
  - Missing sample ratio above threshold => fail reason: `nan_ratio_high`.
- Basic physiological plausibility:
  - Implausible dominant heart rate estimate => fail reason: `hr_implausible`.

## Reporting
- Always compute QC pass rate.
- Always report Top-N fail reasons.
- QC thresholds must be versioned and auditable.
