# ECG Feature Definitions

## Goal
Define feature semantics and formulas for the analysis pipeline.

## Core timing features
- RR interval (`rr_mean`, `rr_std`):
  - `rr_mean`: mean R-R interval in seconds.
  - `rr_std`: standard deviation of R-R interval in seconds.

## Heart-rate features
- Mean heart rate (`mean_hr`):
  - Formula: `mean_hr = 60 / rr_mean` when `rr_mean > 0`.

## QT/QTc features
- QT interval (`qt_ms`):
  - Measured from Q onset to T wave end, in milliseconds.
- RR interval in QT correction (`rr_sec`):
  - RR interval in seconds.
- Corrected QT (`qtc_ms`):
  - Bazett default: `qtc_ms = qt_ms / sqrt(rr_sec)`.
  - Alternative methods (if enabled): Fridericia, Framingham.
  - Method used must be stored in metadata.

## Sanity ranges
- `mean_hr`: expected research sanity range 30-180 bpm.
- `rr_std`: must be non-negative.

## Missingness policy
- Feature-level missing rates must be monitored.
- Exceeding configured thresholds should fail evaluation checks.
