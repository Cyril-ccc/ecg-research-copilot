# Cohort Templates Reference

## Purpose
This document defines approved cohort templates used by the planner and execution pipeline.
The content is policy-level guidance only and is not executable SQL.

## template: electrolyte_hyperkalemia
- Intent: Build cohorts for hyperkalemia-related research questions.
- Required params:
  - k_threshold (float): default 5.5
  - window_hours (int): default 6
  - label_keyword (string): default "potassium"
- Output expectation:
  - Must include `subject_id`, `hadm_id`, `index_time`, `cohort_label`.

## template: diagnosis_icd
- Intent: Build diagnosis cohorts using ICD prefixes.
- Required params:
  - icd_prefixes (list[string])
  - window_hours (int): default 24
- Output expectation:
  - Must include `subject_id`, `hadm_id`, `index_time`, `cohort_label`.

## template: medication_exposure
- Intent: Build pre/post drug exposure cohorts.
- Required params:
  - source (string): default "prescriptions"
  - drug_keywords (list[string])
  - pre_hours (int): default 24
  - post_hours (int): default 24
- Output expectation:
  - Must include `subject_id`, `hadm_id`, `index_time`, `cohort_label`.

## Constraints
- Use only approved templates.
- Do not generate ad-hoc SQL from user text directly.
