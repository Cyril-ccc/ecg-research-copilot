# Report Pipeline Contract (v1)

统一根目录：`artifacts_root/<run_id>/...`（示例：`storage/artifacts/<run_id>/...`）。

## 1) 输入契约（来自 artifacts）

| 文件 | 是否必需 | 必需字段/键 | 可选字段/键 | 备注 |
|---|---|---|---|---|
| `cohort.parquet` | 是 | `subject_id`, `hadm_id`, `index_time` | `cohort_label`, `record_id`, `age`, `gender` | `record_id` 缺失时不可做 ECG 逐条分析 |
| `ecg_map.parquet` | 是（做 ECG 逐条分析时） | `record_id`, `subject_id` | `ecg_time`, `source` | run 内冻结映射，不依赖全局 manifest 实时状态 |
| `cohort_summary.json` | 是 | `distinct_subjects`, `total_rows`, `missing_rates` | `gender_distribution`, `age_distribution`, `index_time_range` | 与现有 `build_cohort` 输出一致 |
| `ecg_qc.parquet` | 是 | `record_id`, `qc_pass`, `qc_reasons`, `qc_version` | `flatline_ratio`, `clipping_ratio`, `nan_ratio`, `powerline_score`, `baseline_wander_score` | 由 `pipelines/ecg_qc.py` 产出 |
| `ecg_features.parquet` | 是 | `record_id`, `mean_hr`, `rr_mean`, `rr_std`, `lead_amplitude_p2p_mean`, `lead_amplitude_p2p_std`, `feature_version`, `qc_version` | `source`, `detected_peak_count`, `code_commit` | 由 `pipelines/ecg_features.py` 产出 |
| `covariates.parquet` | 否 | 至少一组连接键：`record_id` 或 `subject_id`(+`hadm_id`) | `age`, `gender`, `race`, `bmi`, `comorbidity_*` | cohort 缺协变量时补充 |

### 连接规则（固定）

1. `ecg_qc.parquet` 与 `ecg_features.parquet` 使用 `record_id` 内连接。  
2. `analysis_dataset` 拼接时，`record_id -> subject_id` 必须来自 run 内冻结的 `ecg_map.parquet`。  
3. cohort/covariates 与 ECG 合并优先 `record_id`。  
4. 若无 `record_id`，允许输出 cohort 级统计；涉及 ECG 特征比较必须失败并在元数据写明原因。

## 2) 输出契约（写回同一 run）

### 固定输出路径

- `analysis_tables/analysis_dataset.parquet`（report 前置拼接底表）
- `analysis_tables/cohort_counts.parquet`
- `analysis_tables/feature_summary.parquet`
- `analysis_tables/group_compare.parquet`
- `plots/`
- `report.md`
- `run_metadata.json`

### 输出字段定义

`analysis_tables/analysis_dataset.parquet`（report 用长表底座）
- 至少包含：`record_id`, `subject_id`, `cohort_label`, `index_time`
- 以及：`sex`（可得则填，否则 `Unknown`）, `age`, `age_bin`, `dataset_source`
- ECG 特征列来自 `ecg_features.parquet`（如 `mean_hr`, `rr_mean`, `rr_std` 等）

`analysis_tables/cohort_counts.parquet`（各分层人数，long format）
- `stratifier`, `level`, `n_rows`, `n_subjects`, `n_record_ids`, `pct_rows`

`analysis_tables/feature_summary.parquet`（均值/方差/分位数，long format）
- `feature_name`, `group_var`, `group_value`, `n`
- `mean`, `var`, `std`, `p25`, `p50`, `p75`, `min`, `max`, `missing_rate`

`analysis_tables/group_compare.parquet`（组间差异/检验/效应量）
- `feature_name`, `group_var`, `group_a`, `group_b`
- `n_a`, `n_b`, `mean_a`, `mean_b`, `diff_mean`
- `test_method`, `p_value`, `effect_name`, `effect_size`

`plots/`
- 至少 1 张图（`.png`/`.svg`）；建议附 `plots/plots_summary.json`（图文件清单与说明）。

`report.md`
- 必含章节：`Cohort Overview`、`ECG QC & Feature Summary`、`Group Comparison`、`Limitations`。

`run_metadata.json`
- 必需键：`pipeline_name`(=`report_pipeline`), `pipeline_version`, `run_id`, `created_at`(UTC ISO8601), `status`(`success`/`partial`/`failed`), `warnings`, `input_files`, `output_files`。
