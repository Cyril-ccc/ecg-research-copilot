# ECG Research Copilot (Local-first)

## 1) 项目目标
ECG Research Copilot 是一个**本地运行（local-first）**的研究助手：把研究问题（自然语言）转成可复现的研究流程（runs），并产出可追踪的工件（artifacts），包括：
- 运行记录：每次 run 的 question / params / status / artifacts_path
- 工具调用：例如安全 SQL 执行壳（后续接 cohort builder / LLM 生成 SQL 也必须走笼子）
- 审计日志：谁在何时做了什么（audit_logs），便于合规与追踪

> 设计原则：数据不出域、可复现、可审计、可扩展。

---

## 2) 快速启动（3 条命令）
前置要求：安装 Docker Desktop（含 docker compose）。Windows 推荐使用 PowerShell。

在项目根目录执行：

1) 启动服务（Postgres + Redis + API + Worker）
```powershell
docker compose up --build -d
```

2) 等待约 10 秒后，验证 API 健康
```powershell
curl http://127.0.0.1:8000/health
```

3) 打开 Cohort Builder UI
浏览器访问：http://127.0.0.1:8000/ui/

---

## 3) 如何新增 Cohort 模板

所有模板都在 [`services/api/app/core/cohort_templates.py`](services/api/app/core/cohort_templates.py)。

### 步骤

**Step 1 - 实现模板函数**

函数签名为 `(params: dict[str, Any]) -> CohortResult`，返回一个包含 SQL 和标准化参数的 `CohortResult`：

```python
def template_my_cohort(params: dict[str, Any]) -> CohortResult:
    """
    用一句话描述这个 cohort 的含义。
    index_time 取 xxx。
    """
    # 1) 从 params 读取并校验输入
    my_val = str(params.get("my_param", "default")).strip()

    # 2) 构造 WHERE 子句（使用 _tbl() 添加 schema 前缀）
    where_parts = [f"t.some_col = '{my_val}'"]

    # 3) 返回 SQL + 归一化参数
    sql = f"""
    SELECT
      t.subject_id AS subject_id,
      t.hadm_id    AS hadm_id,
      MIN(t.some_time) AS index_time,
      'my_cohort'::text AS cohort_label
    FROM {_tbl("my_table")} t
    WHERE {' AND '.join(where_parts)}
    GROUP BY t.subject_id, t.hadm_id
    ORDER BY MIN(t.some_time) ASC
    """
    return CohortResult(sql=sql.strip(), params={"my_param": my_val})
```

> **规则**：
> - 所有列必须带表别名前缀（`t.col`，不能写裸列名）
> - 不能用 `SELECT *`
> - 只能查 [`config/schema_whitelist.yaml`](config/schema_whitelist.yaml) 中允许的表和字段
> - `index_time` 取该 cohort 的"首次事件时间"

**Step 2 - 注册到 TEMPLATES 字典**

在文件末尾的 `TEMPLATES` 字典中添加一行：

```python
TEMPLATES: dict[str, TemplateFn] = {
    "electrolyte_hyperkalemia": template_electrolyte_hyperkalemia,
    "diagnosis_icd":            template_diagnosis_icd,
    "medication_exposure":      template_medication_exposure,
    "my_cohort":                template_my_cohort,   # <- 新增
}
```

**Step 3 - 在白名单中允许新表（如需要）**

如果 SQL 用到了新表，先在 [`config/schema_whitelist.yaml`](config/schema_whitelist.yaml) 中声明：

```yaml
allow:
  public:
    my_table: [subject_id, hadm_id, some_col, some_time]
```

**Step 4 - 写测试**

在 `services/api/tests/test_cohort_smoke.py` 中参照现有测试，mock 假数据并验证 artifacts 输出。

**Step 5 - 调用**

```bash
curl -X POST http://127.0.0.1:8000/tools/build_cohort \
  -H "Content-Type: application/json" \
  -d '{"template_name": "my_cohort", "params": {"my_param": "value"}, "run_id": "<run_id>", "limit": 100}'
```

或直接在 UI（http://127.0.0.1:8000/ui/）选择模板后填参数运行。

---

## 4) 安全设计

请参见 [SECURITY.md](SECURITY.md)。

---

## 5) Cohort 输出如何映射到 ECG `record_id`（约定 / TODO）

当前仓库里，`build_cohort` 产生的是临床 cohort（例如 `subject_id` / `hadm_id` / `index_time`），而 ECG 流水线（QC/特征）需要 `record_id` 列表。建议先采用下面的约定：

1) 输入要求（约定）
- `cohort.parquet` 至少包含：`subject_id`, `hadm_id`, `index_time`

2) 映射目标
- 输出 `ecg_record_ids.txt`（每行一个 `record_id`），用于 `POST /tools/extract_ecg_features`

3) 规则建议（TODO，待固化脚本）
- 数据来源：`storage/ecg_manifest.parquet`（含 `record_id`、record path、来源）
- 连接键：优先 `subject_id + hadm_id`
- 时间窗口：`ecg_time` 落在 `index_time ± X 小时`（例如 `-24h ~ +24h`）
- 多条 ECG 冲突时：取距离 `index_time` 最近的一条；若同距，取最早采集
- 结果去重：`record_id` 去重并保序

4) 后续落地
- 新增一个独立步骤/脚本：`cohort_to_ecg_ids.py`，输入 `cohort.parquet`，输出 `ecg_record_ids.txt`
- 再把输出直接喂给 `/tools/extract_ecg_features`

---

## 6) 如何生成 Report（Demo 全链路）

在项目根目录执行：

```bash
make demo_report
```

默认会跑 10 条 ECG（可改 `DEMO_SAMPLE_N`）并把产物写到：

- `storage/artifacts/demo-report/report.md`
- `storage/artifacts/demo-report/run_metadata.json`

可自定义运行参数：

```bash
make demo_report DEMO_RUN_ID=demo-report-8 DEMO_SAMPLE_N=8
```

该命令会按顺序执行：`build_cohort -> ecg_qc -> ecg_features -> assemble_analysis_dataset -> build_analysis_tables -> build_report_plots -> generate_report`。

---

## 7) Gold Eval 工作流

### 7.1 如何新增 gold question

gold 配置文件位置：`evals/gold_questions.yaml`。每个条目是一个任务，建议结构如下：

```yaml
- id: G013
  category: diagnosis
  name: afib_icd_i48_window24
  cohort_template: diagnosis_icd
  params:
    icd_prefixes:
      - I48
    window_hours: 24
  expectations:
    cohort_subjects_min: 50
    cohort_subjects_max: 80000
    required_artifacts:
      - cohort.parquet
      - ecg_qc.parquet
      - ecg_features.parquet
      - report.md
    qc:
      pass_rate_min: 0.60
    features:
      mean_hr_min: 30
      mean_hr_max: 180
    report:
      must_have_sections:
        - Cohort definition
        - Data & QC
        - Results
        - Limitations
      must_mention:
        - missing
        - confounding
```

新增时建议按下面检查：

1. `id` 全局唯一（推荐 `Gxxx` 递增）。
2. `cohort_template` 必须是已注册模板（如 `electrolyte_hyperkalemia`、`diagnosis_icd`、`medication_exposure`）。
3. `expectations` 阈值要与任务难度匹配（避免过松或过严）。
4. 先跑 smoke，再跑 full，确认该 gold 稳定。

### 7.2 smoke / full 运行方式

先确保评估依赖服务已启动：

```powershell
docker compose up -d db redis api worker
```

smoke（快速回归，默认前 3 条、每条最多 50 条 ECG）：

```bash
make eval_smoke
```

可调 smoke 范围：

```bash
make eval_smoke EVAL_SMOKE_N=5 EVAL_SMOKE_MAX_RECORDS=20
```

full（完整执行，默认全部 gold、每条不限制 ECG 数）：

```bash
make eval_full
```

可限制 full 数量（例如只跑前 20 条）：

```bash
make eval_full EVAL_FULL_N=20
```

建议在 eval 后执行检查与汇总：

```bash
make eval_check_cohort
make eval_check_ecg
make eval_check_report
make eval_summary
```

### 7.3 如何更新 baseline（避免随意覆盖）

`evals/baselines/*.json` 是漂移对比基线，**不要直接覆盖**。推荐流程：

1. 先跑 `eval_full` + 各检查，确认通过。
2. 把新 baseline 写到候选目录，而不是正式目录：

```bash
make eval_write_baseline EVAL_BASELINE_DIR=/workspace/evals/baselines_candidates/2026-03-06
```

3. 用候选 baseline 再跑一次 ECG 检查：

```bash
make eval_check_ecg EVAL_BASELINE_DIR=/workspace/evals/baselines_candidates/2026-03-06
```

4. 人工审查差异（必须看变更和原因）：

```bash
git diff -- evals/baselines evals/baselines_candidates/2026-03-06
```

5. 仅在确认“分布变化是预期变化”后，再把候选文件逐个提升到正式目录并提交 PR。

baseline 更新 PR 建议至少包含：
- baseline 变更文件（`evals/baselines/*.json`）
- 对应 `eval_runs/summary_full.json` 或关键检查结果
- 变更原因说明（例如数据版本变化、QC 规则变化、特征算法升级）


### 7.4 Agent Eval（Planner/Guardrail）

新增了面向 Agent 链路的评测集：`evals/agent_tests.yaml`（正常/含糊/恶意/超大请求）。

新增药物支持的详细步骤请看：[`docs/ADDING_DRUG.md`](docs/ADDING_DRUG.md)。

快速回归（默认前 5 条）：

```bash
make eval_agent_smoke
```

可调 smoke 规模：

```bash
make eval_agent_smoke AGENT_EVAL_SMOKE_N=5 AGENT_EVAL_SMOKE_MAX_RECORDS=20
```

生成 agent 评测摘要：

```bash
make eval_agent_summary
```

输出文件默认在：
- `eval_runs/agent_summary_smoke.json`
- `eval_agent_summary.md`

### 7.5 故意破坏开关演示（CI 红灯）

用于演示“放宽白名单/输出限制”时，agent eval 会立刻失败并定位到具体 `ATxxx`。

执行：

```bash
make eval_agent_fault_demo
```

默认会跑两种故障模式：
- `whitelist_relaxed`：模拟白名单被放宽
- `output_leak`：模拟最终输出出现禁字段泄露

可指定目标测试（默认 `AT001`）：

```bash
make eval_agent_fault_demo AGENT_FAULT_TARGET_TEST_ID=AT001
```

结果文件：
- `eval_runs/fault_demo/agent_summary_fault_whitelist_relaxed.json`
- `eval_runs/fault_demo/agent_summary_fault_output_leak.json`
- `eval_runs/fault_demo/fault_demo_result.json`

如果你要在分支里直接触发“CI 红灯”验证（命令返回非 0）：

```bash
make eval_agent_smoke_break_whitelist
make eval_agent_smoke_break_output
```

这两条会在 `eval_summary` 里定位到具体失败用例（默认 `AT001`，可用 `AGENT_FAULT_TARGET_TEST_ID` 覆盖）。


### 7.6 无数据环境 / CI 说明

- `pytest` 现在默认不依赖本地 Postgres；测试环境会通过 `APP_INIT_DB_ON_STARTUP=0` 跳过启动时建表，因此在没有数据库、没有 demo 数据时也能跑单元测试。
- `eval_smoke` / `eval_agent_smoke` 仍然需要公开 demo 数据，因为它们会真实导入临床表和 ECG `record_list`。
- 不要把 demo 数据直接提交进 Git 仓库。仓库已通过 `.gitignore` 忽略 `/data/`；CI 会在运行前自动执行 `scripts/fetch_demo_data.py` 从 PhysioNet 下载公开 demo 数据。

本地如果要跑 smoke eval，先执行：

```powershell
python .\scripts\fetch_demo_data.py --data-dir .\data
```

然后再启动服务并导入 demo 数据：

```powershell
docker compose up -d db redis api worker
```

```powershell
docker compose run --rm -T -v "${PWD}:/workspace" -w /workspace/services/api api uv run python /workspace/scripts/import_data.py --database-url postgresql+psycopg://ecg:ecg@db:5432/ecg --clinical-root /workspace/data/mimic-iv-clinical-database-demo-2.2 --ecg-root /workspace/data/mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1 --schema mimiciv
```

下载脚本支持只下其中一类数据：

```powershell
python .\scripts\fetch_demo_data.py --dataset clinical
python .\scripts\fetch_demo_data.py --dataset ecg
```
## 8) 从 Demo 升级到 Full MIMIC-IV（实操步骤）

> 说明：仓库已改成可参数化导入与运行，但我无法访问你本地的 full MIMIC-IV 文件；你按下面步骤在本机执行即可。

### 8.1 你需要改/确认的地方

1. `docker-compose.yml` 已支持这 3 个环境变量（无需再改硬编码路径）：
- `MIMIC_DATA_HOST_DIR`：宿主机数据目录，挂载到容器 `/data`
- `ECG_DATA_DIR`：容器内 ECG 数据目录（例如 `/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0`）
- `ECG_MANIFEST_PATH`：容器内全局 manifest 输出路径（默认 `/storage/ecg_manifest.parquet`）

2. 新导入脚本：`scripts/import_data.py`
- 支持 chunk 导入、可选 ICU、可选索引/ANALYZE、可选表子集

3. 新 manifest 脚本：`scripts/build_ecg_manifest.py`
- 从 `mimiciv.record_list` 读索引
- 批量读取 WFDB header
- 生成包含 `ecg_time` 的 `storage/ecg_manifest.parquet`

### 8.2 数据目录准备

确保你的数据结构至少满足：

- 临床数据：
  - `<MIMIC_DATA_HOST_DIR>/mimic-iv-2.2/hosp/*.csv.gz`
  - `<MIMIC_DATA_HOST_DIR>/mimic-iv-2.2/icu/*.csv.gz`（可选）
- ECG 数据：
  - `<MIMIC_DATA_HOST_DIR>/mimic-iv-ecg-1.0/record_list.csv`
  - `<MIMIC_DATA_HOST_DIR>/mimic-iv-ecg-1.0/files/.../*.hea and *.dat`

### 8.3 配置 `.env`

复制一份配置模板：

```bash
cp .env.example .env
```

然后按你的实际路径改：

```env
MIMIC_DATA_HOST_DIR=/mnt/d/mimic-data
ECG_DATA_DIR=/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0
ECG_MANIFEST_PATH=/storage/ecg_manifest_full.parquet
DATA_SCHEMA=mimiciv_full
```

### 8.4 启动数据库与缓存

```bash
docker compose up -d db redis
```

### 8.5 导入 MIMIC-IV 临床 + ECG 索引

```bash
docker compose run --rm -T -v "${PWD}:/workspace" -w /workspace api \
  /app/.venv/bin/python scripts/import_data.py \
  --database-url postgresql+psycopg://ecg:ecg@db:5432/ecg \
  --clinical-root /data/mimic-iv-3.1 \
  --ecg-root /data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
  --schema mimiciv_full \
  --chunksize 200000 \
  --if-exists replace \
  --create-indexes \
  --analyze
```

如果你暂时不导 ICU，可保持默认（不加 `--include-icu`）。

### 8.6 重建全局 ECG Manifest（关键）

```bash
docker compose run --rm -T -v "${PWD}:/workspace" -w /workspace api \
  /app/.venv/bin/python scripts/build_ecg_manifest.py \
  --database-url postgresql://ecg:ecg@db:5432/ecg \
  --schema mimiciv_full \
  --record-list-table record_list \
  --data-dir /data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
  --output-path /workspace/storage/ecg_manifest_full.parquet \
  --batch-size 2000 \
  --log-every 10000
```

这一步会生成：
- `storage/ecg_manifest.parquet`
- `storage/ecg_manifest.summary.json`

### 8.7 启动 API + Worker

```bash
docker compose up -d api worker
```

### 8.8 验证是否切换成功

先做 3 个快速检查：

1. 检查 manifest 是否有 `ecg_time` 列（药物前后窗分析必须）
2. 跑 1 条药物前后问题（如氯化钾 24h）
3. 检查 run 产物 `run_metadata.json`：
- `compare_by` 应为 `window_group`
- `group_compare_rows` 应大于 0（在有配对样本时）

建议命令：

```bash
docker compose run --rm -T -v "${PWD}:/workspace" -w /workspace api \
  /app/.venv/bin/python -c "import pandas as pd; df=pd.read_parquet('/workspace/storage/ecg_manifest_full.parquet'); print(df.columns.tolist()); print('rows=',len(df)); print('ecg_time_non_null=',int(df['ecg_time'].notna().sum()) if 'ecg_time' in df.columns else -1)"
```

### 8.9 常见问题

- `failed to resolve host 'postgres'`：说明容器网络里服务名不匹配，compose 内统一用 `db`，不要写 `localhost`。
- `no manifest rows written`：通常是 `record_list.path` 对应的 `.hea` 文件路径不对，先检查 `MIMIC_DATA_HOST_DIR` 和 `ECG_DATA_DIR`。
- `group_compare_rows=0`：先看是否配对样本太少（pre/post 同时存在比例低），其次看 `ecg_time` 是否缺失。


