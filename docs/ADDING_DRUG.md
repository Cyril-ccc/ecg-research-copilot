# 新增药物到 Agent（Medication Exposure）操作手册

> 目标：让自然语言里的新药名，稳定路由到 `medication_exposure`，并生成正确的前后窗分析结果。

## 1. 适用范围

当你希望支持一个新药（例如“地尔硫卓”“阿托伐他汀”）时，按本文操作。

成功标准：
- 问题可以路由到 `medication_exposure`。
- `build_cohort` 的 `params.drug_keywords` 是你定义的 canonical 名。
- 不会误路由成 `diagnosis_icd` 或其它模板。

---

## 2. 先确认数据库里是否有这个药

先确认真实表里药名写法，避免只改别名但数据里根本匹配不到。

重点列：
- `mimiciv_full.prescriptions.drug`
- `mimiciv_full.pharmacy.medication`

示例 SQL（按你的 schema 改）：

```sql
SELECT LOWER(drug) AS name, COUNT(*)
FROM mimiciv_full.prescriptions
WHERE drug IS NOT NULL
GROUP BY LOWER(drug)
ORDER BY COUNT(*) DESC
LIMIT 100;
```

```sql
SELECT LOWER(medication) AS name, COUNT(*)
FROM mimiciv_full.pharmacy
WHERE medication IS NOT NULL
GROUP BY LOWER(medication)
ORDER BY COUNT(*) DESC
LIMIT 100;
```

建议先定一个 canonical 关键词（例如 `diltiazem`），再把常见别名映射到它。

---

## 3. 必改文件（最小闭环）

### Step A: Planner 别名映射

文件：`services/api/app/agent/planner.py`

位置：`MEDICATION_ALIAS_CANDIDATES`

做法：
- 增加 canonical 键。
- 加中英文别名、缩写、常见拼写。

示例：

```python
"diltiazem": ["diltiazem", "cardizem", "地尔硫卓"],
```

---

### Step B: Runner 规范化映射

文件：`services/api/app/agent/runner.py`

位置：`DRUG_KEYWORD_CANONICAL_MAP`

做法：
- 把所有别名归一到同一个 canonical。
- 保证工具执行阶段不会出现别名漂移。

示例：

```python
"地尔硫卓": "diltiazem",
"cardizem": "diltiazem",
"diltiazem": "diltiazem",
```

---

### Step C: 更新知识库药物映射文档

文件：`knowledge_base/drug_alias_map.md`

做法：
- 新增药物条目（aliases + suggested `drug_keywords`）。

示例：

```md
- diltiazem
  - aliases: diltiazem, cardizem, 地尔硫卓
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["diltiazem"]
```

---

### Step D: （推荐）更新检索优先规则

文件：`services/api/app/agent/knowledge_base.py`

位置：`DOC_PRIORITY_RULES`

做法：
- 在药物相关正则里加新药关键词，提升 RAG 命中 `drug_alias_map.md` 的概率。

---

### Step E: 加回归测试

文件：`evals/agent_tests.yaml`

做法：
- 增加 1 条新药问题。
- 增加期望：
  - `template_name: medication_exposure`
  - `drug_keyword: <your_canonical_name>`

示例：

```yaml
- id: AT0XX
  category: normal
  name: medication_pre_post_diltiazem
  question: 比较地尔硫卓用药前24小时和后24小时的ECG特征变化。
  expected:
    status: SUCCEEDED
    template_name: medication_exposure
    drug_keyword: diltiazem
```

---

## 4. 让改动生效

### 4.1 重建 API/Worker

```powershell
docker compose up -d --build api worker
```

### 4.2 重建知识库索引（如果你改了 knowledge_base）

> 你是 Docker 挂载执行，建议用 `/app/.venv/bin/python`，不要用 `uv run`（避免挂载后 `.venv` 识别问题）。

```powershell
docker compose run --rm -T `
  -e PYTHONPATH=/workspace/services/api `
  -v "${PWD}:/workspace" `
  -w /workspace/services/api `
  api /app/.venv/bin/python /workspace/scripts/index_knowledge_base.py `
  --kb-dir /workspace/knowledge_base `
  --version v1 `
  --replace-version `
  --ollama-base-url http://host.docker.internal:11434 `
  --embedding-model qwen3-embedding:0.6b
```

---

## 5. 验证清单

1. 在 Agent UI 提问新药问题。
2. 打开 `plan.json`：
   - `build_cohort.args.template_name == medication_exposure`
3. 看 `agent_trace.json` 的 `validated_args`：
   - `drug_keywords` 是否是 canonical 名。
4. 看产物：
   - `cohort.parquet`
   - `ecg_qc.parquet`
   - `ecg_features.parquet`
   - `report.md`

---

## 6. 常见问题

### Q1: 还是路由成别的药（如默认胺碘酮）

通常是 `planner.py` 没加别名，或改了代码但容器没重建。

### Q2: full 模式切换后报 `No valid records found in manifest for given record_ids`

通常是旧 `run_id` 复用了旧缓存。处理方式：
- 新建 run（不要复用旧 run_id）。
- 确认 `.env` 已切到 full 且 `ECG_MANIFEST_PATH` 正确。

### Q3: 药物 cohort 为 0

先回到第 2 步，确认真实表里药名字段到底是什么写法，再补 aliases 或改 canonical。

---

## 7. 建议的提交说明模板

```text
feat(agent): add medication alias mapping for <drug>

- planner: add <drug> aliases in MEDICATION_ALIAS_CANDIDATES
- runner: canonicalize aliases to <drug>
- kb: update knowledge_base/drug_alias_map.md
- eval: add AT0XX routing assertion for <drug>
```
