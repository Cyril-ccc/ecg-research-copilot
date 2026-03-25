# ECG Research Copilot

A local-first, auditable ECG research copilot that turns natural-language research questions into reproducible data workflows, artifacts, and grounded answers.

## What It Does

This project upgrades a traditional ECG analysis pipeline into an agent-driven research system:

1. The user asks a question in natural language.
2. A planner generates a structured `plan.json`.
3. A guarded tool executor runs whitelisted tools only.
4. Pipelines build cohorts, run ECG QC, extract features, and generate reports.
5. The final answer is written only from generated artifacts, not from LLM memory.

Typical outputs include:

- `cohort.parquet`
- `ecg_qc.parquet`
- `ecg_features.parquet`
- `analysis_tables/*.parquet`
- `plots/*.png`
- `report.md`
- `final_answer.md`
- `agent_trace.json`

## Why This Project

Most LLM-based research demos stop at “chat with data”. This repository takes a stricter approach:

- local-first execution
- artifact-grounded answers
- auditable tool calls
- schema-validated tool arguments
- prompt-injection filtering
- patient-level output restrictions

The goal is not a chatbot. The goal is a reproducible research copilot.

## Architecture

Core flow:

```text
Question
  -> Planner
  -> Tool Registry / Tool Executor
  -> Data Pipelines
  -> Artifacts
  -> Answer Writer
  -> Final Answer
```

Key components:

- `services/api/app/agent/planner.py`
  - generates strict JSON plans from the question and knowledge-base context
- `services/api/app/agent/tool_registry.py`
  - defines the only allowed tools and their schemas
- `services/api/app/agent/tool_executor.py`
  - validates inputs, enforces permissions, logs tool calls
- `services/api/app/agent/runner.py`
  - runs the full agent workflow and writes `agent_trace.json`
- `services/api/app/agent/answer_writer.py`
  - writes the final answer only from artifacts and summaries
- `pipelines/`
  - cohort building, ECG QC, feature extraction, analysis tables, plots, and reports

## Main Features

### 1. Planner + Tooling

The agent produces a structured execution plan instead of directly answering from free text.

### 2. Guarded Execution

Only registry-approved tools can run. Invalid tool names or invalid arguments are rejected before execution.

### 3. Artifact-Grounded Reporting

`final_answer.md` must cite numbers from generated artifacts such as `cohort_summary.json` or `analysis_tables/*.parquet` summaries.

### 4. Safety Constraints

The system blocks or downgrades unsafe requests such as:

- exporting patient-level identifiers
- ignoring security rules
- executing destructive instructions
- requesting oversized runs beyond configured limits

### 5. Evaluation and CI

The repository includes:

- cohort checks
- ECG/QC checks
- report checks
- agent smoke evaluation
- CI automation via GitHub Actions

## Tech Stack

- FastAPI
- PostgreSQL
- Redis
- Docker Compose
- Ollama (`qwen3:14b`, `qwen3-embedding:0.6b`)
- Pandas / Parquet-based artifacts
- pytest + GitHub Actions

## Repository Structure

```text
services/api/        FastAPI app, agent orchestration, routes, tests
pipelines/           ECG processing and report-generation pipelines
evals/               gold questions, agent tests, eval runners and checks
knowledge_base/      cohort templates, QC rules, feature defs, security policies
scripts/             data import, manifest build, mode switching, smoke helpers
config/              schema whitelist and configuration
```

## Quick Start

### 1. Start services

```powershell
docker compose up --build -d
```

### 2. Check API health

```powershell
curl http://127.0.0.1:8000/health
```

### 3. Open the UI

- Cohort UI: `http://127.0.0.1:8000/ui/`
- Agent UI: `http://127.0.0.1:8000/ui/agent`

## Example Questions

- `入院 6 小时 K>5.5 的患者，ECG mean_hr 与 RR std 的总体分布是什么？`
- `比较胺碘酮用药前24小时和后24小时ECG特征`
- `AF患者住院期间的ECG风险分层`

## Evaluation

Smoke evaluation:

```bash
make eval_smoke
make eval_agent_smoke
```

Useful files:

- `evals/gold_questions.yaml`
- `evals/agent_tests.yaml`
- `evals/check_cohort.py`
- `evals/check_ecg.py`
- `evals/check_report.py`

## Data Policy

This public repository does not contain:

- MIMIC-IV raw data
- MIMIC-IV-ECG waveform data
- generated artifact outputs
- local development storage

Demo/smoke workflows are designed to fetch public demo data at runtime instead of storing datasets in Git.

## Safety and Security

See `SECURITY.md` for the detailed threat model and restrictions.

Important safeguards include:

- tool whitelist
- Pydantic schema validation
- audit logging
- prompt-injection filtering for retrieved KB snippets
- no patient-level export in final answers
- small-cell suppression in outputs

## Current Limitations

This project is a research copilot, not a clinical decision system.

Known limitations:

- current ECG feature coverage is incomplete for strong arrhythmia risk stratification
- some disease/drug mappings still rely on curated alias maps and fallback templates
- final conclusions remain limited by cohort definition, missingness, confounding, and time-window design

## For Portfolio / Interview Use

This repository is best demonstrated as:

- a local-first medical research copilot
- an auditable agent system with tool and data guardrails
- a practical bridge between natural-language questions and reproducible biomedical analysis workflows

It should not be presented as a production clinical diagnosis system.
