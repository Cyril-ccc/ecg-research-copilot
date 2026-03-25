DEMO_RUN_ID ?= demo-report
DEMO_SAMPLE_N ?= 10
EVAL_SMOKE_N ?= 3
EVAL_SMOKE_MAX_RECORDS ?= 50
EVAL_FULL_N ?= 0
EVAL_FULL_MAX_RECORDS ?= 0
EVAL_CHECK_OUTPUT ?= /workspace/eval_results.json
EVAL_CHECK_ECG_OUTPUT ?= /workspace/eval_ecg_results.json
EVAL_CHECK_REPORT_OUTPUT ?= /workspace/eval_report_results.json
EVAL_BASELINE_DIR ?= /workspace/evals/baselines
EVAL_SUMMARY_JSON ?= eval_runs/summary_smoke.json
EVAL_SUMMARY_MD ?= eval_summary.md
AGENT_EVAL_SMOKE_N ?= 5
AGENT_EVAL_SMOKE_MAX_RECORDS ?= 20
AGENT_EVAL_FULL_N ?= 0
AGENT_EVAL_FULL_MAX_RECORDS ?= 2000
AGENT_EVAL_SUMMARY_JSON ?= eval_runs/agent_summary_smoke.json
AGENT_EVAL_SUMMARY_MD ?= eval_agent_summary.md
AGENT_EVAL_FAULT_MODE ?= none
AGENT_EVAL_FAULT_TARGET_TEST_ID ?= AT001
AGENT_FAULT_TARGET_TEST_ID ?= AT001
AGENT_FAULT_MODES ?= whitelist_relaxed,output_leak
AGENT_FAULT_OUTPUT_DIR ?= /workspace/eval_runs/fault_demo
KB_VERSION ?= v1
KB_QUERY ?= QTc 是怎么算的？
KB_TOP_K ?= 5
KB_DOC_TYPES ?= template,qc,feature,report,security
KB_OLLAMA_BASE_URL ?= http://host.docker.internal:11434
KB_EMBEDDING_MODEL ?= qwen3-embedding:0.6b
DOCKER_API_RUN = docker compose run --rm -T -v "$(PWD)":/workspace -w /workspace/services/api api
DOCKER_API_RUN_ISOLATED = docker compose run --rm -T -e UV_PROJECT_ENVIRONMENT=/tmp/uv-venv -e PYTHONPATH=/workspace/services/api -v "$(PWD)":/workspace -w /workspace/services/api api
DOCKER_API_PY = docker compose run --rm -T -e PYTHONPATH=/workspace/services/api -v "$(PWD)":/workspace -w /workspace/services/api api /app/.venv/bin/python

.PHONY: demo_report eval_smoke eval_full eval_agent_smoke eval_agent_full eval_agent_fault_demo eval_agent_smoke_break_whitelist eval_agent_smoke_break_output eval_check_cohort eval_check_ecg eval_write_baseline eval_check_report eval_summary eval_agent_summary kb_index kb_query

demo_report:
	$(DOCKER_API_RUN) uv run python /workspace/pipelines/demo_report.py --run-id $(DEMO_RUN_ID) --sample-n $(DEMO_SAMPLE_N)

eval_smoke:
	$(DOCKER_API_RUN) uv run python /workspace/evals/runner.py --mode smoke --smoke-n $(EVAL_SMOKE_N) --smoke-max-records $(EVAL_SMOKE_MAX_RECORDS)

eval_full:
	$(DOCKER_API_RUN) uv run python /workspace/evals/runner.py --mode full --full-n $(EVAL_FULL_N) --full-max-records $(EVAL_FULL_MAX_RECORDS)

eval_agent_smoke:
	$(DOCKER_API_PY) /workspace/evals/agent_runner.py --mode smoke --smoke-n $(AGENT_EVAL_SMOKE_N) --smoke-max-records $(AGENT_EVAL_SMOKE_MAX_RECORDS) --fault-mode $(AGENT_EVAL_FAULT_MODE) --fault-target-test-id $(AGENT_EVAL_FAULT_TARGET_TEST_ID) --output /workspace/$(AGENT_EVAL_SUMMARY_JSON)

eval_agent_full:
	$(DOCKER_API_PY) /workspace/evals/agent_runner.py --mode full --full-n $(AGENT_EVAL_FULL_N) --full-max-records $(AGENT_EVAL_FULL_MAX_RECORDS) --fault-mode $(AGENT_EVAL_FAULT_MODE) --fault-target-test-id $(AGENT_EVAL_FAULT_TARGET_TEST_ID) --output /workspace/eval_runs/agent_summary_full.json

eval_agent_fault_demo:
	$(DOCKER_API_PY) /workspace/evals/agent_fault_demo.py --modes "$(AGENT_FAULT_MODES)" --target-test-id $(AGENT_FAULT_TARGET_TEST_ID) --smoke-n $(AGENT_EVAL_SMOKE_N) --smoke-max-records $(AGENT_EVAL_SMOKE_MAX_RECORDS) --output-dir $(AGENT_FAULT_OUTPUT_DIR)

eval_agent_smoke_break_whitelist:
	$(DOCKER_API_PY) /workspace/evals/agent_runner.py --mode smoke --smoke-n $(AGENT_EVAL_SMOKE_N) --smoke-max-records $(AGENT_EVAL_SMOKE_MAX_RECORDS) --fault-mode whitelist_relaxed --fault-target-test-id $(AGENT_FAULT_TARGET_TEST_ID) --output /workspace/$(AGENT_EVAL_SUMMARY_JSON)

eval_agent_smoke_break_output:
	$(DOCKER_API_PY) /workspace/evals/agent_runner.py --mode smoke --smoke-n $(AGENT_EVAL_SMOKE_N) --smoke-max-records $(AGENT_EVAL_SMOKE_MAX_RECORDS) --fault-mode output_leak --fault-target-test-id $(AGENT_FAULT_TARGET_TEST_ID) --output /workspace/$(AGENT_EVAL_SUMMARY_JSON)

eval_check_cohort:
	$(DOCKER_API_RUN) uv run python /workspace/evals/check_cohort.py --gold /workspace/evals/gold_questions.yaml --eval-runs-root /workspace/eval_runs --artifacts-root /workspace/storage/artifacts --output $(EVAL_CHECK_OUTPUT)

eval_check_ecg:
	$(DOCKER_API_RUN) uv run python /workspace/evals/check_ecg.py --gold /workspace/evals/gold_questions.yaml --eval-runs-root /workspace/eval_runs --artifacts-root /workspace/storage/artifacts --baseline-dir $(EVAL_BASELINE_DIR) --output $(EVAL_CHECK_ECG_OUTPUT)

eval_write_baseline:
	$(DOCKER_API_RUN) uv run python /workspace/evals/check_ecg.py --gold /workspace/evals/gold_questions.yaml --eval-runs-root /workspace/eval_runs --artifacts-root /workspace/storage/artifacts --baseline-dir $(EVAL_BASELINE_DIR) --write-baseline --output $(EVAL_CHECK_ECG_OUTPUT)

eval_check_report:
	$(DOCKER_API_RUN) uv run python /workspace/evals/check_report.py --gold /workspace/evals/gold_questions.yaml --eval-runs-root /workspace/eval_runs --artifacts-root /workspace/storage/artifacts --output $(EVAL_CHECK_REPORT_OUTPUT)

eval_summary:
	python evals/write_eval_summary.py --summary-json $(EVAL_SUMMARY_JSON) --output $(EVAL_SUMMARY_MD)

eval_agent_summary:
	python evals/write_eval_summary.py --summary-json $(AGENT_EVAL_SUMMARY_JSON) --output $(AGENT_EVAL_SUMMARY_MD)

kb_index:
	$(DOCKER_API_RUN_ISOLATED) uv run python /workspace/scripts/index_knowledge_base.py --kb-dir /workspace/knowledge_base --version $(KB_VERSION) --replace-version --ollama-base-url $(KB_OLLAMA_BASE_URL) --embedding-model $(KB_EMBEDDING_MODEL)

kb_query:
	$(DOCKER_API_RUN_ISOLATED) uv run python /workspace/scripts/query_knowledge_base.py --query "$(KB_QUERY)" --top-k $(KB_TOP_K) --doc-types "$(KB_DOC_TYPES)" --version $(KB_VERSION) --ollama-base-url $(KB_OLLAMA_BASE_URL) --embedding-model $(KB_EMBEDDING_MODEL)




