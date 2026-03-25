# Public Repo Checklist

Use this checklist before sharing the repository link in resumes, interviews, or public posts.

## Repository Safety

1. Confirm there is only one public commit:
   - `git rev-list --count HEAD`
2. Confirm no data directories are tracked:
   - `data/`
   - `storage/`
   - `services/api/storage/`
   - `eval_runs/`
3. Confirm no local env files are tracked:
   - `env.demo`
   - `env.full`
   - `.env`
4. Confirm no generated data files are tracked:
   - `.parquet`
   - `.csv`
   - `.hea`
   - `.dat`

## README Quality

1. README clearly explains:
   - project goal
   - architecture
   - safety model
   - limitations
2. README does not overclaim:
   - avoid calling it a clinical diagnosis system
   - avoid claiming validated medical performance
3. README includes:
   - quick start
   - example questions
   - evaluation entry points

## Demo Readiness

1. Docker services can start locally.
2. `http://127.0.0.1:8000/ui/agent` opens correctly.
3. At least two demo questions run end-to-end.
4. At least one malicious prompt is rejected correctly.
5. Final answers show traceable numeric sources.

## Interview Materials

1. Prepare one architecture diagram.
2. Prepare one short 3-5 minute demo video.
3. Prepare one longer technical walkthrough.
4. Prepare screenshots for:
   - Agent UI
   - plan.json
   - step logs
   - report / plots
   - final answer with cited numbers

## Final Positioning

Recommended positioning:

- local-first ECG research copilot
- auditable agent for biomedical analysis workflows
- artifact-grounded reporting system

Avoid positioning it as:

- autonomous clinical diagnosis AI
- medically validated risk prediction product
- patient-facing healthcare tool
