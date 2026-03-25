from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1. Create a run to get a run_id
response = client.post("/runs", json={"question": "test cohort summary", "params": {}})
print("Create run response:", response.status_code, response.json())
run_id = response.json().get("run_id")

# 2. Build cohort
payload = {
    "template_name": "electrolyte_hyperkalemia",
    "params": {
        "k_threshold": 6.0,
        "label_keyword": "potassium"
    },
    "run_id": run_id,
    "limit": 10
}
print(f"Building cohort for run_id: {run_id}")
resp2 = client.post("/tools/build_cohort", json=payload)
print("Build cohort response:", resp2.status_code)
if resp2.status_code != 200:
    print(resp2.text)

# 3. Check artifacts endpoint
print(f"Fetching artifacts for run_id: {run_id}")
resp3 = client.get(f"/runs/{run_id}/artifacts")
print("Artifacts response:", resp3.status_code)
print("Artifacts:", resp3.json())

# 4. Check summary content
import json
from pathlib import Path
from app.core.config import ARTIFACTS_DIR
summary_path = ARTIFACTS_DIR / run_id / "cohort_summary.json"
if summary_path.exists():
    print(f"Summary content: {json.loads(summary_path.read_text('utf-8'))}")
else:
    print("cohort_summary.json not found!")
