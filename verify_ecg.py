import sys
from pathlib import Path

# Add services/api to sys.path so we can import app
API_ROOT = Path(__file__).resolve().parent / "services" / "api"
sys.path.insert(0, str(API_ROOT))

import pandas as pd
from app.core.datasets.reader import read_ecg_record

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_OUT = PROJECT_ROOT / "storage" / "ecg_manifest.parquet"
DATA_DIR = PROJECT_ROOT / "data" / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"

df = pd.read_parquet(MANIFEST_OUT)
samples = df.sample(3)

print("--- Random 3 ECGs Verification ---")
for _, row in samples.iterrows():
    print(f"\nRecord ID: {row['record_id']}")
    print(f"Manifest Info ->  target fs: {row['fs']}, leads: {row['n_leads']}, samples: {row['n_samples']}")
    
    waveform, fs, lead_names, meta = read_ecg_record(DATA_DIR, row["path"], source=row["source"])
    
    print(f"Actual Data   ->  fs: {fs}, shape: {waveform.shape}, leads: {len(lead_names)}, names: {lead_names}")
    
    # Assert correctness
    assert fs == row['fs'], f"FS mismatch: {fs} != {row['fs']}"
    assert waveform.shape == (row['n_samples'], row['n_leads']), f"Shape mismatch: {waveform.shape}"
    assert len(lead_names) == row['n_leads'], f"Lead count mismatch: {len(lead_names)}"

print("\nAll shapes, frequencies, and lengths match perfectly! ✅")
