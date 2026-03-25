# Disease to ICD Map (Research Heuristic)

## Purpose
Provide conservative disease-to-ICD prefixes for cohort template planning.
These mappings are planning hints only and must be reviewed for study validity.

## Usage Rules
- Prefer ICD prefixes over exact ICD codes for broad cohort entry.
- Add `icd_version` when your dataset mixes ICD-9 and ICD-10.
- Keep mappings auditable in run artifacts.

## Core mappings
- atrial fibrillation
  - keywords: AF, atrial fibrillation, 房颤, 心房颤动
  - icd_prefixes: ["I48", "42731"]
- stemi / acute myocardial infarction
  - keywords: STEMI, AMI, myocardial infarction, 心梗
  - icd_prefixes: ["I21", "410"]
- nstemi
  - keywords: NSTEMI
  - icd_prefixes: ["I21.4", "410.7"]
- heart failure
  - keywords: heart failure, CHF, 心衰
  - icd_prefixes: ["I50", "428"]
- aki
  - keywords: AKI, acute kidney injury, 急性肾损伤
  - icd_prefixes: ["N17", "584"]

## Safety Notes
- Do not infer diagnosis labels from free text alone.
- If uncertainty is high, ask for clarification or use explicit defaults.
