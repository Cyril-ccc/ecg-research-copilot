# Drug Alias Map (Research Heuristic)

## Purpose
Map common natural-language drug mentions to canonical medication keywords used by cohort templates.

## Usage Rules
- For `medication_exposure`, prefer `drug_keywords` unless exact formulary names are known.
- Keep `source` explicit (`prescriptions` by default).
- Include pre/post windows (`pre_hours`, `post_hours`) in plan args.

## Alias mappings
- amiodarone
  - aliases: amiodarone, cordarone, 胺碘酮
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["amiodarone"]
- metoprolol
  - aliases: metoprolol, lopressor, 美托洛尔
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["metoprolol"]
- furosemide
  - aliases: furosemide, lasix, 呋塞米
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["furosemide"]
- potassium chloride
  - aliases: potassium chloride, kcl, 氯化钾
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["potassium chloride"]

- ascorbic acid
  - aliases: ascorbic acid, vitamin c, vit c, 维生素c, 维生素 c
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["ascorbic acid"]

- digoxin
  - aliases: digoxin, lanoxin, deslanoside, cedilanid, 西地兰, 地高辛
  - suggested template args:
    - source: prescriptions
    - drug_keywords: ["digoxin"]
## Safety Notes
- Avoid exposing patient-level medication records in answers.
- Treat these aliases as planning aids, not prescribing guidance.
