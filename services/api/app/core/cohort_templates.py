from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.core.config import DATA_SCHEMA


@dataclass(frozen=True)
class CohortResult:
    sql: str
    params: dict[str, Any]


TemplateFn = Callable[[dict[str, Any]], CohortResult]


def _require(params: dict[str, Any], key: str) -> Any:
    if key not in params:
        raise ValueError(f"missing param: {key}")
    return params[key]


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _tbl(name: str) -> str:
    return f"{DATA_SCHEMA}.{name}"


def _as_int_list(raw: Any, key: str) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key} must be a non-empty list")
    out: list[int] = []
    for v in raw:
        out.append(int(v))
    return out


def _as_str_list(raw: Any, key: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key} must be a non-empty list")
    out: list[str] = []
    for v in raw:
        s = str(v).strip()
        if not s:
            raise ValueError(f"{key} contains empty string")
        out.append(s)
    return out


def _append_time_window_clauses(
    where_parts: list[str],
    *,
    qualified_col: str,
    start_key: str,
    end_key: str,
    params: dict[str, Any],
) -> dict[str, str]:
    out: dict[str, str] = {}
    if params.get(start_key):
        start = str(params[start_key]).strip()
        where_parts.append(f"{qualified_col} >= {_sql_quote(start)}")
        out[start_key] = start
    if params.get(end_key):
        end = str(params[end_key]).strip()
        where_parts.append(f"{qualified_col} < {_sql_quote(end)}")
        out[end_key] = end
    return out


def template_electrolyte_hyperkalemia(params: dict[str, Any]) -> CohortResult:
    """
    基于 labevents + d_labitems 构建高钾 cohort。
    首次满足阈值的 charttime 作为 index_time（按 subject_id, hadm_id 聚合）。
    """
    k_threshold = float(params.get("k_threshold", 5.5))
    itemids = _as_int_list(params.get("lab_itemids"), "lab_itemids") if "lab_itemids" in params else []
    label_keyword = str(params.get("label_keyword", "potassium")).strip().lower()
    if not label_keyword and not itemids:
        raise ValueError("label_keyword cannot be empty when lab_itemids is not provided")

    where_parts = [
        "le.valuenum IS NOT NULL",
        "le.charttime IS NOT NULL",
        f"le.valuenum >= {k_threshold}",
    ]
    normalized: dict[str, Any] = {"k_threshold": k_threshold}

    if itemids:
        itemid_sql = ", ".join(str(i) for i in itemids)
        where_parts.append(f"le.itemid IN ({itemid_sql})")
        normalized["lab_itemids"] = itemids
    else:
        like_value = f"%{label_keyword}%"
        where_parts.append(f"LOWER(dl.label) LIKE {_sql_quote(like_value)}")
        normalized["label_keyword"] = label_keyword

    normalized.update(
        _append_time_window_clauses(
            where_parts,
            qualified_col="le.charttime",
            start_key="charttime_start",
            end_key="charttime_end",
            params=params,
        )
    )

    sql = f"""
    SELECT
      le.subject_id AS subject_id,
      le.hadm_id AS hadm_id,
      MIN(le.charttime) AS index_time,
      'electrolyte_hyperkalemia'::text AS cohort_label
    FROM {_tbl("labevents")} le
    JOIN {_tbl("d_labitems")} dl
      ON dl.itemid = le.itemid
    WHERE {' AND '.join(where_parts)}
    GROUP BY le.subject_id, le.hadm_id
    ORDER BY MIN(le.charttime) ASC
    """
    return CohortResult(sql=sql.strip(), params=normalized)


def template_diagnosis_icd(params: dict[str, Any]) -> CohortResult:
    """
    基于 diagnoses_icd（可联 admissions）按 ICD code 前缀/列表选 cohort。
    index_time 使用 admittime。
    """
    icd_codes = _as_str_list(params.get("icd_codes"), "icd_codes") if "icd_codes" in params else []
    icd_prefixes = _as_str_list(params.get("icd_prefixes"), "icd_prefixes") if "icd_prefixes" in params else []
    if not icd_codes and not icd_prefixes:
        raise ValueError("one of icd_codes or icd_prefixes is required")

    code_clauses: list[str] = []
    normalized: dict[str, Any] = {}
    if icd_codes:
        code_sql = ", ".join(_sql_quote(c) for c in icd_codes)
        code_clauses.append(f"d.icd_code IN ({code_sql})")
        normalized["icd_codes"] = icd_codes
    if icd_prefixes:
        prefix_clauses = [f"d.icd_code LIKE {_sql_quote(p + '%')}" for p in icd_prefixes]
        code_clauses.append("(" + " OR ".join(prefix_clauses) + ")")
        normalized["icd_prefixes"] = icd_prefixes

    where_parts = ["(" + " OR ".join(code_clauses) + ")"]
    if "icd_version" in params and params.get("icd_version") is not None:
        icd_version = int(params["icd_version"])
        where_parts.append(f"d.icd_version = {icd_version}")
        normalized["icd_version"] = icd_version

    normalized.update(
        _append_time_window_clauses(
            where_parts,
            qualified_col="a.admittime",
            start_key="admittime_start",
            end_key="admittime_end",
            params=params,
        )
    )

    sql = f"""
    SELECT
      d.subject_id AS subject_id,
      d.hadm_id AS hadm_id,
      MIN(a.admittime) AS index_time,
      'diagnosis_icd'::text AS cohort_label
    FROM {_tbl("diagnoses_icd")} d
    JOIN {_tbl("admissions")} a
      ON a.subject_id = d.subject_id
     AND a.hadm_id = d.hadm_id
    WHERE {' AND '.join(where_parts)}
    GROUP BY d.subject_id, d.hadm_id
    ORDER BY MIN(a.admittime) ASC
    """
    return CohortResult(sql=sql.strip(), params=normalized)


def template_medication_exposure(params: dict[str, Any]) -> CohortResult:
    """
    按药物给药事件建 cohort。默认使用 prescriptions，也支持 pharmacy。
    index_time 取首次 starttime（按 subject_id, hadm_id 聚合）。
    pre/post window 先作为标准化参数返回，供下游特征抽取使用。
    """
    source = str(params.get("source", "prescriptions")).strip().lower()
    if source not in {"prescriptions", "pharmacy"}:
        raise ValueError("source must be one of: prescriptions, pharmacy")

    name_col = "drug" if source == "prescriptions" else "medication"
    alias = "m"

    drug_names = _as_str_list(params.get("drug_names"), "drug_names") if "drug_names" in params else []
    drug_keywords = _as_str_list(params.get("drug_keywords"), "drug_keywords") if "drug_keywords" in params else []
    if not drug_names and not drug_keywords:
        raise ValueError("one of drug_names or drug_keywords is required")

    drug_match_clauses: list[str] = []
    normalized: dict[str, Any] = {"source": source}
    if drug_names:
        name_sql = ", ".join(_sql_quote(v) for v in drug_names)
        drug_match_clauses.append(f"{alias}.{name_col} IN ({name_sql})")
        normalized["drug_names"] = drug_names
    if drug_keywords:
        kw_clauses = [f"LOWER({alias}.{name_col}) LIKE {_sql_quote('%' + kw.lower() + '%')}" for kw in drug_keywords]
        drug_match_clauses.append("(" + " OR ".join(kw_clauses) + ")")
        normalized["drug_keywords"] = [kw.lower() for kw in drug_keywords]

    where_parts = [
        f"{alias}.starttime IS NOT NULL",
        "(" + " OR ".join(drug_match_clauses) + ")",
    ]
    normalized.update(
        _append_time_window_clauses(
            where_parts,
            qualified_col=f"{alias}.starttime",
            start_key="starttime_start",
            end_key="starttime_end",
            params=params,
        )
    )

    if "pre_hours" in params and params.get("pre_hours") is not None:
        normalized["pre_hours"] = int(params["pre_hours"])
    if "post_hours" in params and params.get("post_hours") is not None:
        normalized["post_hours"] = int(params["post_hours"])

    sql = f"""
    SELECT
      {alias}.subject_id AS subject_id,
      {alias}.hadm_id AS hadm_id,
      MIN({alias}.starttime) AS index_time,
      'medication_exposure'::text AS cohort_label
    FROM {_tbl(source)} {alias}
    WHERE {' AND '.join(where_parts)}
    GROUP BY {alias}.subject_id, {alias}.hadm_id
    ORDER BY MIN({alias}.starttime) ASC
    """
    return CohortResult(sql=sql.strip(), params=normalized)


TEMPLATES: dict[str, TemplateFn] = {
    "electrolyte_hyperkalemia": template_electrolyte_hyperkalemia,
    "diagnosis_icd": template_diagnosis_icd,
    "medication_exposure": template_medication_exposure,
}
