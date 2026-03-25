"""
SQL safety unit tests — no DB connection required.
These tests exercise validate_and_rewrite_sql directly.
"""
import pytest

from app.core.sql_safety import SqlPolicy, validate_and_rewrite_sql

# ── shared minimal whitelist ──────────────────────────────────────────────────
WL: dict[str, dict[str, set[str]]] = {
    "public": {
        "labevents": {"subject_id", "hadm_id", "itemid", "charttime", "valuenum"},
        "d_labitems": {"itemid", "label"},
    }
}

POLICY = SqlPolicy(
    max_limit=100,
    allow_schema="public",
    whitelist=WL,
    require_qualified_columns=True,
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _assert_rejected(sql: str, error_fragment: str):
    """SQL must raise ValueError whose message contains error_fragment."""
    with pytest.raises(ValueError, match=error_fragment):
        validate_and_rewrite_sql(sql, POLICY)


# ── negative tests (malicious / disallowed SQL) ───────────────────────────────

def test_drop_table_rejected():
    _assert_rejected(
        "DROP TABLE public.labevents",
        "forbidden keyword",
    )


def test_delete_rejected():
    _assert_rejected(
        "DELETE FROM public.labevents le WHERE le.subject_id = 1",
        "forbidden keyword",
    )


def test_semicolon_injection_rejected():
    _assert_rejected(
        "SELECT le.subject_id FROM public.labevents le; DROP TABLE x",
        "semicolon",
    )


def test_star_select_rejected():
    _assert_rejected(
        "SELECT le.* FROM public.labevents le",
        r"star",
    )


def test_unqualified_column_rejected():
    _assert_rejected(
        "SELECT subject_id FROM public.labevents le",
        "unqualified column",
    )


def test_table_not_in_whitelist_rejected():
    _assert_rejected(
        "SELECT s.secret FROM public.secret_table s",
        "table not allowed",
    )


def test_column_not_in_whitelist_rejected():
    _assert_rejected(
        "SELECT le.password FROM public.labevents le",
        "column not allowed",
    )


def test_create_table_rejected():
    _assert_rejected(
        "CREATE TABLE public.labevents (id int)",
        "forbidden keyword",
    )


# ── positive tests ────────────────────────────────────────────────────────────

def test_valid_select_passes_and_limit_injected():
    sql = (
        "SELECT le.subject_id, le.valuenum "
        "FROM public.labevents le "
        "WHERE le.valuenum >= 5.5"
    )
    out, meta = validate_and_rewrite_sql(sql, POLICY)
    assert "LIMIT" in out.upper()
    assert "public.labevents" in meta["tables_used"]


def test_limit_capped_at_policy_max():
    sql = (
        "SELECT le.subject_id FROM public.labevents le LIMIT 99999"
    )
    out, _ = validate_and_rewrite_sql(sql, POLICY)
    # Rewritten SQL should cap limit at policy.max_limit (100)
    assert "100" in out


def test_join_valid_tables_passes():
    sql = (
        "SELECT le.subject_id, le.valuenum, dl.label "
        "FROM public.labevents le "
        "JOIN public.d_labitems dl ON dl.itemid = le.itemid "
        "WHERE le.valuenum >= 5.5"
    )
    out, meta = validate_and_rewrite_sql(sql, POLICY)
    assert "LIMIT" in out.upper()
    assert len(meta["tables_used"]) == 2
