import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

FORBIDDEN_RAW = re.compile(
    r"\b(copy|create|drop|insert|update|delete|alter|grant|revoke|truncate|vacuum|analyze)\b",
    re.IGNORECASE,
)

MAX_LIMIT_DEFAULT = 5000


@dataclass
class SqlPolicy:
    max_limit: int = MAX_LIMIT_DEFAULT
    allow_schema: str = "public"
    # whitelist: {"public": {"table": {"col1","col2"}}}
    whitelist: dict[str, dict[str, set[str]]] | None = None
    require_qualified_columns: bool = True  # MVP：要求列都带表/别名前缀（更好做字段白名单）


def _build_alias_map(tree: exp.Expression) -> dict[str, str]:
    """
    把 FROM/JOIN 里的表别名映射回真实表名。
    例如 FROM patients p -> {"p": "patients", "patients": "patients"}
    """
    alias_map: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        table_name = t.name
        alias = t.alias
        if alias:
            alias_map[str(alias)] = table_name
        alias_map[table_name] = table_name
    return alias_map


def _get_table_schema(t: exp.Table, default_schema: str) -> tuple[str, str]:
    schema = t.db or default_schema
    return str(schema), t.name


def validate_and_rewrite_sql(sql: str, policy: SqlPolicy) -> tuple[str, dict[str, Any]]:
    s = sql.strip()
    if not s:
        raise ValueError("empty sql")

    if ";" in s:
        raise ValueError("semicolon is not allowed")

    if FORBIDDEN_RAW.search(s):
        raise ValueError("forbidden keyword detected")

    # 解析（只允许单条语句）
    parsed = sqlglot.parse(s, read="postgres")
    if len(parsed) != 1:
        raise ValueError("only one statement is allowed")
    tree = parsed[0]

    # 禁止任何非 SELECT 类节点（多一道防线）
    forbidden_nodes = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Copy, exp.Command)
    if any(True for _ in tree.find_all(forbidden_nodes)):
        raise ValueError("only SELECT queries are allowed")

    # 顶层必须是 query（SELECT/UNION/CTE->SELECT 等）
    if not isinstance(tree, exp.Expression):
        raise ValueError("invalid sql")

    # whitelist 校验：表
    wl = policy.whitelist or {}
    alias_map = _build_alias_map(tree)

    tables_used: list[tuple[str, str]] = []
    for t in tree.find_all(exp.Table):
        schema, table = _get_table_schema(t, policy.allow_schema)
        tables_used.append((schema, table))
        if schema not in wl or table not in wl[schema]:
            raise ValueError(f"table not allowed: {schema}.{table}")

    # whitelist 校验：字段
    # MVP 策略：要求列都带表/别名前缀，且不允许 *
    for col in tree.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            raise ValueError("star (*) is not allowed; select explicit columns")

        col_name = col.name
        qualifier = col.table

        if policy.require_qualified_columns and not qualifier:
            raise ValueError(f"unqualified column not allowed: {col_name}")

        table_ref = alias_map.get(str(qualifier), None) if qualifier else None
        if not table_ref:
            raise ValueError(f"unknown column qualifier: {qualifier}")

        # 默认 schema（MVP：只允许一个 schema）
        schema = policy.allow_schema
        allowed_cols = wl.get(schema, {}).get(table_ref, set())
        if col_name not in allowed_cols:
            raise ValueError(f"column not allowed: {schema}.{table_ref}.{col_name}")

    # 强制 LIMIT（无则加；超上限则改）
    # sqlglot 对 limit 的表示可能在不同 query 结构上略有差异；MVP 用字符串层面追加/替换也可
    # 但这里尽量用 AST：对最外层加 limit
    limit_expr = tree.args.get("limit")
    if limit_expr and isinstance(limit_expr, exp.Limit):
        n = limit_expr.expression
        try:
            val = int(n.name) if isinstance(n, exp.Literal) else None
        except Exception:
            val = None
        if val is None or val > policy.max_limit:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(policy.max_limit)))
    else:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(policy.max_limit)))

    rewritten = tree.sql(dialect="postgres")

    meta = {
        "tables_used": [f"{sch}.{tbl}" for sch, tbl in tables_used],
        "max_limit": policy.max_limit,
        "schema": policy.allow_schema,
    }
    return rewritten, meta