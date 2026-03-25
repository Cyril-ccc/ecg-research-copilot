# SECURITY.md — SQL 防线设计

本文档描述 ECG Research Copilot 如何防止用户或 LLM 生成的 SQL 破坏数据库安全性。

---

## 防线概述

所有 SQL（无论来自用户手动输入、`/tools/run_sql`，还是 `/tools/build_cohort` 的模板生成）必须先通过以下四层防线，才能到达数据库。

```
用户 / LLM
    │
    ▼
┌──────────────────────────────────────┐
│  Layer 1：词法拦截（正则）             │
│  DROP / DELETE / INSERT / CREATE … 等 │
│  含分号的多语句一律拒绝               │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│  Layer 2：AST 校验（sqlglot）         │
│  只允许 SELECT 类节点                 │
│  禁止 INSERT/UPDATE/DELETE/DDL 节点   │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│  Layer 3：表 & 字段白名单             │
│  只允许 schema_whitelist.yaml 中声明  │
│  的表和字段；列名必须带表别名前缀     │
│  禁止 SELECT *                        │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│  Layer 4：执行硬护栏（Postgres）      │
│  transaction_read_only = on           │
│  statement_timeout = '8s'             │
│  强制注入 LIMIT，上限 5000 行         │
└──────────────────────┬───────────────┘
                       │
                       ▼
                  Postgres (只读会话)
```

---

## 各防线详情

### Layer 1 — 词法拦截（正则，`sql_safety.py`）

在解析 AST 之前，用正则快速检查原始 SQL 字符串：

| 规则 | 拦截内容 |
|---|---|
| 禁止关键词 | `DROP`, `DELETE`, `INSERT`, `UPDATE`, `CREATE`, `ALTER`, `GRANT`, `REVOKE`, `TRUNCATE`, `VACUUM`, `COPY`, `ANALYZE` |
| 禁止分号 | 防止多语句注入（`SELECT 1; DROP TABLE x`） |

**为什么不依赖 AST？** AST 解析偶有方言差异，正则作为快速一道防线可阻止最明显的攻击。

---

### Layer 2 — AST 结构校验（sqlglot）

使用 [sqlglot](https://github.com/tobymao/sqlglot) 将 SQL 解析为 AST：

- 只允许 **单条语句**（`len(parsed) != 1` → 拒绝）
- 顶层节点不能是 `INSERT / UPDATE / DELETE / CREATE / DROP / ALTER / COPY / Command`

---

### Layer 3 — 表 & 字段白名单（`config/schema_whitelist.yaml`）

白名单以 YAML 配置声明，格式为：

```yaml
allow:
  public:
    labevents: [subject_id, hadm_id, itemid, charttime, valuenum]
    d_labitems: [itemid, label]
    # ...
```

校验点：

| 检查 | 行为 |
|---|---|
| 表不在白名单 | 拒绝，返回 `table not allowed: schema.table` |
| 列不在白名单 | 拒绝，返回 `column not allowed: schema.table.col` |
| 列不带表前缀 | 拒绝，返回 `unqualified column not allowed` |
| `SELECT *` | 拒绝，返回 `star (*) is not allowed` |

新增允许字段只需编辑 `config/schema_whitelist.yaml`，无需改代码。

---

### Layer 4 — 数据库执行硬护栏

每次执行前，在同一事务中设置 Postgres 会话级参数：

```sql
SET LOCAL transaction_read_only = on;   -- 即使校验漏掉写操作也无法执行
SET LOCAL statement_timeout = '8s';     -- 超时自动终止，防资源耗尽
SET LOCAL search_path = public;         -- 锁定 schema，防跨 schema 访问
```

同时，查询结果强制注入 `LIMIT`（上限 5000 行，cohort 接口默认 1000 行）。

---

## 审计日志

每次 SQL 请求（允许或拒绝）均写入 `audit_logs` 表：

| 字段 | 内容 |
|---|---|
| `run_id` | 关联的研究 run |
| `actor` | 请求来源（`X-Actor` header，默认 `local`） |
| `action` | `run_sql_request` / `build_cohort_request` / `build_cohort_execute` 等 |
| `payload` | SQL SHA-256、是否允许、拒绝原因、使用的表 |

所有审计记录**不可删除**（无 DELETE 接口），便于合规追踪。

---

## 已知局限 & 未来改进

| 局限 | 说明 |
|---|---|
| 只支持 PostgreSQL 方言 | sqlglot 以 `read="postgres"` 解析，其他数据库方言可能绕过 |
| 子查询的表/列白名单 | 当前会校验所有嵌套 SELECT 中的表，但极复杂 CTE 可能有边界情况 |
| actor 为自报字段 | `X-Actor` header 来自客户端，未做身份验证（local-first 场景可接受） |
