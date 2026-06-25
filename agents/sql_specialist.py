"""
SQL Specialist Agent
--------------------
Single responsibility: translate the architect's plan into a single clean
read-only SQL query that can be executed against the source database.

Consumes:  state["architect_plan"], state["schema_context"], state["error_log"]
Produces:  state["sql_query"], state["error_log"]

- Strips markdown fences from model output.
- Strips <think>...</think> reasoning blocks (emitted by some models).
- Retries on prior failures via error_log feedback.
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

# ── model singleton ────────────────────────────────────────────────────────────
from config.model import llm_pipeline as _llm, safe_invoke

log = logging.getLogger(__name__)


# ── system prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = textwrap.dedent("""
    You are a SQL Specialist Agent writing queries for a SQLite database.
    Your only job is to write a single, clean, read-only SQL query.

    RULES:
    - Output ONLY the SQL query. No explanation. No markdown. No preamble.
    - End the query with a semicolon.
    - Use only tables and columns that appear in SCHEMA CONTEXT.
    - Forbidden keywords: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, SELECT *.

    SQLITE DATE FUNCTIONS (use these ONLY):
    - Last 30 days:  WHERE date_col >= DATE('now', '-30 days')
    - Last 7 days:   WHERE date_col >= DATE('now', '-7 days')
    - Last quarter:  WHERE date_col >= DATE('now', '-3 months')
    - Last year:     WHERE date_col >= DATE('now', '-1 year')
    - This month:    WHERE date_col >= DATE('now', 'start of month')
    - This year:     WHERE date_col >= DATE('now', 'start of year')
    - Extract year:  strftime('%Y', date_col)
    - Extract month: strftime('%m', date_col)

    FORBIDDEN FUNCTIONS (these are PostgreSQL/MySQL, not SQLite):
    - DATE_TRUNC()      → use DATE('now', 'start of year') or strftime() instead
    - DATE_SUB()        → use DATE('now', '-N days') instead
    - CURDATE()         → use DATE('now') instead
    - GETDATE()         → use DATE('now') instead
    - NOW()             → use DATETIME('now') instead
    - EXTRACT()         → use strftime() instead

    SUBQUERY RULES (CRITICAL FOR SQLITE):
    - Every subquery in a FROM clause MUST have an alias.
      CORRECT: SELECT * FROM (SELECT id FROM users) AS sub
      WRONG:   SELECT * FROM (SELECT id FROM users)
    - Do not use double parentheses `))` unnecessarily.

    If you cannot produce a valid query, output exactly: ERROR
""").strip()


# ── prompt builder ─────────────────────────────────────────────────────────────
def _build_prompt(
    plan: dict[str, Any],
    schema: str,
    error_log: list[str],
) -> str:
    """Destructure architect plan into labeled prose sections."""
    objective    = plan.get("objective", "")
    tables       = ", ".join(plan.get("tables", []))
    raw_filters = plan.get("filters", [])
    filters_list = []
    for f in raw_filters:
        if isinstance(f, dict):
            filters_list.append(f.get("condition", str(f)))
        elif isinstance(f, str):
            filters_list.append(f)
    filters = "\n  - ".join(filters_list) or "None"

    raw_aggs = plan.get("aggregations", [])
    aggs_list = []
    for a in raw_aggs:
        if isinstance(a, dict):
            agg_str = f"{a.get('type', 'AGG')}({a.get('column', '?')}) AS {a.get('alias', 'value')}"
            aggs_list.append(agg_str)
        elif isinstance(a, str):
            aggs_list.append(a)
    aggregations = "\n  - ".join(aggs_list) or "None"
    output_cols  = ", ".join(plan.get("output_columns", []))
    order_by     = plan.get("order_by") or "None"
    limit        = plan.get("limit")
    warnings     = plan.get("warnings", [])

    joins_raw = plan.get("joins", [])
    if joins_raw:
        join_parts = []
        for j in joins_raw:
            if isinstance(j, dict):
                join_parts.append(
                    f"  - {j.get('left', '?')} JOIN {j.get('right', '?')} ON {j.get('on', '?')}"
                )
            else:
                join_parts.append(f"  - {j}")
        joins = "\n".join(join_parts)
    else:
        joins = "None"

    feedback_text = ""
    if error_log:
        feedback_text = textwrap.dedent(f"""
            PRIOR ATTEMPT FAILED — correct ALL of these in your query:
            {chr(10).join(f"  - {e}" for e in error_log[-3:])}

            Write the SQL query now.
        """).strip()

    return textwrap.dedent(f"""
        OBJECTIVE:
        {objective}

        TABLES TO USE:
        {tables}

        JOINS:
        {joins}

        FILTERS:
        {filters}

        AGGREGATIONS:
        {aggregations}

        OUTPUT COLUMNS:
        {output_cols}

        ORDER BY:
        {order_by}

        LIMIT:
        {limit if limit is not None else "None"}

        WARNINGS FROM ARCHITECT:
        {chr(10).join(f"  - {w}" for w in warnings) if warnings else "None"}

        SCHEMA CONTEXT:
        {schema if schema else "(no schema context provided)"}

        {feedback_text}

        Write the SQL query now.
    """).strip()


def _extract_sql(text: str) -> str:
    """
    Robust SQL extraction — strips <think> tags, markdown fences, and preamble.

    Some models (e.g. Qwen3) emit <think>...</think> reasoning blocks by
    default; they must be stripped BEFORE looking for SQL. We also handle
    unclosed <think> tags (model truncated mid-thought) by treating <think>
    as "discard to end". Harmless no-op for models that don't emit think
    tags (e.g. Llama 3.3, gpt-oss-120b).
    """
    # 1. Strip <think>...</think> blocks (closed)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 2. Strip unclosed <think> ... (to end of string)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Try markdown fences first (```sql ... ``` or ``` ... ```)
    fence = "`" * 3
    pattern = fence + r"(?:sql)?(.*?)" + fence
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 4. No fences — find the first SELECT and grab from there to first ';'
    m = re.search(r"\bSELECT\b", text, re.IGNORECASE)
    if m:
        sql_part = text[m.start():]
        end = sql_part.find(";")
        if end != -1:
            return sql_part[: end + 1].strip()
        return sql_part.strip()

    # 5. Fallback — return whatever's left after think-tag stripping
    return text.strip()


# ── node ───────────────────────────────────────────────────────────────────────
def sql_specialist_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node. Translates architect_plan into a SQL query string.
    Raises on unrecoverable input problems so the orchestrator circuit breaker fires.
    """
    plan: dict[str, Any] = state.get("architect_plan", {})
    schema: str = state.get("schema_context", "").strip()
    error_log: list[str] = state.get("error_log", [])

    # ── pre-invoke guards ──────────────────────────────────────────────────────
    if not plan:
        raise ValueError("sql_specialist_node received empty architect_plan")

    if plan.get("feasible") is False:
        raise ValueError(
            f"sql_specialist_node called with infeasible plan: "
            f"{plan.get('infeasible_reason', 'no reason given')}"
        )

    if not plan.get("tables"):
        raise ValueError("sql_specialist_node: architect plan has no tables — cannot generate SQL")

    # ── build prompt ───────────────────────────────────────────────────────────
    user_content = _build_prompt(plan, schema, error_log)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    # ── invoke model ───────────────────────────────────────────────────────────
    log.info(
        "SQL Specialist invoking model | tables=%s | retry_feedback=%d errors",
        plan.get("tables"),
        len(error_log),
    )
    response = safe_invoke(_llm, messages)
    raw_output: str = response.content

    # ── clean output ───────────────────────────────────────────────────────────
    sql_query = _extract_sql(raw_output)

    if sql_query.upper() == "ERROR":
        msg = "SQL Specialist returned ERROR — could not produce valid query"
        log.error(msg)
        error_log.append(f"[sql_specialist] {msg}")
        return {**state, "sql_query": "", "error_log": error_log}

    log.info("SQL Specialist complete | query_len=%d chars", len(sql_query))
    return {**state, "sql_query": sql_query, "error_log": error_log}
