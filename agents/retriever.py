"""
Retriever Agent
---------------
Single responsibility: fetch and format schema context from the
database before the architect plans anything.

*Optimized with Semantic Schema Sampling for categorical columns.*

Consumes:  state["user_request"]
Produces:  state["schema_context"]  ← formatted string, ready for prompt injection

Design principles:
- No LLM inference. Pure data retrieval only.
- Keyword extraction from user_request to scope which tables to fetch.
- Degrades gracefully: if DB is unreachable, returns empty context with a warning.
- Never blocks the pipeline — schema_context can be empty; downstream agents handle it.

Supported backends (configure via env vars):
  RETRIEVER_BACKEND = "sqlite" | "snowflake" | "mock"
  DB_PATH           = path to SQLite file (sqlite backend)
  SNOWFLAKE_* = standard Snowflake env vars (snowflake backend)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import textwrap
from typing import Any

log = logging.getLogger("retriever")

# ── backend selection ──────────────────────────────────────────────────────────
# Default changed from 'mock' → 'sqlite' to prevent silent fallback to a
# 2-table fake schema during dataset generation. If you actually want mock,
# set RETRIEVER_BACKEND=mock explicitly and you'll get a loud warning.
RETRIEVER_BACKEND: str = os.getenv("RETRIEVER_BACKEND", "sqlite").lower()
DB_PATH: str = os.getenv("DB_PATH", "memory/sandbox.db")

if RETRIEVER_BACKEND == "mock":
    log.warning(
        "⚠️  RETRIEVER_BACKEND=mock — generated SQL will NOT be validated "
        "against a real database (Layer 3 dry-run is skipped). "
        "Use this only for smoke tests, not dataset generation."
    )
elif RETRIEVER_BACKEND == "sqlite" and not os.path.exists(DB_PATH):
    log.warning(
        "⚠️  DB_PATH '%s' does not exist — run seed_ecommerce.py first.",
        DB_PATH,
    )


# ── keyword extraction ─────────────────────────────────────────────────────────
def _extract_table_hints(user_request: str) -> list[str]:
    """
    Returns empty list to signal the backend to fetch all tables.
    Works across all schemas (v2, e-commerce, SaaS, retail) without
    hardcoding table names. The backend caps at 10 tables to protect
    context window size.
    """
    return []


# ── schema formatters ──────────────────────────────────────────────────────────
def _format_schema(tables: list[dict[str, Any]]) -> str:
    """
    Convert raw table metadata into a compact string for prompt injection.
    """
    if not tables:
        return ""

    lines: list[str] = []
    for table in tables:
        name = table.get("table_name", "unknown")
        columns = table.get("columns", [])
        lines.append(f"TABLE {name}")
        for col in columns:
            col_name = col.get("name", "?")
            col_type = col.get("type", "UNKNOWN")
            nullable = "" if col.get("notnull") else "  -- nullable"
            pk = " PK" if col.get("pk") else ""
            
            # Semantic Sampling Optimization
            samples = col.get("samples", [])
            sample_str = f"  -- samples: {samples}" if samples else ""
            
            lines.append(f"  {col_name:<25} {col_type}{pk}{nullable}{sample_str}")
        lines.append("")  # blank line between tables

    return "\n".join(lines).strip()


# ── backends ───────────────────────────────────────────────────────────────────
def _fetch_sqlite(hints: list[str]) -> list[dict[str, Any]]:
    """
    Fetch table schemas from a local SQLite database.
    Scopes to tables matching hints; falls back to all tables if no hints match.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in cursor.fetchall()]

        # scope to hinted tables if possible
        if hints:
            target_tables = [t for t in all_tables if any(h in t.lower() for h in hints)]
            if not target_tables:
                log.warning("No tables matched hints %s — fetching all tables", hints)
                target_tables = all_tables
        else:
            target_tables = all_tables

        # limit to avoid flooding the context window
        target_tables = target_tables[:20]
        log.info("Fetching schema for tables: %s", target_tables)

        results: list[dict[str, Any]] = []
        for table_name in target_tables:
            cursor.execute(f"PRAGMA table_info({table_name})")  # noqa: S608
            columns = []
            for row in cursor.fetchall():
                col_name = row[1]
                col_type = row[2]
                is_notnull = bool(row[3])
                is_pk = bool(row[5])
                samples = []
                
                # Fetch semantic samples for Text/Varchar columns
                if "TEXT" in col_type.upper() or "VARCHAR" in col_type.upper():
                    try:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 10")
                        samples = [str(r[0]) for r in cursor.fetchall()]
                    except sqlite3.Error:
                        pass
                        
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "notnull": is_notnull,
                    "pk": is_pk,
                    "samples": samples
                })
            results.append({"table_name": table_name, "columns": columns})

        conn.close()
        return results

    except sqlite3.Error as exc:
        log.error("SQLite retrieval failed: %s", exc)
        return []


def _fetch_snowflake(hints: list[str]) -> list[dict[str, Any]]:
    """
    Fetch table schemas from Snowflake via INFORMATION_SCHEMA.
    Requires: SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA env vars.
    """
    try:
        import snowflake.connector  # type: ignore

        database = os.environ["SNOWFLAKE_DATABASE"]
        schema   = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")

        conn = snowflake.connector.connect(
            user       = os.environ["SNOWFLAKE_USER"],
            password   = os.environ["SNOWFLAKE_PASSWORD"],
            account    = os.environ["SNOWFLAKE_ACCOUNT"],
            database   = database,
            schema     = schema,
        )
        cursor = conn.cursor()

        # build hint filter
        if hints:
            like_clauses = " OR ".join(
                f"LOWER(TABLE_NAME) LIKE '%{h.lower()}%'" for h in hints
            )
            where_clause = f"AND ({like_clauses})"
        else:
            where_clause = ""

        cursor.execute(textwrap.dedent(f"""
            SELECT
                TABLE_NAME
            FROM {database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{schema}'
            {where_clause}
            LIMIT 20
        """))
        table_names = [row[0] for row in cursor.fetchall()]
        log.info("Snowflake tables matched: %s", table_names)

        results: list[dict[str, Any]] = []
        for table_name in table_names:
            cursor.execute(textwrap.dedent(f"""
                SELECT
                    COLUMN_NAME
                  , DATA_TYPE
                  , IS_NULLABLE
                FROM {database}.INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{schema}'
                  AND TABLE_NAME   = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """))
            
            columns = []
            for row in cursor.fetchall():
                col_name = row[0]
                col_type = row[1]
                is_notnull = row[2] == "NO"
                samples = []
                
                # Fetch semantic samples for Text/Varchar columns in Snowflake
                if "TEXT" in col_type.upper() or "VARCHAR" in col_type.upper() or "STRING" in col_type.upper():
                    try:
                        cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{database}"."{schema}"."{table_name}" SAMPLE (1000 ROWS) WHERE "{col_name}" IS NOT NULL LIMIT 3')
                        samples = [str(r[0]) for r in cursor.fetchall()]
                    except Exception:
                        pass
                
                columns.append({
                    "name":    col_name,
                    "type":    col_type,
                    "notnull": is_notnull,
                    "pk":      False,   # INFORMATION_SCHEMA doesn't expose PK directly
                    "samples": samples
                })
            results.append({"table_name": table_name, "columns": columns})

        conn.close()
        return results

    except KeyError as exc:
        log.error("Missing Snowflake env var: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        log.error("Snowflake retrieval failed: %s", exc)
        return []


def _fetch_mock(hints: list[str]) -> list[dict[str, Any]]:
    """
    Mock backend for Kaggle / local development without a live database.
    """
    log.info("Using mock schema backend (RETRIEVER_BACKEND=mock)")
    return [
        {
            "table_name": "orders",
            "columns": [
                {"name": "order_id",     "type": "INTEGER", "notnull": True,  "pk": True},
                {"name": "customer_id",  "type": "INTEGER", "notnull": True,  "pk": False},
                {"name": "region",       "type": "TEXT",    "notnull": False, "pk": False, "samples": ["North America", "EMEA", "APAC"]},
                {"name": "total_amount", "type": "REAL",    "notnull": False, "pk": False},
                {"name": "status",       "type": "TEXT",    "notnull": False, "pk": False, "samples": ["completed", "refunded", "pending"]},
                {"name": "created_at",   "type": "TEXT",    "notnull": False, "pk": False},
            ],
        },
        {
            "table_name": "customers",
            "columns": [
                {"name": "customer_id",  "type": "INTEGER", "notnull": True,  "pk": True},
                {"name": "customer_name","type": "TEXT",    "notnull": True,  "pk": False},
                {"name": "country",      "type": "TEXT",    "notnull": False, "pk": False, "samples": ["US", "UK", "ID"]},
                {"name": "created_at",   "type": "TEXT",    "notnull": False, "pk": False},
            ],
        },
    ]


# ── dispatch ───────────────────────────────────────────────────────────────────
_BACKENDS = {
    "sqlite":    _fetch_sqlite,
    "snowflake": _fetch_snowflake,
    "mock":      _fetch_mock,
}


def _fetch_schema(hints: list[str]) -> list[dict[str, Any]]:
    backend_fn = _BACKENDS.get(RETRIEVER_BACKEND)
    if backend_fn is None:
        log.error(
            "Unknown RETRIEVER_BACKEND '%s'. Valid options: %s",
            RETRIEVER_BACKEND,
            list(_BACKENDS.keys()),
        )
        return []
    return backend_fn(hints)


# ── node ───────────────────────────────────────────────────────────────────────
def retriever_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node. Extracts table hints from the user request,
    fetches schema metadata, and formats it for downstream prompt injection.
    """
    user_request: str = state.get("user_request", "").strip()

    if not user_request:
        log.warning("Retriever received empty user_request — returning empty schema context")
        return {**state, "schema_context": ""}

    # ── extract hints ──────────────────────────────────────────────────────────
    hints = _extract_table_hints(user_request)

    # ── fetch schema ───────────────────────────────────────────────────────────
    log.info("Retriever fetching schema | backend=%s | hints=%s", RETRIEVER_BACKEND, hints)
    try:
        raw_tables = _fetch_schema(hints)
    except Exception as exc:  # noqa: BLE001
        log.error("Schema fetch failed entirely: %s — continuing with empty context", exc)
        raw_tables = []

    # ── format ────────────────────────────────────────────────────────────────
    schema_context = _format_schema(raw_tables)

    if not schema_context:
        log.warning(
            "Retriever produced empty schema context | backend=%s | hints=%s",
            RETRIEVER_BACKEND,
            hints,
        )
    else:
        table_count = len(raw_tables)
        log.info(
            "Retriever complete | tables=%d | context_len=%d chars",
            table_count,
            len(schema_context),
        )

    return {**state, "schema_context": schema_context}
