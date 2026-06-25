"""
Validator Agent
---------------
Single responsibility: verify the SQL query is correct, safe, and
faithful to the architect's plan before it reaches execution.

Consumes:  state["sql_query"], state["architect_plan"], state["schema_context"]
Produces:  state["validation_result"]

Validation runs in three layers in strict order:
  Layer 1 — Static Analysis    (regex, no model) — always runs
  Layer 2 — Logical Validation (LLM single-pass cross-check vs plan & schema)
  Layer 3 — Execution Dry-Run  (EXPLAIN QUERY PLAN via SQLite — skipped on mock backend)

Never raises. All failures encode as passed=False in ValidationResult.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import textwrap
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

# ── model singleton ────────────────────────────────────────────────────────────
from config.model import llm_pipeline as _llm, safe_invoke

log = logging.getLogger(__name__)

RETRIEVER_BACKEND: str = os.getenv("RETRIEVER_BACKEND", "sqlite").lower()
DB_PATH: str = os.getenv("DB_PATH", "memory/sandbox.db")

# Dataset generation mode: skip Layer 2 LLM validation to cut Groq calls
# from 3/prompt → 2/prompt (~33% latency + cost reduction).
# Layer 1 (regex) + Layer 3 (EXPLAIN dry-run) remain — they're free and
# deterministic, and they catch the most common failure modes.
SKIP_LLM_VALIDATION: bool = os.getenv("SKIP_LLM_VALIDATION", "false").lower() == "true"

# ── result contract ────────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    passed: bool
    layer: str          # which layer made the final decision
    reason: str         # human-readable verdict
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed":   self.passed,
            "layer":    self.layer,
            "reason":   self.reason,
            "failures": self.failures,
            "warnings": self.warnings,
        }


# ── helpers ────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict[str, Any]:
        """
        Robust JSON extraction — strips <think> tags, then bypasses markdown
        fences by locating curly braces.

        Qwen3 emits <think>...</think> reasoning blocks by default; strip them
        first so they don't pollute the JSON parse.
        """
        # Strip <think>...</think> blocks (closed or unclosed)
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in LLM output:\n{text[:300]}")
        cleaned = text[start:end]
        return json.loads(cleaned)


def _llm_check(prompt: str) -> dict[str, Any]:
    """
    Call the LLM with a validation prompt. Returns parsed JSON result.

    Fail-closed on any error — better to retry a prompt than let invalid
    SQL slip into the training dataset. The previous 'conservative pass'
    behavior was dangerous for dataset generation: a Groq 429 during
    validation would mark broken SQL as valid and pollute the SFT set.
    """
    try:
        response = safe_invoke(_llm, [HumanMessage(content=prompt)])
        return _extract_json(response.content)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM validation call failed: %s — failing CLOSED.", exc)
        return {
            "passed": False,
            "reason": f"LLM validation failed (fail-closed): {exc}",
            "failures": [f"LLM validation call failed: {exc}"],
            "warnings": [],
        }


# ── layer 1 — static analysis ─────────────────────────────────────────────────
# Forbidden patterns that must never appear in a generated query.
_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bDROP\b",     "DROP statement detected"),
    (r"\bDELETE\b",   "DELETE statement detected"),
    (r"\bTRUNCATE\b", "TRUNCATE statement detected"),
    (r"\bINSERT\b",   "INSERT statement detected"),
    (r"\bUPDATE\b",   "UPDATE statement detected"),
    (r"\bCREATE\b",   "CREATE statement detected"),
    (r"\bALTER\b",    "ALTER statement detected"),
]

_WARN_PATTERNS: list[tuple[str, str]] = [
    (r"SELECT\s+\*", "SELECT * detected — should list columns explicitly"),
]


def _layer1_static(sql: str) -> ValidationResult:
    failures: list[str] = []
    warnings: list[str] = []

    upper = sql.upper()
    for pattern, message in _FORBIDDEN_PATTERNS:
        if re.search(pattern, upper):
            failures.append(message)

    for pattern, message in _WARN_PATTERNS:
        if re.search(pattern, upper, re.IGNORECASE):
            warnings.append(message)

    if not sql.strip().endswith(";"):
        warnings.append("Query does not end with ';'")

    if failures:
        return ValidationResult(
            passed=False, layer="layer1_static",
            reason=f"Static analysis blocked query: {'; '.join(failures)}",
            failures=failures, warnings=warnings,
        )

    return ValidationResult(
        passed=True, layer="layer1_static",
        reason="Static analysis passed", warnings=warnings,
    )


# ── layer 2 — logical validation (plan alignment + schema compliance) ────────
# Merged into a single LLM call to halve validation latency on local hardware.
_LAYER2_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a SQL validation agent.

    ARCHITECT PLAN:
    {plan}

    SCHEMA CONTEXT:
    {schema}

    SQL QUERY:
    {sql}

    IMPORTANT: date columns stored as TEXT in ISO format (e.g. '2026-06-15') are fully compatible with SQLite DATE() functions. Do NOT flag TEXT date columns as incompatible.

    Perform two checks in one pass:
    1. PLAN ALIGNMENT: Does the query implement all filters, joins, aggregations, and output columns from the plan?
    2. SCHEMA COMPLIANCE: Do all tables and columns in the query actually exist in the schema context?

    Respond ONLY with valid JSON:
    {{
      "passed": true or false,
      "reason": "<one sentence verdict covering both checks>",
      "failures": ["<specific failure 1>", "<specific failure 2>"],
      "warnings": ["<optional warning>"]
    }}
""").strip()


def _layer2_logical_validation(
    sql: str, plan: dict[str, Any], schema_context: str
) -> ValidationResult:
    """
    Single LLM call covering both plan alignment and schema compliance.
    Replaces the original two-call approach to halve local inference time.
    If schema_context is empty, only plan alignment is checked.
    """
    schema_section = schema_context if schema_context else "No schema context provided — skip schema compliance check."

    prompt = _LAYER2_PROMPT_TEMPLATE.format(
        plan=json.dumps(plan, indent=2),
        schema=schema_section,
        sql=sql,
    )
    result = _llm_check(prompt)
    passed = bool(result.get("passed", True))
    return ValidationResult(
        passed=passed,
        layer="layer2_logical_validation",
        reason=result.get("reason", "Logical validation check complete"),
        failures=result.get("failures", []),
        warnings=result.get("warnings", []),
    )

# ── layer 3 — execution dry-run ─────────────────────────────────────────────
def _layer3_dry_run(sql: str) -> ValidationResult:
    """
    Run EXPLAIN QUERY PLAN against a real SQLite DB.

    Guard: skipped entirely on mock backend — no real DB exists there.
    Also skipped if DB_PATH does not exist yet (pre-Step 4).
    """
    if RETRIEVER_BACKEND == "mock":
        log.info("Layer 3 skipped — mock backend, no real DB to validate against.")
        return ValidationResult(
            passed=True, layer="layer3_dry_run",
            reason="Skipped — mock backend",
        )

    if not os.path.exists(DB_PATH):
        log.warning("Layer 3 skipped — DB_PATH '%s' does not exist.", DB_PATH)
        return ValidationResult(
            passed=True, layer="layer3_dry_run",
            reason=f"Skipped — DB not found at {DB_PATH}",
            warnings=[f"Layer 3 dry-run skipped: {DB_PATH} missing"],
        )

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # EXPLAIN QUERY PLAN validates syntax and object references without executing.
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return ValidationResult(
            passed=True, layer="layer3_dry_run",
            reason="EXPLAIN QUERY PLAN succeeded — syntax and schema references valid",
        )
    except sqlite3.Error as exc:
        return ValidationResult(
            passed=False, layer="layer3_dry_run",
            reason=f"Execution dry-run failed: {exc}",
            failures=[str(exc)],
        )


# ── node ───────────────────────────────────────────────────────────────────────
def validator_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node. Runs four validation layers in strict order.
    Never raises — all failures encode as passed=False.
    Returns updated state with validation_result populated.
    Also appends failure reasons to error_log for downstream retry feedback.
    """
    sql: str = state.get("sql_query", "").strip()
    plan: dict[str, Any] = state.get("architect_plan", {})
    schema_context: str = state.get("schema_context", "").strip()
    error_log: list[str] = list(state.get("error_log", []))

    if not sql:
        result = ValidationResult(
            passed=False, layer="pre-check",
            reason="No SQL query in state",
            failures=["sql_query is empty"],
        )
        return {**state, "validation_result": result.to_dict()}

    all_warnings: list[str] = []

    # ── Layer 1 ────────────────────────────────────────────────────────────────
    r1 = _layer1_static(sql)
    all_warnings.extend(r1.warnings)
    if not r1.passed:
        log.warning("Validation failed at Layer 1: %s", r1.reason)
        error_log.append(f"[validator:layer1] {r1.reason} | failures: {r1.failures}")
        result = ValidationResult(
            passed=False, layer=r1.layer, reason=r1.reason,
            failures=r1.failures, warnings=all_warnings,
        )
        return {**state, "validation_result": result.to_dict(), "error_log": error_log}

    # ── Layer 2 (Logical Validation) ───────────────────────────────────────────
    # Skippable for dataset generation via SKIP_LLM_VALIDATION=true.
    # Saves ~33% of Groq API calls per prompt — major speedup on long runs.
    if SKIP_LLM_VALIDATION:
        log.debug("Layer 2 skipped (SKIP_LLM_VALIDATION=true)")
        r2 = ValidationResult(
            passed=True, layer="layer2_skipped",
            reason="Skipped — SKIP_LLM_VALIDATION=true",
        )
    else:
        r2 = _layer2_logical_validation(sql, plan, schema_context)
    all_warnings.extend(r2.warnings)
    if not r2.passed:
        log.warning("Validation failed at Layer 2: %s", r2.reason)
        error_log.append(f"[validator:layer2] {r2.reason} | failures: {r2.failures}")
        result = ValidationResult(
            passed=False, layer=r2.layer, reason=r2.reason,
            failures=r2.failures, warnings=all_warnings,
        )
        return {**state, "validation_result": result.to_dict(), "error_log": error_log}

    # ── Layer 3 (Execution Dry-Run) ────────────────────────────────────────────
    r3 = _layer3_dry_run(sql)
    all_warnings.extend(r3.warnings)
    if not r3.passed:
        log.warning("Validation failed at Layer 3: %s", r3.reason)
        error_log.append(f"[validator:layer3] {r3.reason} | failures: {r3.failures}")
        result = ValidationResult(
            passed=False, layer=r3.layer, reason=r3.reason,
            failures=r3.failures, warnings=all_warnings,
        )
        return {**state, "validation_result": result.to_dict(), "error_log": error_log}

    # ── all layers passed ──────────────────────────────────────────────────────
    log.info("Validation passed all layers.")
    result = ValidationResult(
        passed=True, layer="all",
        reason="Query passed all validation layers",
        warnings=all_warnings,
    )
    return {**state, "validation_result": result.to_dict(), "error_log": error_log}
