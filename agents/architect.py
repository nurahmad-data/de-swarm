"""
Architect Agent
---------------
Single responsibility: decompose a natural-language request into a structured
JSON task plan. The architect NEVER writes SQL. It only plans.

Consumes:  state["user_request"], state["schema_context"], state["error_log"]
Produces:  state["architect_plan"], state["error_log"]

The plan is consumed by the SQL Specialist to generate the actual query.
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
    You are an Architect Agent in a SQL data pipeline.
    Your only job is to decompose a business request into a structured task plan.

    RULES:
    - Do NOT write any SQL.
    - STRICT SCHEMA ENFORCEMENT: Never invent tables or columns. Use EXACTLY what is in the SCHEMA CONTEXT.
    - WATCH OUT FOR NAMES: Use 'customer_name' or 'product_name' instead of a generic 'name'.

    SCHEMA INFERENCE RULES (CRITICAL):
    - Sample values shown in the schema context are EXAMPLES, not exhaustive lists.
    - If a request mentions a specific value (e.g., 'webhook_triggered', 'Enterprise', 'overage'),
      assume it exists as a value in an appropriate categorical column (e.g., 'event_type', 'plan_type', 'line_type').
      Do NOT mark the request as infeasible just because the exact value is not in the sample list.
    - If a request mentions a metric that can be DERIVED from existing columns
      (e.g., 'overage amount' = SUM(amount) WHERE line_type='overage',
      'ticket volume' = COUNT(*) FROM support_tickets, 'MRR' = SUM(amount) FROM invoices),
      mark it as FEASIBLE and specify the derivation in the plan.
    - When in doubt about feasibility, lean toward FEASIBLE and let the SQL Specialist handle the implementation.

    - If the schema context is genuinely empty or the request requires a table/column that truly does not exist,
      set "feasible": false and explain why.
    - Output ONLY valid JSON. No preamble. No markdown. No explanation outside the JSON.
    
    OUTPUT CONTRACT (you must match this schema exactly):
    {
      "feasible": true | false,
      "infeasible_reason": "<only if feasible is false, otherwise omit>",
      "objective": "<one sentence summary of what the query must achieve>",
      "tables": ["<table1>", "<table2>"],
      "filters": ["<condition1>", "<condition2>"],
      "aggregations": ["<agg1>"],
      "joins": [
        {"left": "<table>", "right": "<table>", "on": "<condition>"}
      ],
      "output_columns": ["<col1>", "<col2>"],
      "order_by": "<column and direction, or null>",
      "limit": <integer or null>,
      "warnings": ["<any ambiguities the SQL Specialist must be aware of>"]
    }
""").strip()


# ── output validation ──────────────────────────────────────────────────────────
# Only 'feasible' is globally required. Other keys are conditional.
_REQUIRED_KEYS = {"feasible"}

# These keys are ONLY required when feasible=true
_REQUIRED_IF_FEASIBLE = {
    "objective", "tables", "filters",
    "aggregations", "joins", "output_columns",
}


def _validate_plan(plan: dict[str, Any]) -> list[str]:
    """Return a list of validation errors. Empty list = plan is valid."""
    errors: list[str] = []

    # 1. Check the absolute minimum: 'feasible' must exist
    missing = _REQUIRED_KEYS - plan.keys()
    if missing:
        errors.append(f"Missing required keys: {missing}")
        return errors  # Stop here if 'feasible' is missing

    if not isinstance(plan.get("feasible"), bool):
        errors.append("'feasible' must be a boolean")
        return errors

    # 2. If feasible=true, require the full plan structure
    if plan.get("feasible"):
        missing_if_feasible = _REQUIRED_IF_FEASIBLE - plan.keys()
        if missing_if_feasible:
            errors.append(f"Missing required keys for feasible plan: {missing_if_feasible}")

        if not plan.get("tables"):
            errors.append("'tables' must be non-empty when feasible=true")

        if not plan.get("output_columns"):
            errors.append(
                "'output_columns' must be non-empty when feasible=true — "
                "derive explicit column names from schema_context; do not use SELECT *"
            )

    # 3. If feasible=false, require an infeasible_reason
    else:
        if not plan.get("infeasible_reason"):
            errors.append("'infeasible_reason' required when feasible=false")

    # 4. Warnings should always be a list if present
    if not isinstance(plan.get("warnings", []), list):
        errors.append("'warnings' must be a list")

    return errors


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Safely extract JSON from LLM output.
    Strips <think>...</think> tags (emitted by some models like Qwen3), then
    finds the first '{' and the last '}' to bypass markdown fences entirely.
    Harmless no-op for models that don't emit think tags (e.g. Llama 3.3).
    """
    # Strip thinking-mode reasoning blocks (closed or unclosed) — defensive
    # against models like Qwen3 that emit <think>...</think> by default.
    raw = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.DOTALL | re.IGNORECASE)
    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output:\n{raw[:300]}")

    cleaned = raw[start:end]
    return json.loads(cleaned)


# ── node ───────────────────────────────────────────────────────────────────────
def architect_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node. Translates user_request + schema_context into architect_plan.
    Raises on unrecoverable input problems so the orchestrator circuit breaker fires.
    """
    user_request: str = state.get("user_request", "").strip()
    schema_context: str = state.get("schema_context", "").strip()
    error_log: list[str] = list(state.get("error_log", []))

    # ── pre-invoke guards ──────────────────────────────────────────────────────
    if not user_request:
        raise ValueError("architect_node received empty user_request")

    # ── build prompt ───────────────────────────────────────────────────────────
    user_content = textwrap.dedent(f"""
        USER REQUEST:
        {user_request}

        SCHEMA CONTEXT:
        {schema_context if schema_context else "(no schema context provided)"}

        Decompose the request into a structured task plan.
        Output ONLY valid JSON matching the contract in the system prompt.
    """).strip()

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    # ── invoke model ───────────────────────────────────────────────────────────
    log.info(
        "Architect invoking model | request_len=%d | prior_failures=%d",
        len(user_request), len(error_log),
    )
    response = safe_invoke(_llm, messages)
    raw_output: str = response.content

    # ── parse + validate ───────────────────────────────────────────────────────
    try:
        plan = _extract_json(raw_output)
    except (json.JSONDecodeError, ValueError) as exc:
        msg = f"Architect output was not valid JSON: {exc}"
        log.error(msg)
        log.error("Raw output (first 500 chars): %s", raw_output[:500])
        error_log.append(f"[architect] {msg}")
        return {
            **state,
            "architect_plan": {"feasible": False, "infeasible_reason": msg},
            "error_log": error_log,
        }

    errors = _validate_plan(plan)
    if errors:
        msg = f"Architect plan failed validation: {errors}"
        log.error(msg)
        log.error("Plan: %s", json.dumps(plan, indent=2)[:500])
        error_log.append(f"[architect] {msg}")
        return {
            **state,
            "architect_plan": {"feasible": False, "infeasible_reason": msg},
            "error_log": error_log,
        }

    # ── log warnings (non-fatal) ───────────────────────────────────────────────
    for warning in plan.get("warnings", []):
        log.warning("Architect warning: %s", warning)

    log.info(
        "Architect plan complete | tables=%s | feasible=%s",
        plan.get("tables", []), plan.get("feasible"),
    )

    return {**state, "architect_plan": plan, "error_log": error_log}
