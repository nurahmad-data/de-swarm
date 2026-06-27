"""
de-swarm Orchestrator
---------------------
Main agent — owns the LangGraph state machine, routing logic,
circuit breaker, and retry discipline.

All nodes are imported from agents/. The orchestrator never
executes SQL or writes prompts — it only routes and validates.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ── agents ────────────────────────────────────────────────────────────────────
from agents.architect import architect_node
from agents.retriever import retriever_node
from agents.sql_specialist import sql_specialist_node
from agents.validator import validator_node

# ── constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
OUTPUT_DIR = Path("agent-output")
HANDOFF_DIR = Path("handoff")

# Checkpointer selection — env var so you can switch back to SqliteSaver
# if you ever need persistent cross-session checkpoints.
# Default: memory (in-process) — required for concurrent dataset generation.
# SqliteSaver causes 'database is locked' under ThreadPoolExecutor.
CHECKPOINTER: str = os.getenv("CHECKPOINTER", "memory").lower()

# Disable per-prompt JSON dumps during dataset generation.
# Default: off (silent) — set DEBUG_OUTPUT=true for interactive debugging.
# Generating 5000 prompts with debug output on writes 20,000+ timestamped
# files to disk, slowing every prompt and trashing the filesystem.
DEBUG_OUTPUT: bool = os.getenv("DEBUG_OUTPUT", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("orchestrator")


# ── shared state ──────────────────────────────────────────────────────────────
class SwarmState(TypedDict):
    user_request: str
    schema_context: str
    architect_plan: dict[str, Any]
    sql_query: str
    validation_result: dict[str, Any]
    current_node: str
    retry_count: int
    error_log: list[str]
    status: Literal["running", "success", "failed"]
    messages: Annotated[list, add_messages]


# ── routing helpers ────────────────────────────────────────────────────────────
def _write_output(name: str, content: str | dict) -> Path | None:
    """Write per-node debug output. Skipped entirely when DEBUG_OUTPUT=false."""
    if not DEBUG_OUTPUT:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = OUTPUT_DIR / f"{name}-{ts}.json"
    payload = content if isinstance(content, str) else json.dumps(content, indent=2)
    path.write_text(payload)
    log.info("Output written → %s", path)
    return path


def _write_handoff(state: SwarmState) -> None:
    """Write handoff snapshot. Skipped entirely when DEBUG_OUTPUT=false."""
    if not DEBUG_OUTPUT:
        return
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "architect_plan": state.get("architect_plan"),
        "schema_context": state.get("schema_context"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = HANDOFF_DIR / "handoff.json"
    path.write_text(json.dumps(snapshot, indent=2))
    log.info("Handoff written → %s", path)


def _escalate(state: SwarmState, node: str, error: str) -> SwarmState:
    errors = state.get("error_log", [])
    errors.append(f"[{node}] attempt {state['retry_count']}: {error}")
    _write_output(
        f"failed-{node}",
        {"node": node, "retry_count": state["retry_count"], "errors": errors},
    )
    log.error("Circuit breaker triggered on %s after %d attempts.", node, MAX_RETRIES)
    return {**state, "error_log": errors, "status": "failed"}


# ── routing functions ─────────────────────────────────────────────────────────
def route_after_retriever(state: SwarmState) -> str:
    if state["status"] == "failed":
        return END
    return "architect"


def route_after_architect(state: SwarmState) -> str:
    if state["status"] == "failed":
        if state["retry_count"] < MAX_RETRIES:
            log.warning("Architect failed — retry %d/%d", state["retry_count"], MAX_RETRIES)
            return "architect"
        return END
    _write_handoff(state)
    return "sql_specialist"


def route_after_sql(state: SwarmState) -> str:
    if state["status"] == "failed":
        if state["retry_count"] < MAX_RETRIES:
            log.warning("SQL Specialist failed — retry %d/%d", state["retry_count"], MAX_RETRIES)
            return "sql_specialist"
        return END
    return "validator"


def route_after_validator(state: SwarmState) -> str:
    result = state.get("validation_result", {})

    if result.get("passed"):
        log.info("Validation passed — pipeline complete.")
        return END

    if state["retry_count"] < MAX_RETRIES:
        failed_layer = result.get("layer")
        failures = result.get("failures", [])

        # Deep Routing Optimization
        if failed_layer == "layer2_logical_validation" and state["retry_count"] >= 1:
            log.warning("Plan is fundamentally flawed. Routing back to Architect.")
            return "architect"

        # NEW: Route hard schema hallucinations directly back to Architect
        if failed_layer == "layer3_dry_run":
            error_text = " ".join(failures).lower()
            if "no such table" in error_text or "no such column" in error_text:
                log.warning("Schema hallucination detected. Routing back to Architect to fix the plan.")
                return "architect"

        log.warning(
            "Validation failed at %s. Routing back to SQL Specialist (retry %d/%d).",
            failed_layer, state["retry_count"], MAX_RETRIES
        )
        return "sql_specialist"

    _escalate(state, "validator", result.get("reason", "unknown"))
    return END

# ── wrapper nodes ─────────────────────────────────────────────────────────────
def retriever_wrapper(state: SwarmState) -> SwarmState:
    log.info("▶ retriever")
    try:
        updated = retriever_node(state)
        _write_output("retriever", updated.get("schema_context", ""))
        return {**updated, "current_node": "retriever", "status": "running"}
    except Exception as exc:
        return _escalate({**state, "retry_count": state["retry_count"] + 1}, "retriever", str(exc))

def architect_wrapper(state: SwarmState) -> SwarmState:
    log.info("▶ architect (attempt %d)", state["retry_count"] + 1)
    try:
        updated = architect_node(state)
        _write_output("architect", updated.get("architect_plan", {}))
        is_retry = bool(updated.get("error_log"))
        new_retry_count = 0 if not is_retry else state["retry_count"]
        return {**updated, "current_node": "architect", "status": "running", "retry_count": new_retry_count}
    except Exception as exc:
        count = state["retry_count"] + 1
        if count >= MAX_RETRIES:
            return _escalate({**state, "retry_count": count}, "architect", str(exc))
        return {**state, "retry_count": count, "status": "running"}

def sql_wrapper(state: SwarmState) -> SwarmState:
    log.info("▶ sql_specialist (attempt %d)", state["retry_count"] + 1)
    try:
        updated = sql_specialist_node(state)
        _write_output("sql_specialist", updated.get("sql_query", ""))
        is_retry = bool(updated.get("error_log"))
        new_retry_count = 0 if not is_retry else state["retry_count"]
        return {**updated, "current_node": "sql_specialist", "status": "running", "retry_count": new_retry_count}
    except Exception as exc:
        count = state["retry_count"] + 1
        if count >= MAX_RETRIES:
            return _escalate({**state, "retry_count": count}, "sql_specialist", str(exc))
        return {**state, "retry_count": count, "status": "running"}

def validator_wrapper(state: SwarmState) -> SwarmState:
    log.info("▶ validator (attempt %d)", state["retry_count"] + 1)
    try:
        updated = validator_node(state)
        _write_output("validator", updated.get("validation_result", {}))
        result = updated.get("validation_result", {})
        new_count = state["retry_count"] + 1 if not result.get("passed") else state["retry_count"]
        return {**updated, "current_node": "validator", "retry_count": new_count, "status": "running"}
    except Exception as exc:
        count = state["retry_count"] + 1
        if count >= MAX_RETRIES:
            return _escalate({**state, "retry_count": count}, "validator", str(exc))
        return {**state, "retry_count": count, "status": "running"}


# ── graph assembly ─────────────────────────────────────────────────────────────
def build_graph(checkpointer) -> StateGraph:
    graph = StateGraph(SwarmState)
    graph.add_node("retriever", retriever_wrapper)
    graph.add_node("architect", architect_wrapper)
    graph.add_node("sql_specialist", sql_wrapper)
    graph.add_node("validator", validator_wrapper)

    graph.add_edge(START, "retriever")
    graph.add_conditional_edges("retriever", route_after_retriever)
    graph.add_conditional_edges("architect", route_after_architect)
    graph.add_conditional_edges("sql_specialist", route_after_sql)
    graph.add_conditional_edges("validator", route_after_validator)

    return graph.compile(checkpointer=checkpointer)


def _make_checkpointer():
    """
    Factory: returns a fresh checkpointer per `run()` invocation.

    MemorySaver is used by default — required for concurrent dataset
    generation under ThreadPoolExecutor. SqliteSaver would hit
    'database is locked' on parallel writes.

    Each call returns a NEW MemorySaver instance, so threads never share
    checkpointer state (fully isolated).
    """
    if CHECKPOINTER == "sqlite":
        # Fallback path for debugging — NOT recommended for concurrent runs.
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = Path("memory/checkpoint.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(db_path))
    # Default: in-memory — instant, thread-safe, no disk I/O.
    return MemorySaver()


# ── entry point ───────────────────────────────────────────────────────────────
def run(user_request: str, thread_id: str = "default") -> SwarmState:
    # Each call gets a fresh checkpointer — fully isolated across threads.
    # MemorySaver is process-local and instant, so no per-call setup cost.
    checkpointer = _make_checkpointer()
    app = build_graph(checkpointer)

    initial_state: SwarmState = {
        "user_request": user_request,
        "schema_context": "",
        "architect_plan": {},
        "sql_query": "",
        "validation_result": {},
        "current_node": "start",
        "retry_count": 0,
        "error_log": [],
        "status": "running",
        "messages": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    log.info("Pipeline started | thread_id=%s", thread_id)
    final_state = app.invoke(initial_state, config=config)
    if final_state.get("validation_result", {}).get("passed"):
        final_state = {**final_state, "status": "success"}
    log.info("Pipeline finished | status=%s", final_state["status"])

    return final_state

if __name__ == "__main__":
    import sys
    request = " ".join(sys.argv[1:]) or "Show total revenue by region for last 30 days"
    result = run(request)
    print(f"\nStatus : {result['status']}")
    print(f"SQL    : {result.get('sql_query', 'N/A')}")
    if result["error_log"]:
        print("Errors :")
        for e in result["error_log"]:
            print(f"  {e}")
