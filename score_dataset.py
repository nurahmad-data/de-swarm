"""
score_dataset.py
----------------
Execute every SQL in a generated dataset JSONL against the e-commerce SQLite
DB and append execution-level scoring columns.

This is the quality gate between dataset generation and SFT training. It
catches the failure modes the de-swarm validator cannot:

  - SQL that passes Layer 3 (EXPLAIN QUERY PLAN) but fails at execution
    (runtime errors: NULL in arithmetic, malformed dates, division by zero,
    type coercion bugs, GROUP BY/HAVING issues)
  - SQL that executes successfully but returns 0 rows (model shouldn't learn
    to produce empty-result queries — they're usually wrong date filters)
  - Schema/plan misalignment invisible to static analysis (wrong JOIN type,
    missing DISTINCT, off-by-one date filter)

Output: <input_stem>_scored.jsonl
Each row gets these new fields appended:
  - execution_success   : bool   — did the SQL run without error?
  - execution_error     : str    — error message (empty if success)
  - row_count           : int    — number of rows returned
  - has_results         : bool   — row_count > 0
  - result_preview      : list[dict]  — first 3 rows, truncated
  - execution_time_ms   : int    — query latency in milliseconds

Usage:
  python3 score_dataset.py [INPUT_JSONL] [--db PATH] [--output PATH]
                                                [--in-place]
                                                [--workers N]
                                                [--no-stats]
                                                [--incremental-stats]

Defaults:
  INPUT_JSONL  = dataset/de-swarm-dataset-v3.jsonl
  --db         = $DB_PATH or memory/ecommerce.db
  --output     = <input_stem>_scored.jsonl
  --workers    = 4 (SQLite reads parallelize fine across connections)

Environment variables:
  DB_PATH   default: memory/ecommerce.db  (consistent with validator.py)

Resume behavior:
  When --output exists, prompts already present in it are skipped. Safe to
  re-run after Ctrl+C or any interruption.

Examples:
  # Score the default dataset file
  python3 score_dataset.py

  # Score with explicit input + custom DB
  python3 score_dataset.py dataset/my_run.jsonl --db memory/ecommerce.db

  # Score and overwrite input (auto-creates .bak backup)
  python3 score_dataset.py --in-place
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("score_dataset")

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT = Path("dataset/de-swarm-dataset-v3.jsonl")
DEFAULT_DB = os.getenv("DB_PATH", "memory/ecommerce.db")
DEFAULT_WORKERS = 4
MAX_PREVIEW_ROWS = 3
MAX_PREVIEW_COLS = 5
MAX_CELL_LEN = 50
QUERY_TIMEOUT_S = 5.0      # SQLite connect timeout
BUSY_TIMEOUT_MS = 5000     # PRAGMA busy_timeout — concurrent read protection


# ── SQL cleaning ───────────────────────────────────────────────────────────────
_MD_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _clean_sql(raw: str) -> str:
    """
    Defensive SQL cleaning.

    The sql_specialist.py already strips markdown fences, but we re-strip
    here in case rows came from an older pipeline, were hand-edited, or
    got corrupted. Defense in depth — never trust input.
    """
    if not raw:
        return ""
    sql = raw.strip()
    # Strip markdown code fences (```sql ... ``` → ...)
    sql = _MD_FENCE_RE.sub("", sql).strip()
    # Some models emit trailing prose after the semicolon. Keep only the
    # first statement up to and including the first ';'.
    if ";" in sql:
        first = sql.split(";", 1)[0].strip()
        if first:
            sql = first + ";"
    return sql


# ── execution ──────────────────────────────────────────────────────────────────
def _execute_sql(db_path: str, sql: str) -> dict[str, Any]:
    """
    Execute a single SQL query against the e-commerce DB in READ-ONLY mode.

    Returns a dict with execution_success, execution_error, row_count,
    has_results, result_preview, execution_time_ms.

    Defense in depth:
      - READ-ONLY mode (mode=ro) prevents any write even if SQL contains
        forbidden keywords that slipped past the validator's regex layer
      - busy_timeout prevents crashes on concurrent reads
      - Single-statement enforcement (first non-empty up to ';')
      - Per-query timeout via SQLite connection timeout
    """
    result: dict[str, Any] = {
        "execution_success": False,
        "execution_error": "",
        "row_count": 0,
        "has_results": False,
        "result_preview": [],
        "execution_time_ms": 0,
    }

    cleaned = _clean_sql(sql)
    if not cleaned:
        result["execution_error"] = "empty_sql"
        return result

    t0 = time.time()
    conn = None
    try:
        # READ-ONLY connection — defense in depth. Even if a DROP statement
        # somehow slipped past the validator's regex layer, SQLite will
        # refuse to execute it here.
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=QUERY_TIMEOUT_S,
        )
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        cursor = conn.cursor()

        cursor.execute(cleaned)
        rows = cursor.fetchall()
        col_names = (
            [d[0] for d in cursor.description] if cursor.description else []
        )

        elapsed_ms = int((time.time() - t0) * 1000)
        result["execution_success"] = True
        result["row_count"] = len(rows)
        result["has_results"] = len(rows) > 0
        result["execution_time_ms"] = elapsed_ms

        # Build truncated preview — useful for spot-checking failures
        preview_cols = col_names[:MAX_PREVIEW_COLS]
        for row in rows[:MAX_PREVIEW_ROWS]:
            row_dict = {}
            for col, val in zip(preview_cols, row[:MAX_PREVIEW_COLS]):
                sval = str(val) if val is not None else "NULL"
                if len(sval) > MAX_CELL_LEN:
                    sval = sval[:MAX_CELL_LEN - 3] + "..."
                row_dict[col] = sval
            result["result_preview"].append(row_dict)

    except sqlite3.Error as exc:
        elapsed_ms = int((time.time() - t0) * 1000)
        result["execution_error"] = f"sqlite_error: {exc}"
        result["execution_time_ms"] = elapsed_ms
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - t0) * 1000)
        result["execution_error"] = f"{type(exc).__name__}: {exc}"
        result["execution_time_ms"] = elapsed_ms
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    return result


# ── per-row worker ─────────────────────────────────────────────────────────────
def _score_row(row: dict[str, Any], db_path: str) -> dict[str, Any]:
    """Add scoring fields to a single dataset row."""
    sql = row.get("sql", "")
    exec_result = _execute_sql(db_path, sql)
    return {**row, **exec_result}


# ── I/O ────────────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all rows from a JSONL file, skipping malformed lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in %s: %s",
                            line_num, path, exc)
    return rows


def _load_scored_prompts(output_path: Path) -> set[str]:
    """
    Resume support — load prompts already in the output file so we don't
    re-score them. Only skips rows that have an `execution_success` field
    (i.e., scoring completed on that row).
    """
    scored: set[str] = set()
    if not output_path.exists():
        return scored
    with output_path.open("r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "execution_success" in data and "prompt" in data:
                    scored.add(data["prompt"])
            except json.JSONDecodeError:
                continue
    return scored


# ── stats ──────────────────────────────────────────────────────────────────────
def _compute_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics over scored rows."""
    total = len(rows)
    if total == 0:
        return {"total": 0}

    exec_success = sum(1 for r in rows if r.get("execution_success"))
    has_results = sum(1 for r in rows if r.get("has_results"))
    both = sum(
        1 for r in rows
        if r.get("execution_success") and r.get("has_results")
    )
    validation_passed = sum(1 for r in rows if r.get("validation_passed"))
    sft_eligible = sum(
        1 for r in rows
        if r.get("validation_passed")
        and r.get("execution_success")
        and r.get("has_results")
    )

    # Row-count distribution buckets — surfaces "empty result" failures
    buckets: Counter = Counter()
    for r in rows:
        if not r.get("execution_success"):
            buckets["failed"] += 1
            continue
        n = r.get("row_count", 0)
        if n == 0:
            buckets["0_rows"] += 1
        elif n <= 10:
            buckets["1-10_rows"] += 1
        elif n <= 100:
            buckets["11-100_rows"] += 1
        else:
            buckets["100+_rows"] += 1

    # Top error patterns — surfaces recurring failure modes
    error_patterns: Counter = Counter()
    for r in rows:
        err = r.get("execution_error", "")
        if not err:
            continue
        # Normalize: keep first 80 chars so similar errors cluster
        error_patterns[err[:80]] += 1

    # Latency stats (successful queries only)
    times = [
        r.get("execution_time_ms", 0)
        for r in rows
        if r.get("execution_success")
    ]

    return {
        "total": total,
        "validation_passed_count": validation_passed,
        "validation_pass_rate_pct": round(validation_passed / total * 100, 2),
        "execution_success_count": exec_success,
        "execution_success_rate_pct": round(exec_success / total * 100, 2),
        "has_results_count": has_results,
        "has_results_rate_pct": round(has_results / total * 100, 2),
        "exec_success_and_has_results_count": both,
        "exec_success_and_has_results_rate_pct": round(both / total * 100, 2),
        "sft_eligible_count": sft_eligible,
        "sft_eligible_rate_pct": round(sft_eligible / total * 100, 2),
        "row_count_distribution": dict(buckets),
        "top_errors": error_patterns.most_common(10),
        "latency_ms": {
            "min": min(times) if times else 0,
            "max": max(times) if times else 0,
            "mean": round(statistics.mean(times), 2) if times else 0,
            "p50": round(statistics.median(times), 2) if times else 0,
        },
    }


def _print_stats(stats: dict[str, Any]) -> None:
    """Pretty-print stats to stdout for live monitoring."""
    log.info("═══ Scoring Summary ═══")
    log.info("  Total rows                : %d", stats["total"])
    log.info("  Validation passed         : %d (%.2f%%)",
             stats["validation_passed_count"],
             stats["validation_pass_rate_pct"])
    log.info("  Execution success         : %d (%.2f%%)",
             stats["execution_success_count"],
             stats["execution_success_rate_pct"])
    log.info("  Has results (row_count>0) : %d (%.2f%%)",
             stats["has_results_count"],
             stats["has_results_rate_pct"])
    log.info("  ─────────────────────────────────────")
    log.info("  ★ SFT-eligible (val+exec+rows): %d (%.2f%%)",
             stats["sft_eligible_count"],
             stats["sft_eligible_rate_pct"])
    log.info("")
    log.info("  Row count distribution:")
    for bucket, count in sorted(stats["row_count_distribution"].items()):
        log.info("    %-15s : %d", bucket, count)
    log.info("")
    log.info("  Latency (ms): min=%d  p50=%d  mean=%d  max=%d",
             stats["latency_ms"]["min"], stats["latency_ms"]["p50"],
             stats["latency_ms"]["mean"], stats["latency_ms"]["max"])
    if stats["top_errors"]:
        log.info("")
        log.info("  Top error patterns:")
        for err, count in stats["top_errors"]:
            log.info("    [%d] %s", count, err)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a generated dataset JSONL by executing each SQL "
                    "against the e-commerce SQLite DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Input JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite DB path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <input_stem>_scored.jsonl)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file (auto-creates <input>.bak backup)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip stats computation and printing",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        return 1

    if not os.path.exists(args.db):
        log.error("DB file not found: %s — run seed_ecommerce.py first", args.db)
        return 1

    # Resolve output path
    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_scored.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # In-place mode: auto-backup before touching the input
    if args.in_place:
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        log.info("IN-PLACE mode — backing up to %s", backup)
        backup.write_text(input_path.read_text())

    log.info("Loading input: %s", input_path)
    rows = _load_jsonl(input_path)
    log.info("Loaded %d rows", len(rows))

    # Resume: skip already-scored prompts (separate-output mode only)
    already_scored: set[str] = set()
    if not args.in_place and output_path.exists():
        already_scored = _load_scored_prompts(output_path)
        if already_scored:
            log.info("Resume — %d rows already scored, %d remaining",
                     len(already_scored), len(rows) - len(already_scored))

    to_score = [r for r in rows if r.get("prompt") not in already_scored]

    if not to_score:
        log.info("Nothing to do — all rows already scored.")
        if not args.no_stats:
            all_rows = _load_jsonl(output_path)
            stats = _compute_stats(all_rows)
            _print_stats(stats)
            stats_path = output_path.parent / f"{output_path.stem}_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2))
            log.info("Stats written → %s", stats_path)
        return 0

    # Score with thread pool — SQLite reads parallelize fine across
    # separate connections, and we're network/CPU-bound on parsing, not
    # DB-bound (e-commerce DB is tiny: 200 orders, 30 customers).
    write_lock = threading.Lock()
    passed = 0
    failed = 0
    start_time = time.time()

    def _worker(row: dict[str, Any]) -> dict[str, Any]:
        return _score_row(row, args.db)

    try:
        if args.in_place:
            # Collect all then rewrite atomically
            scored_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, r): r for r in to_score}
                for i, fut in enumerate(as_completed(futures), 1):
                    scored = fut.result()
                    scored_rows.append(scored)
                    if scored.get("execution_success"):
                        passed += 1
                    else:
                        failed += 1
                    if i % 50 == 0 or i == len(to_score):
                        _log_progress(i, len(to_score), passed, failed,
                                      start_time)
            # Sort by original order to keep the file readable
            prompt_to_scored = {r["prompt"]: r for r in scored_rows}
            ordered = [prompt_to_scored.get(r.get("prompt"), r) for r in rows]
            log.info("Writing scored output (in-place): %s", output_path)
            with output_path.open("w") as f:
                for r in ordered:
                    f.write(json.dumps(r) + "\n")
        else:
            # Stream-write: append as each row completes (resume-safe)
            with ThreadPoolExecutor(max_workers=args.workers) as pool, \
                 output_path.open("a") as f:
                futures = {pool.submit(_worker, r): r for r in to_score}
                for i, fut in enumerate(as_completed(futures), 1):
                    scored = fut.result()
                    with write_lock:
                        f.write(json.dumps(scored) + "\n")
                        f.flush()
                    if scored.get("execution_success"):
                        passed += 1
                    else:
                        failed += 1
                    if i % 50 == 0 or i == len(to_score):
                        _log_progress(i, len(to_score), passed, failed,
                                      start_time)

    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl+C). Partial results saved.")
        return 130

    elapsed = time.time() - start_time
    log.info("Scoring complete | elapsed=%.1fs | pass=%d | fail=%d",
             elapsed, passed, failed)

    # Stats — read full output to include both fresh + resumed rows
    if not args.no_stats:
        all_rows = _load_jsonl(output_path)
        stats = _compute_stats(all_rows)
        _print_stats(stats)
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        log.info("Stats written → %s", stats_path)

    return 0


def _log_progress(i: int, total: int, passed: int, failed: int,
                  start_time: float) -> None:
    """Emit a progress line with rolling ETA."""
    elapsed = time.time() - start_time
    rate = i / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate if rate > 0 else 0
    log.info(
        "─── [%d/%d] exec_pass=%d  exec_fail=%d  | %.1f rows/s | ETA: %.0fs",
        i, total, passed, failed, rate, eta,
    )


if __name__ == "__main__":
    sys.exit(main())
