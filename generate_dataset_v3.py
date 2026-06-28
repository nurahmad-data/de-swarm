"""
generate_dataset_v3.py
----------------------
Batch script to generate a curated query/SQL dataset using the de-swarm pipeline.

Optimized for Groq cloud inference:
  - Concurrent workers (ThreadPoolExecutor) — default 4, tune via MAX_WORKERS
  - Rate-limit aware (the rate limiter lives in config.model.safe_invoke)
  - Resume logic only skips SUCCESS+VALIDATED prompts (was: skipped everything
    including failed → permanently losing retryable prompts on resume)
  - Per-row elapsed time recorded for ETA + post-hoc analysis

Output: dataset/de-swarm-dataset-v3.jsonl
Each line is a JSON object:
  {
    "prompt": "<user request>",
    "sql": "<generated SQL>",
    "architect_plan": {...},
    "validation_passed": true,
    "validation_layer": "all",
    "status": "success" | "failed" | "api_error",
    "error_log": [],
    "elapsed_seconds": 12.34,
    "completed_at": "2026-06-17T12:34:56Z"
  }

Environment variables:
  PROMPT_FILE        default: ecommerce_prompts.txt
                     (was v3_augmented_prompts.txt — filename mismatch)
  DATASET_OUTPUT     default: de-swarm-dataset-v3.jsonl
  MAX_WORKERS        default: 4  (Groq free tier; bump to 8-10 for dev tier)
  SKIP_LLM_VALIDATION  set to "true" to skip validator Layer 2 LLM call
                       (saves ~33% Groq calls per prompt — strongly recommended)
  DEBUG_OUTPUT       set to "true" to write per-prompt JSON dumps (default false)

Run:
  SKIP_LLM_VALIDATION=true python3 generate_dataset_v3.py
"""

import hashlib
import json
import os
import statistics
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

# ── configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("dataset_gen_v3")

# ── output ─────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATASET_DIR / os.getenv("DATASET_OUTPUT", "de-swarm-dataset-v3.jsonl")

# ── concurrency ────────────────────────────────────────────────────────────────
# Default 4 workers stays safe under Groq's 30 RPM free tier when paired with
# the rate limiter in config.model. Each prompt makes 2 LLM calls (architect +
# sql_specialist) when SKIP_LLM_VALIDATION=true, so 4 workers × 2 calls = 8 RPM
# per worker, well under the 28 RPM limiter cap.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# ── prompts ────────────────────────────────────────────────────────────────────
# FIXED: default filename now matches what augment_prompts_ecommerce.py writes.
# Previously defaulted to v3_augmented_prompts.txt → FileNotFoundError.
prompt_file = DATASET_DIR / os.getenv("PROMPT_FILE", "ecommerce_prompts.txt")
try:
    with prompt_file.open("r") as f:
        PROMPTS = [line.strip() for line in f if line.strip()]
    log.info("Loaded %d augmented prompts from %s", len(PROMPTS), prompt_file)
except FileNotFoundError:
    log.error("%s not found. Run augment_prompts_ecommerce.py first.", prompt_file)
    exit(1)


# ── resume helper ─────────────────────────────────────────────────────────────
def _load_completed() -> set[str]:
    """
    Load prompts that have ALREADY SUCCEEDED from existing output file.

    FIXED: previously added any non-api_error row to `completed`, including
    failed rows. This caused prompts that failed due to transient Groq 429s
    (escalated through the orchestrator's circuit breaker) to be permanently
    skipped on resume. Now we only skip prompts that both:
      - status == "success"
      - validation_passed == True

    This means a re-run will retry every failed prompt. Slightly more work,
    but the only way to guarantee a clean 5000-row dataset.
    """
    completed = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if (
                        "prompt" in data
                        and data.get("status") == "success"
                        and data.get("validation_passed") is True
                    ):
                        completed.add(data["prompt"])
                except (json.JSONDecodeError, KeyError):
                    log.warning("Skipping malformed line in output file.")
                    continue
    return completed


# ── runner ─────────────────────────────────────────────────────────────────────
def run_pipeline(prompt: str) -> dict:
    """Run a single prompt through the de-swarm pipeline and return the result."""
    from orchestrator import run
    try:
        stable_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        result = run(prompt, thread_id=f"dataset-v3-{stable_hash}")
        return {
            "prompt": prompt,
            "sql": result.get("sql_query", ""),
            "architect_plan": result.get("architect_plan", {}),
            "validation_passed": result.get("validation_result", {}).get("passed", False),
            "validation_layer": result.get("validation_result", {}).get("layer", ""),
            "status": result.get("status", "failed"),
            "error_log": result.get("error_log", []),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        log.error("Pipeline failed for prompt '%s': %s", prompt[:50], exc)
        return {
            "prompt": prompt,
            "sql": "",
            "architect_plan": {},
            "validation_passed": False,
            "validation_layer": "",
            "status": "api_error",
            "error_log": [str(exc)],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }


# ── per-prompt worker ──────────────────────────────────────────────────────────
write_lock = threading.Lock()


def process_one(prompt: str) -> dict:
    """Process a single prompt: run pipeline, time it, append to JSONL."""
    t0 = time.time()
    result = run_pipeline(prompt)
    elapsed = time.time() - t0
    result["elapsed_seconds"] = round(elapsed, 2)

    with write_lock:
        with OUTPUT_FILE.open("a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

    return result


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    total = len(PROMPTS)
    passed = 0
    failed = 0
    skipped = 0
    execution_times: list[float] = []

    # ── resume: skip already-succeeded prompts ─────────────────────────────────
    completed = _load_completed()
    to_process = [p for p in PROMPTS if p not in completed]

    if completed:
        log.info(
            "Resuming — %d prompts already succeeded, %d to (re)process.",
            len(completed), len(to_process),
        )
    else:
        log.info(
            "Starting fresh dataset generation | prompts=%d | workers=%d | output=%s",
            total, MAX_WORKERS, OUTPUT_FILE,
        )
    log.info(
        "Config | SKIP_LLM_VALIDATION=%s | MAX_WORKERS=%d",
        os.getenv("SKIP_LLM_VALIDATION", "false"), MAX_WORKERS,
    )

    start_time = time.time()

    try:
        # Concurrent execution via ThreadPoolExecutor.
        # Network-bound (Groq API), not CPU-bound → threads beat processes here.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(process_one, p): p for p in to_process}

            for i, fut in enumerate(as_completed(futures), 1):
                prompt = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    # Shouldn't happen — process_one catches everything — but
                    # guard against thread-pool-level failures just in case.
                    log.error("Worker crashed on prompt '%s': %s", prompt[:50], exc)
                    result = {
                        "prompt": prompt,
                        "sql": "",
                        "status": "api_error",
                        "validation_passed": False,
                        "error_log": [str(exc)],
                        "elapsed_seconds": 0.0,
                    }

                elapsed = result.get("elapsed_seconds", 0.0)
                execution_times.append(elapsed)

                if result["status"] == "success" and result["validation_passed"]:
                    passed += 1
                else:
                    failed += 1

                # ── live ETA using rolling window of last 20 prompts ──────
                window = execution_times[-20:]
                avg_time = statistics.mean(window)
                remaining = len(to_process) - i
                # Account for concurrency in ETA — wall-clock time, not summed time
                eta_seconds = (avg_time / MAX_WORKERS) * max(remaining, 0)
                eta_h, rem = divmod(int(eta_seconds), 3600)
                eta_m, _ = divmod(rem, 60)

                status_icon = "✓" if (result["status"] == "success" and result["validation_passed"]) else "✗"
                log.info(
                    "─── [%d/%d] %s | %.1fs | sql_len=%d | ETA: %02d:%02d | %s",
                    i, len(to_process), status_icon, elapsed,
                    len(result.get("sql", "")), eta_h, eta_m,
                    prompt[:60],
                )

    except KeyboardInterrupt:
        log.warning("Pipeline manually interrupted by user (Ctrl+C).")

    finally:
        total_time = time.time() - start_time
        # Count skipped = total prompts that were already completed at start
        skipped = total - len(to_process)
        log.info("═══ Dataset generation complete ═══")
        log.info("  Total prompts   : %d", total)
        log.info("  Skipped (done)  : %d", skipped)
        log.info("  Passed          : %d", passed)
        log.info("  Failed          : %d", failed)
        log.info("  Wall time       : %.1fs (%.1f min)", total_time, total_time / 60)
        if execution_times:
            log.info("  Avg per prompt  : %.1fs (concurrency-adjusted: %.1fs)",
                     statistics.mean(execution_times),
                     statistics.mean(execution_times) / MAX_WORKERS)
        log.info("  Output          : %s", OUTPUT_FILE)
        log.info("  Safe to resume — re-run script to continue from where it stopped.")


if __name__ == "__main__":
    main()
