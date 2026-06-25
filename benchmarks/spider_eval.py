#!/usr/bin/env python3
"""
spider_eval.py — Run de-sql-3b-q8 on Spider dev set, compute Execution Accuracy.

Usage:
    python spider_eval.py --model de-sql-3b-q8 --n 200
    python spider_eval.py --model de-sql-3b-q8          # full 1034
    python spider_eval.py --model qwen2.5-coder:3b --n 200   # base comparison
"""
import argparse, json, sqlite3, re, time, sys
from pathlib import Path
import requests

SPIDER_DIR = Path("~/de-swarm/benchmarks/spider_data").expanduser()
DEV_JSON = SPIDER_DIR / "dev.json"
DB_DIR = Path("~/de-swarm/benchmarks/test-suite-sql-eval/database").expanduser()
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are an expert SQLite query generator. Given a database schema and a natural language question, output ONLY valid SQLite SQL. No explanations, no markdown. End with a semicolon.

Schema:
{schema}"""

def get_schema(db_path):
    """Read CREATE TABLE statements from sqlite_master."""
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        ddl = "\n\n".join(r[0] for r in cur.fetchall())
        conn.close()
        return ddl
    except Exception as e:
        return f"-- ERROR: {e}"

def extract_sql(text):
    text = re.sub(r"", "", text, flags=re.DOTALL | re.IGNORECASE)
    fence = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence: text = fence.group(1)
    text = text.strip()
    m = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|PRAGMA)\b", text, re.IGNORECASE)
    if m: text = text[m.start():]
    semi = text.find(";")
    if semi != -1: text = text[: semi + 1]
    return text.strip()

def call_ollama(model, system, user, timeout=120):
    r = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.9, "num_ctx": 4096, "num_predict": 300,
                    "stop": ["<|im_end|>", "<|im_start|>"]}
    }, timeout=timeout + 30)
    r.raise_for_status()
    return r.json()["message"]["content"]

def execute_sql(db_path, sql, timeout_ms=5000):
    if not sql: return False, "EMPTY"
    if re.match(r"\s*(CREATE|DROP|ALTER|INSERT|UPDATE|DELETE|ATTACH|DETACH)\b", sql, re.IGNORECASE):
        return False, "BLOCKED_DDL"
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(f"PRAGMA busy_timeout = {timeout_ms}")
        rows = conn.execute(sql).fetchall()
        conn.close()
        return True, rows
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="de-sql-3b-q8")
    ap.add_argument("--n", type=int, default=200, help="Number of prompts to eval (default 200)")
    ap.add_argument("--output", default="spider_results.json")
    args = ap.parse_args()

    # Load dev set
    with open(DEV_JSON) as f:
        dev = json.load(f)
    
    # Subset
    if args.n and args.n < len(dev):
        dev = dev[:args.n]
    
    print(f"\n  Spider Eval — Model: {args.model}")
    print(f"  Examples: {len(dev)}")
    print(f"  Output: {args.output}\n")
    
    results = []
    ok = 0
    err = 0
    empty = 0
    no_db = 0
    
    t_start = time.time()
    
    for i, ex in enumerate(dev, 1):
        db_id = ex["db_id"]
        db_path = DB_DIR / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            # Try flat layout: database/db_id.sqlite
            db_path = DB_DIR / f"{db_id}.sqlite"
        question = ex["question"]
        gold_sql = ex["query"]
        
        # Verify DB exists
        if not db_path.exists():
            no_db += 1
            print(f"  {i:>4}/{len(dev)} [NO_DB] {db_id}: {question[:50]}")
            results.append({"i": i, "db_id": db_id, "question": question, "gold": gold_sql,
                            "pred": "", "status": "NO_DB", "exec_match": False})
            continue
        
        # Get schema
        schema = get_schema(db_path)
        system = SYSTEM_PROMPT.format(schema=schema)
        
        # Call model
        try:
            raw = call_ollama(args.model, system, question)
            pred_sql = extract_sql(raw)
        except Exception as e:
            err += 1
            print(f"  {i:>4}/{len(dev)} [OLLAMA_ERR] {db_id}: {str(e)[:60]}")
            results.append({"i": i, "db_id": db_id, "question": question, "gold": gold_sql,
                            "pred": "", "status": "OLLAMA_ERR", "exec_match": False})
            continue
        
        if not pred_sql:
            empty += 1
            print(f"  {i:>4}/{len(dev)} [EMPTY] {db_id}: {question[:50]}")
            results.append({"i": i, "db_id": db_id, "question": question, "gold": gold_sql,
                            "pred": "", "status": "EMPTY", "exec_match": False})
            continue
        
        # Execute prediction
        pred_ok, pred_result = execute_sql(db_path, pred_sql)
        
        # Execute gold
        gold_ok, gold_result = execute_sql(db_path, gold_sql)
        
        # Execution accuracy: same row set?
        exec_match = False
        if pred_ok and gold_ok:
            try:
                pred_set = set(map(str, sorted(pred_result)))
                gold_set = set(map(str, sorted(gold_result)))
                exec_match = pred_set == gold_set
            except:
                exec_match = False
        
        if exec_match:
            ok += 1
            marker = "[OK]"
        else:
            err += 1
            marker = "[EXEC_MISS]"
        
        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(dev) - i) / rate if rate > 0 else 0
        
        print(f"  {i:>4}/{len(dev)} {marker} {db_id:<25} {question[:50]:<50} ETA: {eta/60:.1f}min")
        
        results.append({
            "i": i, "db_id": db_id, "question": question,
            "gold": gold_sql, "pred": pred_sql,
            "status": "OK" if exec_match else "EXEC_MISS",
            "exec_match": exec_match,
            "pred_rows": len(pred_result) if pred_ok and isinstance(pred_result, list) else 0,
            "gold_rows": len(gold_result) if gold_ok and isinstance(gold_result, list) else 0
        })
        
        # Save partial results every 10 prompts
        if i % 10 == 0:
            with open(args.output, "w") as f:
                json.dump({"model": args.model, "results": results,
                           "summary": {"total": i, "ok": ok, "err": err, "empty": empty, "no_db": no_db,
                                       "accuracy": round(100 * ok / max(1, i), 1)}}, f, indent=2)
    
    # Final summary
    total = len(dev)
    accuracy = round(100 * ok / max(1, total), 1)
    elapsed = time.time() - t_start
    
    summary = {
        "model": args.model,
        "total": total, "ok": ok, "err": err, "empty": empty, "no_db": no_db,
        "accuracy": accuracy,
        "elapsed_sec": round(elapsed, 1),
        "avg_latency_sec": round(elapsed / max(1, total), 2)
    }
    
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Spider Eval Report — {args.model}")
    print(f"{'='*60}")
    print(f"  Total:      {total}")
    print(f"  OK:         {ok}")
    print(f"  Errors:     {err}")
    print(f"  Empty:      {empty}")
    print(f"  No DB:      {no_db}")
    print(f"  Accuracy:   {accuracy}%")
    print(f"  Elapsed:    {elapsed/60:.1f} min")
    print(f"  Avg/query:  {elapsed/max(1,total):.1f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
