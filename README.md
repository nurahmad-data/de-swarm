# de-swarm

> A multi-agent text-to-SQL distillation pipeline. Generates 5,349 validated training rows from free-tier LLM APIs, fine-tunes Qwen2.5-Coder-3B-Instruct via QLoRA, and ships a 3.3 GB GGUF that runs locally on any 8 GB laptop via Ollama.

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/nurahmad-data/de-sql-3b-v2-gguf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Accuracy: 90%](https://img.shields.io/badge/In--Domain%20Accuracy-90%25-brightgreen)](#evaluation)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
- [Evaluation](#evaluation)
- [Engineering Post-Mortems](#engineering-post-mortems)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Overview

`de-swarm` is a production-grade pipeline that distills the SQL generation capabilities of 70B+ parameter LLMs (Mistral Large, Qwen2.5-72B, DeepSeek-V3, and others) into a 3B parameter student model that runs locally on commodity hardware.

The pipeline orchestrates multiple LLM providers through a LangGraph state machine, generating, validating, and scoring synthetic text-to-SQL training data across three industry-grade database schemas. The resulting dataset is used to QLoRA-fine-tune Qwen2.5-Coder-3B-Instruct, producing a specialist model that achieves **90% execution accuracy** on held-out in-domain prompts and **55.5% zero-shot accuracy** on the Spider benchmark — beating the base 3B model on both.

**Total cloud compute cost: $0.** All dataset generation used free-tier API quotas; training ran on Kaggle's free T4 GPU.

---

## Key Results

### In-Domain Evaluation (100 held-out prompts, 3 schemas)

| Schema        | Prompts | Base Qwen2.5-Coder-3B | de-sql-3b-v2 | Lift      |
| ------------- | ------- | --------------------- | ------------ | --------- |
| E-commerce    | 35      | 68.6%                 | **94.3%**    | +25.7 pp  |
| SaaS / B2B    | 35      | 71.4%                 | **94.3%**    | +22.9 pp  |
| Retail        | 30      | 76.7%                 | **80.0%**    | +3.3 pp   |
| **Overall**   | **100** | **72.0%**             | **90.0%**    | **+18.0 pp** |

### Out-of-Domain Evaluation (Spider 1.0, 200 prompts, 40+ unseen schemas)

| Metric                | Base Qwen2.5-Coder-3B | de-sql-3b-v2 |
| --------------------- | --------------------- | ------------ |
| Execution Accuracy    | 48.5%                 | **55.5%**    |
| Successful Queries    | 97 / 200              | **111 / 200**|
| Avg Latency / Query   | 6.94s                 | 8.56s        |

### Model Footprint

| Property         | Value                              |
| ---------------- | ---------------------------------- |
| Base model       | Qwen2.5-Coder-3B-Instruct          |
| Format           | GGUF (q8_0 quantization)           |
| Size             | 3.3 GB                             |
| RAM at inference | ~4 GB                              |
| Hardware target  | Any 8 GB laptop (CPU-only)         |
| Inference engine | Ollama                             |
| Inference cost   | $0                                 |

---

## Architecture

The pipeline has three logical layers: **schema seeding**, **multi-agent dataset generation**, and **fine-tuning + deployment**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCHEMA LAYER                                  │
│  seed_ecommerce.py   seed_saas.py   seed_retail.py              │
│  (10 tables)         (15 tables)     (11 tables)                │
│       ↓                 ↓               ↓                        │
│  ecommerce.db        saas.db         retail.db                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                DATASET GENERATION LAYER                         │
│                                                                  │
│  augment_prompts_*.py  →  4,000+ NL prompts per schema          │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         LangGraph Orchestrator (orchestrator.py)         │   │
│  │                                                          │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌────────────┐   │   │
│  │  │  Architect   │→ │  SQL Specialist│→ │ Validator  │   │   │
│  │  │  (NL→JSON)   │   │ (JSON→SQL)    │   │ (3-layer)  │   │   │
│  │  └──────────────┘   └──────────────┘   └────────────┘   │   │
│  │         ↑              ↑               ↑                 │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │   Multi-Provider Router (config/model.py)         │   │   │
│  │  │   Mistral | OpenRouter | NVIDIA | Cerebras        │   │   │
│  │  │   Groq | GitHub Models | SambaNova | Gemini       │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  score_dataset.py  →  Execution-validated rows                  │
│  build_sft_dataset.py  →  Deduped, balanced SFT dataset         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              TRAINING & DEPLOYMENT LAYER                        │
│                                                                  │
│  Kaggle T4 GPU  →  QLoRA Fine-Tune (train.ipynb)               │
│         ↓                                                       │
│  HuggingFace Hub  ←  Merged FP16 model                          │
│         ↓                                                       │
│  llama.cpp convert_hf_to_gguf.py  →  FP16 GGUF                 │
│         ↓                                                       │
│  llama-quantize  →  q8_0 GGUF (3.3 GB)                         │
│         ↓                                                       │
│  Ollama (Modelfile.v2)  →  Local inference                      │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Agent Validation (3 layers)

Every generated SQL query passes through three validation gates before entering the training set:

1. **Regex gate** — Blocks DDL (`CREATE`, `DROP`, `ALTER`) and DML (`INSERT`, `UPDATE`, `DELETE`) to ensure only read-only queries are trained on.
2. **LLM semantic check** — A separate LLM call verifies the SQL semantically matches the prompt (e.g., the prompt asked for "revenue" and the SQL actually computes revenue, not count). Skippable via `SKIP_LLM_VALIDATION=true` for speed.
3. **SQLite EXPLAIN QUERY PLAN** — Dry-runs the SQL against the real seeded database. If SQLite can't produce a query plan, the row is rejected.

Only rows passing all three gates are kept.

---

## Repository Structure

```
de-swarm/
├── config/
│   └── model.py                 # Multi-provider LLM config + RateLimiter
├── agents/
│   ├── architect.py             # NL → JSON plan agent
│   ├── sql_specialist.py        # JSON plan → SQL agent
│   ├── validator.py             # 3-layer validation agent
│   └── retriever.py             # Schema DDL fetcher
├── data/
│   ├── ecommerce.db             # Seeded e-commerce database
│   ├── saas.db                  # Seeded SaaS database
│   └── retail.db                # Seeded retail database
├── prompts/
│   ├── ecommerce_prompts.txt    # ~1,500 NL prompts
│   ├── saas_prompts.txt         # ~2,250 NL prompts
│   ├── saas_prompts_3plus.txt   # 3+ table join prompts
│   └── retail_prompts.txt       # ~1,547 NL prompts
├── scripts/
│   ├── seed_ecommerce.py        # E-commerce schema seeder
│   ├── seed_saas.py             # SaaS schema seeder
│   ├── seed_retail.py           # Retail schema seeder
│   ├── augment_prompts_ecommerce.py
│   ├── augment_prompts_saas.py
│   ├── augment_prompts_retail.py
│   ├── generate_dataset_v3.py   # Multi-provider generation runner
│   ├── score_dataset.py         # Execute SQL, drop failures
│   └── build_sft_dataset.py     # Dedup + balance + format for SFT
├── orchestrator.py              # LangGraph state machine
├── nightly_100.sh               # Chunked batch runner (tmux-friendly)
├── train.ipynb                  # Kaggle QLoRA training notebook
├── Modelfile.v2                 # Ollama config (ChatML template)
├── api_server.py                # FastAPI REST wrapper for Ollama
├── eval_ollama.py               # In-domain eval (100 prompts)
├── benchmarks/
│   ├── spider_eval.py           # Spider benchmark runner
│   └── compare_benchmarks.py    # Side-by-side base vs fine-tuned
├── data/final_sft_dataset.jsonl # 5,349 training rows
├── .env.template                # API key template
├── requirements.txt
└── README.md
```

---

## Quick Start

### Run the fine-tuned model (2 minutes)

```bash
# Install Ollama (if not already)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model from HuggingFace
ollama pull hf.co/nurahmad-data/de-sql-3b-v2-gguf

# Run interactively
ollama run hf.co/nurahmad-data/de-sql-3b-v2-gguf "show me top 10 customers by revenue"
```

### Use programmatically

```python
import requests

schema_ddl = """
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
"""

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "hf.co/nurahmad-data/de-sql-3b-v2-gguf",
    "messages": [
        {"role": "system", "content": f"You are a SQL generation assistant. Output ONLY SQL.\n\nSchema:\n{schema_ddl}"},
        {"role": "user", "content": "show me top 10 customers by revenue"}
    ],
    "stream": False,
    "options": {"temperature": 0, "num_predict": 300}
})

print(response.json()["message"]["content"])
# SELECT c.name, SUM(o.total) AS revenue
# FROM customers c
# JOIN orders o ON c.customer_id = o.customer_id
# GROUP BY c.customer_id
# ORDER BY revenue DESC
# LIMIT 10;
```

---

## Installation

### Prerequisites

- Python 3.10+
- Ollama 0.3.10+ (for inference)
- Kaggle account with GPU enabled (for training, optional)
- API keys for at least one LLM provider (for dataset generation)

### Setup

```bash
# Clone
git clone https://github.com/nurahmad-data/de-swarm.git
cd de-swarm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.template .env
# Edit .env and add your API keys (Mistral, Groq, NVIDIA, etc.)

# Seed the databases (creates ecommerce.db, saas.db, retail.db)
python scripts/seed_ecommerce.py
python scripts/seed_saas.py
python scripts/seed_retail.py

# Generate prompts
python scripts/augment_prompts_ecommerce.py
python scripts/augment_prompts_saas.py
python scripts/augment_prompts_retail.py
```

---

## Pipeline Stages

### Stage 1: Schema Seeding

Each `seed_*.py` script creates a SQLite database with realistic synthetic data:

```bash
python scripts/seed_ecommerce.py   # → data/ecommerce.db (50K orders, 12 months)
python scripts/seed_saas.py        # → data/saas.db (200 orgs, 12 months events)
python scripts/seed_retail.py      # → data/retail.db (200K fact rows)
```

### Stage 2: Prompt Augmentation

Generates NL prompts across four complexity tiers:

```bash
python scripts/augment_prompts_ecommerce.py  # → prompts/ecommerce_prompts.txt
python scripts/augment_prompts_saas.py       # → prompts/saas_prompts.txt
python scripts/augment_prompts_retail.py     # → prompts/retail_prompts.txt
```

### Stage 3: Dataset Generation

Runs the multi-agent pipeline across all providers. This is the slow stage — overnight runs recommended:

```bash
# Run 100 prompts per schema per night
./nightly_100.sh

# Or run a specific schema with a specific provider
LLM_PROVIDER=mistral PROMPT_FILE=saas_prompts.txt python scripts/generate_dataset_v3.py
```

### Stage 4: Scoring & SFT Dataset Build

Validates generated SQL by executing against the real databases:

```bash
python scripts/score_dataset.py           # Drops rows where SQL fails to execute
python scripts/build_sft_dataset.py       # Dedupes, balances, formats for SFT
# → data/final_sft_dataset.jsonl (5,349 rows)
```

### Stage 5: Fine-Tuning (Kaggle T4)

Upload `data/final_sft_dataset.jsonl` to Kaggle, then run `train.ipynb`. Key config:

- Base: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Method: QLoRA (4-bit NF4, LoRA r=16, alpha=32)
- Epochs: 2
- Learning rate: 2e-4
- Max seq length: 2048
- Packing: enabled
- `completion_only_loss=True`

### Stage 6: GGUF Conversion & Quantization

```bash
# Convert merged model to FP16 GGUF
python llama.cpp/convert_hf_to_gguf.py ./models/merged-v2 --outtype f16

# Quantize to q8_0 (DO NOT use q4_k_m — see Post-Mortem below)
./llama.cpp/build/bin/llama-quantize \
  ./models/qwen-v2-f16.gguf \
  ./models/qwen-v2-q8_0.gguf \
  q8_0

# Load into Ollama
ollama create de-sql-3b-v2 -f Modelfile.v2
```

---

## Evaluation

### In-Domain (100 prompts)

```bash
# Run against your 3 seeded databases
python eval_ollama.py --model de-sql-3b-v2

# Compare against base model
python eval_ollama.py --model qwen2.5-coder:3b
```

Expected output:

```
============================================================
  de-sql-3b-v2 — Evaluation Report
============================================================
  Schema        Total     OK   Errors  Empty      Acc
  --------------------------------------------------
  ecommerce        35     33        2      0    94.3%
  saas             35     34        1      0    97.1%
  retail           30     24        6      0    80.0%
  --------------------------------------------------
  OVERALL         100     91                    91.0%
============================================================
```

### Spider Benchmark (200 prompts)

```bash
# Download Spider test suite databases (see benchmarks/README.md)
cd benchmarks
python spider_eval.py --model de-sql-3b-v2 --n 200 --output spider_ft_200.json
python spider_eval.py --model qwen2.5-coder:3b --n 200 --output spider_base_200.json
python compare_benchmarks.py
```

---

## Engineering Post-Mortems

### 1. The 4-Bit Quantization Trap

**Symptom:** q4_k_m GGUF produced pure token salad. Execution accuracy: 0%.

**Root Cause:** 4-bit quantization destroys the sharp, high-magnitude weight distributions created by LoRA fine-tuning. The group quantization corrupted embedding lookups, leading to syntax errors and infinite repetition loops (`nesqle_1x_2023_xxx_1_1_1_1...`).

**Fix:** Re-quantized to q8_0 (8-bit). Accuracy restored to 90%.

**Lesson:** For QLoRA-fine-tuned models under 7B parameters, q8_0 is the quantization floor. Do not use q4_k_m.

### 2. CUDA 12.8 Compatibility

**Symptom:** Training on Kaggle T4 ran at 33s/step (CPU fallback) instead of 2s/step (GPU).

**Root Cause:** `bitsandbytes==0.43.1` incompatible with Kaggle's CUDA 12.8 runtime. Silently fell back to CPU.

**Fix:** Upgrade to `bitsandbytes>=0.44.1`.

### 3. TRL API Deprecations

**Symptom:** `DataCollatorForCompletionOnlyLM` import error; `assistant_only_loss` required conversational format.

**Root Cause:** HuggingFace TRL library removed several APIs between 0.10 and 0.12.

**Fix:** Pre-format dataset to a `text` column with ChatML strings, use `dataset_text_field="text"` with `packing=True`.

### 4. VRAM Exhaustion During Merge

**Symptom:** `merge_and_unload()` OOM on Kaggle T4 (16 GB VRAM).

**Fix:** Upload LoRA adapter to HuggingFace from Kaggle, then merge locally on a machine with more RAM.

---

## Roadmap

### Phase 2 — Multi-Agent Agency (Weeks 3-8)

- [ ] Fine-tune 7B validator on 2,000+ failed-prompt RCA labels
- [ ] Complexity router (simple → 3B, medium → 3B + validator, complex → 7B + 3B + validator)
- [ ] Schema RAG — connect to arbitrary databases at inference time
- [ ] FastAPI gateway with retry budget

### Phase 3 — Scale the Dataset (Months 3-6)

- [ ] Expand to 8 schemas using Spider's 140 training databases as fuel
- [ ] Add PostgreSQL and Snowflake dialect support
- [ ] Retrain on 15,000+ rows
- [ ] Full Spider 1,034-prompt benchmark

### Phase 4 — Production Platform (Months 6-12)

- [ ] Dockerized microservices (3B + 7B)
- [ ] React web UI with streaming SQL generation
- [ ] Auto-generated charts from query results
- [ ] Continuous learning pipeline (user feedback → retraining)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{de-sql-3b-v2,
  author       = {Nurahmad},
  title        = {de-swarm: A Multi-Agent Text-to-SQL Distillation Pipeline},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/nurahmad-data/de-swarm}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The fine-tuned model weights are released under the same license as the base Qwen2.5-Coder model (Apache 2.0).

---

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen2.5) for the base model
- [Hugging Face](https://huggingface.co) for PEFT, TRL, and the Hub
- [LangGraph](https://github.com/langchain-ai/langgraph) for the orchestration framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion and quantization
- [Ollama](https://ollama.com) for local inference
- [Spider](https://yale-lily.github.io/spider) for the evaluation benchmark

