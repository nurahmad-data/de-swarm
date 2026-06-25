# de-swarm

**A multi-agent text-to-SQL distillation pipeline.** Trains a 3B-parameter student model (Qwen2.5-Coder-3B-Instruct) from a 120B+ teacher pipeline using synthetic data, then ships it as a local GGUF runnable via Ollama — all at $0 cost.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-de--sql--3b--v2--gguf-orange)](https://huggingface.co/nurahmad-data/de-sql-3b-v2-gguf)

---

## 📊 Project Status

| Phase | Status | Deliverable |
|---|---|---|
| **Phase 1** — Model training | ✅ Complete | 3B q8_0 GGUF, 90% in-domain, 55.5% Spider |
| **Phase 2** — FastAPI gateway | ✅ Complete | [de-swarm-api](https://github.com/nurahmad-data/de-swarm-api) — 6 endpoints, 5-layer SQL safety |
| **Phase 2.3** — Schema RAG | ✅ Complete | 14 production hardening fixes, self-correction loop, 73 tests |
| **Phase 2.5** — 7B validator | ⏳ Planned | QLoRA fine-tune on failed prompts for window functions |
| **Phase 3** — Spider scaling | ⏳ Planned | 140 schemas, 30-50K rows, target 60-65% Spider EX |
| **Phase 4** — Web UI + deploy | ⏳ Planned | React + VPS deploy for live demo |

---

## 🎯 Final Benchmark Numbers (Phase 1)

### In-Domain (100 held-out prompts, 3 trained schemas)

| Schema | Prompts | Base Qwen2.5-Coder-3B | de-sql-3b-v2 (q8_0) | Lift |
|---|---|---|---|---|
| E-commerce | 35 | 68.6% | **94.3%** | +25.7pp |
| SaaS | 35 | 71.4% | **94.3%** | +22.9pp |
| Retail | 30 | 76.7% | **80.0%** | +3.3pp |
| **Overall** | **100** | **72.0%** | **90.0%** | **+18.0pp** |

### Out-of-Domain (Spider 1.0, 200 prompts, 40+ unseen schemas)

| Metric | Base Qwen2.5-Coder-3B | de-sql-3b-v2 (q8_0) |
|---|---|---|
| Execution Accuracy | 48.5% | **55.5%** |
| Successful Queries | 97 / 200 | **111 / 200** |

### Model Footprint

| Property | Value |
|---|---|
| Base model | Qwen2.5-Coder-3B-Instruct |
| Format | GGUF (q8_0 quantization) |
| Size | 3.3 GB |
| RAM at inference | ~4 GB |
| Hardware target | Any 8 GB laptop (CPU-only) |
| Inference engine | Ollama |
| Training cost | $0 (Kaggle T4 + free-tier APIs) |
| Inference cost | $0 |

---

## 🏗️ Architecture

```
                        TEACHER (cloud, 8 free-tier providers)
                        ─────────────────────────────────────
  augment_prompts.py  ──► 1,226 NL prompts (3 complexity tiers)
  seed_ecommerce.py   ──► memory/ecommerce.db (sandbox, pinned BASE_DATE)
                                  │
                                  ▼
                de-swarm pipeline (multi-provider failover)
                ┌─────────────────────────────────────────┐
                │  retriever  →  architect  →  sql_specialist  →  validator  │
                │  (no LLM)    (planner)     (writer)         (3-layer)      │
                └─────────────────────────────────────────┘
                                  │
                generate_dataset_v3.py (concurrent workers)
                                  │
                                  ▼
                dataset/de-swarm-dataset-v3.jsonl (5,349 validated rows)
                                  │
                score_dataset.py  (executes SQL vs DB)
                                  │
                                  ▼
                dataset/sft_train_full.jsonl (Qwen chat format)
                                  │
                                  ▼
                        STUDENT (Kaggle T4)
                        ────────────────────
                kaggle_sft_train.ipynb
                QLoRA SFT on Qwen2.5-Coder-3B-Instruct
                                  │
                                  ▼
                merged model (fp16, ~6 GB)
                                  │
                                  ▼
                        LOCAL MACHINE
                        ─────────────
                llama.cpp → quantize to q8_0 (3.3 GB)
                                  │
                                  ▼
                Ollama / llama.cpp → eval_ollama.py
                                  │
                                  ▼
                        API GATEWAY (Phase 2)
                        ─────────────────────
                de-swarm-api (separate repo)
                FastAPI + Schema RAG + self-correction
```

---

## 📁 Project Structure

```
de-swarm/
├── config/
│   └── model.py                     ← LLM config + safe_invoke + RateLimiter
├── agents/
│   ├── architect.py                 ← NL→JSON plan (uses safe_invoke)
│   ├── sql_specialist.py            ← plan→SQL (uses safe_invoke)
│   ├── validator.py                 ← 3-layer validation (SKIP_LLM_VALIDATION flag)
│   └── retriever.py                 ← schema fetcher (sqlite default)
├── orchestrator.py                  ← LangGraph state machine (MemorySaver)
├── generate_dataset_v3.py           ← Concurrent runner (ThreadPoolExecutor)
├── seed_ecommerce.py                ← SQLite DB seeder (pinned BASE_DATE)
├── augment_prompts.py               ← 1,226 schema-aligned prompts
├── score_dataset.py                 ← Executes SQL vs DB, scores quality
├── build_sft_dataset.py             ← Builds Qwen chat-format SFT dataset
├── kaggle_sft_train.ipynb           ← QLoRA training notebook (28 cells)
├── data/
│   ├── ecommerce.db
│   ├── saas.db
│   └── retail.db
├── prompts/
│   ├── ecommerce_prompts.txt
│   ├── saas_prompts.txt
│   └── retail_prompts.txt
├── data/final_sft_dataset.jsonl     ← 5,349 training rows
├── eval_ollama.py                   ← In-domain eval script
├── benchmarks/
│   ├── spider_eval.py
│   ├── spider_data/dev.json         ← 1,034 Spider dev examples
│   └── test-suite-sql-eval/         ← Spider databases
├── llama.cpp/                       ← Built — has llama-quantize binary
├── Modelfile.v2                     ← Ollama config (q8_0 ship variant)
└── .env                             ← API keys (DO NOT COMMIT)
```

---

## 🚀 Quick Start

### Pull the trained model

```bash
# Pull the canonical ship model (3.3 GB, q8_0)
ollama pull hf.co/nurahmad-data/de-sql-3b-v2-gguf
ollama cp hf.co/nurahmad-data/de-sql-3b-v2-gguf de-sql-3b-q8

# Test it
ollama run de-sql-3b-q8 "show me top 10 customers by revenue"
```

### Run the API gateway

The trained model is served via a production FastAPI gateway:

```bash
git clone https://github.com/nurahmad-data/de-swarm-api.git
cd de-swarm-api
# Follow the README in de-swarm-api for setup
```

**API repo:** [github.com/nurahmad-data/de-swarm-api](https://github.com/nurahmad-data/de-swarm-api)

### Reproduce the training (optional)

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure env vars
export LLM_PROVIDER=groq
export LLM_API_KEY="your-key"
export RETRIEVER_BACKEND=sqlite
export DB_PATH=memory/ecommerce.db
export SKIP_LLM_VALIDATION=true
export MAX_WORKERS=4

# 3. Generate dataset
python3 seed_ecommerce.py
python3 augment_prompts.py
python3 generate_dataset_v3.py

# 4. Score + build SFT dataset
python3 score_dataset.py dataset/de-swarm-dataset-v3.jsonl
python3 build_sft_dataset.py

# 5. Train on Kaggle
# Upload sft_train_full.jsonl as Kaggle dataset
# Run kaggle_sft_train.ipynb on T4 GPU
```

---

## 🔗 Related Repositories

| Repository | Purpose | Status |
|---|---|---|
| **[de-swarm](https://github.com/nurahmad-data/de-swarm)** (this repo) | Model training pipeline | ✅ Phase 1 complete |
| **[de-swarm-api](https://github.com/nurahmad-data/de-swarm-api)** | FastAPI gateway + Schema RAG | ✅ Phase 2.3 complete |
| **[HuggingFace model](https://huggingface.co/nurahmad-data/de-sql-3b-v2-gguf)** | GGUF distribution | ✅ Published |

---

## 📝 Blog Posts

1. ✅ [How I Distilled a 120B Text-to-SQL Pipeline into a 3B Model](https://medium.com/@nurahmad-data) — Phase 1
2. ✅ [Shipping a Local LLM API with FastAPI + Ollama](https://medium.com/@nurahmad-data) — Phase 2
3. ⏳ Scaling Text-to-SQL Distillation to 140 Schemas — Phase 3 (coming soon)
4. ⏳ Multi-Agent SQL Validation with Local 7B Models — Phase 2.5

---

## 🛠️ Key Architectural Decisions

### 1. Multi-provider failover (8 free-tier APIs)
Single-provider approaches die on rate limits. The 8-provider router (Mistral, OpenRouter, NVIDIA NIM, Cerebras, Groq, GitHub Models, SambaNova, Gemini) kept generation running for weeks.

### 2. 3-layer validation (regex + LLM + EXPLAIN)
Without it, 30%+ of generated rows are garbage. Layer 1 (regex) catches forbidden keywords. Layer 2 (LLM) checks plan alignment. Layer 3 (EXPLAIN QUERY PLAN) validates schema references.

### 3. Execution-level scoring
`score_dataset.py` executes every generated SQL against the real DB. Catches queries that pass Layer 3 but return 0 rows due to hallucinated date boundaries or wrong JOIN types.

### 4. q8_0 quantization (NEVER q4_k_m)
q4_k_m destroys LoRA-tuned weight distributions on 3B models — output becomes token salad. q8_0 (3.3 GB) is the ship variant. FP16 (6.2 GB) is the reference backup.

### 5. Pinned BASE_DATE for reproducibility
`datetime.now()` means "last 30 days" queries return 0 rows if you re-run months later. `BASE_DATE=datetime(2026,6,17)` makes the dataset reproducible forever.

---

## ⚠️ Critical Gotchas

1. **q4_k_m is BROKEN** — never use it for QLoRA-fine-tuned models under 7B. Use q8_0 only.
2. **WSL2 default RAM (4 GB) is too small** — configure `.wslconfig` with 12 GB before running local LLMs.
3. **Spider dev databases are sacred** — never train on them. Only use `train_databases/`.
4. **Ollama model name mismatch** — HF-pulled model is `hf.co/nurahmad-data/de-sql-3b-v2-gguf:latest`. Use `ollama cp` to create a short alias.
5. **Kaggle "Save Version" required for CLI download** — notebooks in `/edit` mode aren't downloadable.

---

## 📋 Phase 3 Roadmap (Next 2-3 Weeks)

**Goal:** Push Spider accuracy from 55.5% → 60-65% by scaling training data across 140 Spider train databases.

| Step | Description | Timeline |
|---|---|---|
| 1. Spider schema extraction | Extract schemas from 140 train DBs | 1 day |
| 2. Lexical gap bridging | 3-tier paraphrased prompts (literal/synonym/indirect) | 2 days |
| 3. Multi-teacher generation | 30-50K rows across 8 free-tier providers | 3-5 days |
| 4. Execution filtering | Drop rows that fail execution or return 0 rows | 1 day |
| 5. Retrain on Kaggle | r=32, alpha=64, LR=1e-4, 2 epochs + early stopping | 1 weekend |
| 6. Quantize + benchmark | q8_0 GGUF, full Spider 1,034-prompt eval | 1 day |
| 7. Publish | New HF model + blog post #3 | 1 day |

**Realistic targets:**
- 60% Spider EX — achievable with data scaling + lexical gap fixes
- 65% Spider EX — achievable with multi-teacher + schema randomization
- 70%+ Spider EX — probably requires a 7B model (Phase 2.5)

Full plan: see [PHASE_3_PLAN.md](https://github.com/nurahmad-data/de-swarm-api/blob/main/PHASE_3_PLAN.md) in the API repo.

---

## 💡 Key Lessons Learned

1. **q4_k_m destroys small LoRA models.** Always use q8_0 for QLoRA-fine-tuned models under 7B.
2. **Multi-provider failover is essential.** Single-provider approaches die on rate limits.
3. **3-layer validation is non-negotiable.** Without it, 30%+ of generated rows are garbage.
4. **The dataset is the moat.** 5,349 validated rows took weeks. The model took a weekend.
5. **Ship imperfect things.** 90% accuracy with a published blog beats 95% accuracy with nothing shipped.
6. **Schema RAG > full schema dump.** Cutting prompt size by 75% eliminated timeouts on large schemas.
7. **Self-correction loops recover 50-70% of failures.** Feeding the SQLite error back to the model is the highest-ROI improvement.

---

## License

MIT — see [LICENSE](LICENSE).
