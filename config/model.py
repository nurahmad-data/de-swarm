"""
Centralized LLM Configuration — Multi-Provider Edition
------------------------------------------------------
Supports: groq | cerebras | mistral | openrouter | chutes | deepinfra | together | google

Switch providers via env vars (NO code changes needed):
  LLM_PROVIDER=cerebras
  LLM_API_KEY=csk-...
  LLM_MODEL=gpt-oss-120b    (optional — uses provider default if unset)
"""
import os
import time
import random
import logging
import threading
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

_PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "default_rpm": 28,
        "default_min_spacing": 2.5,
        "signup_url": "console.groq.com",
        "notes": "Free tier: 30 RPM, 100K TPD. ~28 prompts/day. Smoke tests only.",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": "gpt-oss-120b",
        "default_rpm": 3,
        "default_min_spacing": 20,
        "signup_url": "cloud.cerebras.ai",
        "notes": "Free tier: 5 RPM, 1M TPD, 2400 RPD. gpt-oss-120b reasoning model. ~200 prompts/day.",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "default_rpm": 4,
        "default_min_spacing": 12,
        "signup_url": "console.mistral.ai",
        "notes": "Free tier: 4 RPM, 250K TPM. ~100 prompts/hour. No daily TPD cap.",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "default_rpm": 20,
        "default_min_spacing": 3,
        "signup_url": "openrouter.ai",
        "notes": "Free model variants. ~20 RPM, 1000 RPD. Llama 3.3 70B free variant available.",
    },
    "chutes": {
        "base_url": "https://api.chutes.ai/v1",
        "default_model": "chutes-deepseek-v3",
        "default_rpm": 10,
        "default_min_spacing": 6,
        "signup_url": "chutes.ai",
        "notes": "Free tier with rate limits. Various models available.",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
        "default_rpm": 60,
        "default_min_spacing": 1.0,
        "signup_url": "deepinfra.com",
        "notes": "$0.5 free credit. Pay-per-token after. No daily caps. Fast inference.",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "Meta-Llama-3.3-70B-Instruct-Turbo",
        "default_rpm": 60,
        "default_min_spacing": 1.0,
        "signup_url": "api.together.xyz",
        "notes": "$5 free credit (expires 30 days). Best Llama 3.3 70B inference speed.",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash-exp",
        "default_rpm": 10,
        "default_min_spacing": 6,
        "signup_url": "aistudio.google.com",
        "notes": "Free tier: 1500 RPD, 2 RPM on Gemini Flash. ~480 prompts/day.",
    },
    "github": {
    	"base_url": "https://models.inference.ai.azure.com/v1",
    	"default_model": "Llama-3.3-70B-Instruct",
    	"default_rpm": 30,
    	"default_min_spacing": 2.5,
    	"signup_url": "github.com/settings/tokens",
    	"notes": "Free with GitHub account. ~50 RPD per model. Stable (Microsoft-backed).",
    },
    "sambanova": {
    	"base_url": "https://api.sambanova.ai/v1",
    	"default_model": "Meta-Llama-3.3-70B-Instruct",
    	"default_rpm": 30,
    	"default_min_spacing": 2.5,
    	"signup_url": "sambanova.ai",
    	"notes": "Free tier, fast SN40L inference. Llama 3.3 70B + 3.1 405B.",
    },
        "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama-3.3-70b-instruct",
        "default_rpm": 40,
        "default_min_spacing": 1.5,
        "signup_url": "build.nvidia.com",
        "notes": "Free credits on signup. Llama 3.3 70B + Nemotron. Fast inference.",
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
        "default_rpm": 5,
        "default_min_spacing": 12,
        "signup_url": "huggingface.co/settings/tokens",
        "notes": "Free inference API. Slow + queued. Use only as fallback.",
    },
    "cloudflare": {
        "base_url": "https://api.cloudflare.com/client/v4/accounts",
        "default_model": "@cf/meta/llama-3.1-8b-instruct",
        "default_rpm": 50,
        "default_min_spacing": 1.2,
        "signup_url": "dash.cloudflare.com",
        "notes": "Workers AI free tier: 10K neurons/day. Llama 3.1 8B (fast).",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "default_rpm": 60,
        "default_min_spacing": 1.0,
        "signup_url": "platform.deepseek.com",
        "notes": "DeepSeek-V3 reasoning model. Cheap pay-per-token.",
    },
}

if LLM_PROVIDER not in _PROVIDER_CONFIG:
    raise ValueError(
        f"LLM_PROVIDER={LLM_PROVIDER!r} not supported. "
        f"Use one of: {list(_PROVIDER_CONFIG.keys())}"
    )

_cfg = _PROVIDER_CONFIG[LLM_PROVIDER]

API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv(f"{LLM_PROVIDER.upper()}_API_KEY")
    or os.getenv("GROQ_API_KEY")
)

if not API_KEY:
    raise ValueError(
        f"No API key found. Set LLM_API_KEY (or {LLM_PROVIDER.upper()}_API_KEY) in .env. "
        f"Get a {LLM_PROVIDER} key at https://{_cfg['signup_url']}"
    )

MODEL_NAME = os.getenv("LLM_MODEL") or _cfg["default_model"]
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))
TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT", "60"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

from langchain_openai import ChatOpenAI

_llm_kwargs = dict(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    api_key=API_KEY,
    max_tokens=MAX_TOKENS,
    max_retries=MAX_RETRIES,
    timeout=TIMEOUT_SECONDS,
)
if _cfg["base_url"]:
    _llm_kwargs["base_url"] = _cfg["base_url"]

llm_pipeline = ChatOpenAI(**_llm_kwargs)

print(f"[config.model] Initialized {LLM_PROVIDER} LLM -> model='{MODEL_NAME}' "
      f"| base_url={_cfg['base_url'] or 'default'} "
      f"| temp={TEMPERATURE} | max_tokens={MAX_TOKENS} "
      f"| max_retries={MAX_RETRIES} | timeout={TIMEOUT_SECONDS}s")
print(f"[config.model] Provider notes: {_cfg['notes']}")


class RateLimiter:
    def __init__(self, max_calls_per_minute: int, min_spacing_s: float = 2.5):
        self.max = max_calls_per_minute
        self.min_spacing = min_spacing_s
        self.calls: deque = deque()
        self.lock = threading.Lock()
        log.info("RateLimiter initialized | rpm=%d | min_spacing=%.1fs",
                 max_calls_per_minute, min_spacing_s)

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.calls and self.calls[0] < now - 60:
                    self.calls.popleft()
                if self.calls:
                    elapsed = now - self.calls[-1]
                    spacing_wait = max(0, self.min_spacing - elapsed)
                else:
                    spacing_wait = 0
                if spacing_wait > 0:
                    pass
                elif len(self.calls) < self.max:
                    self.calls.append(now)
                    return
                else:
                    spacing_wait = 60 - (now - self.calls[0]) + 0.1
            time.sleep(min(spacing_wait, 5.0))


groq_rate_limiter = RateLimiter(
    max_calls_per_minute=int(os.getenv("LLM_RPM", str(_cfg["default_rpm"]))),
    min_spacing_s=float(os.getenv("LLM_MIN_SPACING_S", str(_cfg["default_min_spacing"]))),
)


def safe_invoke(llm: Any, messages: list) -> Any:
    max_attempts = MAX_RETRIES + 2
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        groq_rate_limiter.acquire()
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            base = min(2 ** attempt, 60)
            wait = base + random.uniform(0, 2)
            log.warning("LLM invoke failed (attempt %d/%d): %s — backing off %.1fs",
                        attempt, max_attempts, type(exc).__name__, wait)
            if attempt < max_attempts:
                time.sleep(wait)
    assert last_exc is not None
    raise last_exc
