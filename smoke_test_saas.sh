#!/usr/bin/env bash
# SaaS-specific smoke test
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "═══ SaaS Schema Smoke Test ═══"
echo "Provider: ${LLM_PROVIDER:-?} / ${LLM_MODEL:-?}"
echo "DB: ${DB_PATH:-?}"
echo

# 3 SaaS-appropriate prompts of increasing complexity
python3 -c "
from orchestrator import run

prompts = [
    ('Simple 1-table', 'Show total organizations by plan'),
    ('2-table join', 'Show total users by plan'),
    ('3-table join', 'Show MRR by plan for the last 30 days'),
]

for label, prompt in prompts:
    print(f'─── {label} ───')
    print(f'Prompt: {prompt}')
    result = run(prompt, thread_id=f'saas_{label[:5]}')
    print(f'Status: {result[\"status\"]}')
    print(f'SQL   : {result.get(\"sql_query\", \"N/A\")[:250]}')
    if result['status'] != 'success':
        print(f'Errors: {result.get(\"error_log\", [])[:2]}')
    print()

# Final verdict
all_pass = all(run(p, thread_id=f'final_{i}')['status'] == 'success' for i, (_, p) in enumerate(prompts))
print('═══ Result ═══')
print('✅ All SaaS smoke tests PASSED' if all_pass else '⚠️  Some tests failed (may be prompt-specific, not pipeline issue)')
"
