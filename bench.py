"""Step 1: one prompt, two completion windows, print latency and usage.

Run:  export SAIL_API_KEY=...   then   uv run bench.py
"""
import os
import sys
import time

from openai import OpenAI

if not os.environ.get("SAIL_API_KEY"):
    sys.exit("export SAIL_API_KEY first (create one at https://app.sailresearch.com/api-keys)")

client = OpenAI(base_url="https://api.sailresearch.com/v1", api_key=os.environ["SAIL_API_KEY"])

MODEL = "zai-org/GLM-5.3-Flash"
PROMPT = "Write a Python function that returns the nth Fibonacci number. Only code, no explanation."

for window in ["asap", "balanced"]:
    for attempt in range(2):
        t0 = time.time()
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            reasoning_effort="low",
            metadata={"completion_window": window},
        )
        dt = time.time() - t0
        u = r.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        print(f"{window:9s} try {attempt + 1}  {dt:6.2f}s  in={u.prompt_tokens}  cached={cached}  out={u.completion_tokens}")

print("\nlast answer:\n", r.choices[0].message.content)
