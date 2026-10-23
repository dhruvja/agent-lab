"""Harness experiment: which feedback loop lifts open-model pass rates on coding tasks, and at what cost?

Each attempt is up to --turns turns. Every turn is graded on public + hidden tests, so turn 1 is the
single-shot result and the last turn is the harness result, on the same attempt.

Harnesses (--harness):
  examples  reply is code only. Public-example failures go back to the model in the same conversation.
  selftest  reply is tests first, then code. Public-example and own-test failures go back to the model,
            which decides from the spec whether each failing test or the code is wrong.
  resample  same prompt as selftest, but on any failure start a fresh conversation with no feedback.
            selftest vs resample isolates the value of the feedback content from just trying again.

Run:  uv run run.py [--harness examples] [--models a,b] [--turns 3] [--repeats 3] [-k task]
Writes runs/<timestamp>_<harness>_<window>.jsonl, one line per attempt with every turn's response.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from tasks import TASKS, extract_code, extract_tests_and_code, run_checks, run_self_tests

MODELS = ["google/gemma-4-12B-it", "zai-org/GLM-5.3-Flash", "deepseek-ai/DeepSeek-V4.1-Flash", "openai/gpt-oss-120b"]
# $ per 1M tokens (input, cached input, output), from docs.sailresearch.com/pricing on 2026-10-17
PRICES = {
    ("google/gemma-4-12B-it", "asap"): (0.30, 0.15, 2.00),
    ("google/gemma-4-12B-it", "balanced"): (0.10, 0.07, 2.00),
    ("zai-org/GLM-5.3-Flash", "asap"): (0.11, 0.02, 0.35),
    ("zai-org/GLM-5.3-Flash", "balanced"): (0.08, 0.02, 0.28),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "asap"): (0.09, 0.006, 0.30),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "balanced"): (0.07, 0.005, 0.22),
    ("openai/gpt-oss-120b", "asap"): (0.06, 0.03, 0.40),
}
SELFTEST_SUFFIX = (
    "\n\nFirst write tests in a ```tests code block: plain assert statements, at least one for every "
    "requirement in the spec, including edge cases. The function is already defined in the tests' scope; "
    "for an expected exception write `with raises(ValueError): ...`. Then write the solution in a "
    "```python code block. No explanation."
)

parser = argparse.ArgumentParser()
parser.add_argument("--harness", default="examples", choices=["examples", "selftest", "resample"])
parser.add_argument("--models", default=",".join(MODELS))
parser.add_argument("--turns", type=int, default=3)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--window", default="asap", choices=["asap", "balanced"])
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("-k", help="only run tasks whose id contains this string")
args = parser.parse_args()

if not os.environ.get("SAIL_API_KEY"):
    sys.exit("export SAIL_API_KEY first (create one at https://app.sailresearch.com/api-keys)")

client = OpenAI(base_url="https://api.sailresearch.com/v1", api_key=os.environ["SAIL_API_KEY"], max_retries=3)
models = args.models.split(",")
tasks = [t for t in TASKS if not args.k or args.k in t.id]
self_testing = args.harness != "examples"


def attempt(model, task, rep):
    first_prompt = task.body + SELFTEST_SUFFIX if self_testing else task.prompt
    messages = [{"role": "user", "content": first_prompt}]
    turns, error = [], None
    for turn in range(1, args.turns + 1):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort="low",
                max_completion_tokens=12000,
                metadata={"completion_window": args.window},
            )
        except Exception as e:  # keep the turns we have; one failed request should not end the run
            error = repr(e)
            break
        text, u = r.choices[0].message.content or "", r.usage
        truncated = r.choices[0].finish_reason == "length"
        tests, code = extract_tests_and_code(text) if self_testing else (None, extract_code(text))

        public_fails = run_checks(code, task.public, task.forbid)
        if truncated:
            public_fails = ["your reply hit the output token limit before the code was complete"] + public_fails
        n_own, own_fails = 0, []
        if self_testing:
            n_own, own_fails = run_self_tests(code, tests) if tests else (0, ["no ```tests block found"])
        all_fails = run_checks(code, task.public + task.hidden, task.forbid)
        details = getattr(u, "prompt_tokens_details", None)
        turns.append({
            "latency_s": round(time.time() - t0, 2),
            "in": u.prompt_tokens,
            "cached": getattr(details, "cached_tokens", None) or 0,
            "out": u.completion_tokens,
            "truncated": truncated,
            "public_fails": public_fails,
            "own_tests": n_own,
            "own_fails": own_fails,
            "passed": not all_fails,
            "fails": all_fails[:5],
            "response": text,
        })
        if not public_fails and not own_fails:
            break

        if args.harness == "resample":
            messages = [{"role": "user", "content": first_prompt}]
            continue
        feedback = []
        if public_fails:
            feedback += ["These task examples failed (the examples are correct):"] + [f"- {f}" for f in public_fails]
        if own_fails:
            feedback += ["These of your tests failed:"] + [f"- {f}" for f in own_fails[:10]]
            feedback += ["For each failing test, decide from the spec whether the test or the code is wrong."]
        feedback += ["Reply with the corrected tests in a ```tests block and the complete corrected code in a "
                     "```python block." if self_testing else
                     "Reply with the complete corrected code in one Python code block."]
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": "\n".join(feedback)}]

    return {
        "harness": args.harness,
        "model": model,
        "task": task.id,
        "rep": rep,
        "pass_first": bool(turns) and turns[0]["passed"],
        "pass_final": bool(turns) and turns[-1]["passed"],
        # turn-1 code was wrong and some own test failed (not necessarily one that targets the actual bug)
        "own_caught_first": bool(turns) and not turns[0]["passed"] and bool(turns[0]["own_fails"]),
        # turn-1 code passed every hidden test, yet one of the model's own tests failed: the test was wrong
        "own_wrong_first": bool(turns) and turns[0]["passed"] and bool(turns[0]["own_fails"]),
        "n_turns": len(turns),
        "in": sum(t["in"] for t in turns),
        "cached": sum(t["cached"] for t in turns),
        "out": sum(t["out"] for t in turns),
        "truncated_turns": sum(t["truncated"] for t in turns),
        "error": error,
        "turns": turns,
    }


out_path = Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}_{args.harness}_{args.window}.jsonl"
out_path.parent.mkdir(exist_ok=True)
jobs = [(m, t, rep) for m in models for t in tasks for rep in range(args.repeats)]
print(f"{len(jobs)} attempts: {len(models)} models x {len(tasks)} tasks x {args.repeats} repeats, "
      f"harness={args.harness}, up to {args.turns} turns, window={args.window}")

rows = []
with out_path.open("w") as f, ThreadPoolExecutor(args.workers) as pool:
    for fut in as_completed(pool.submit(attempt, *job) for job in jobs):
        row = fut.result()
        rows.append(row)
        f.write(json.dumps(row) + "\n")
        f.flush()
        mark = "PASS" if row["pass_final"] else "FAIL"
        change = (" (fixed)" if row["pass_final"] and not row["pass_first"] else
                  " (BROKEN)" if row["pass_first"] and not row["pass_final"] else "")
        print(f"{mark}  {row['model'].split('/')[-1]:22s} {row['task']:16s} rep {row['rep']}  "
              f"turns={row['n_turns']}  out={row['out']}{change}" + (f"  ERROR {row['error'][:80]}" if row["error"] else ""))

n = len(tasks) * args.repeats
print(f"\nharness={args.harness}")
print("| model | turn 1 | final | fixes | breaks | avg turns | input tok | cached | output tok | cost | cost / solve |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in models:
    rs = [r for r in rows if r["model"] == m]
    first, final = sum(r["pass_first"] for r in rs), sum(r["pass_final"] for r in rs)
    fixes = sum(r["pass_final"] and not r["pass_first"] for r in rs)
    breaks = sum(r["pass_first"] and not r["pass_final"] for r in rs)
    tok_in, cached, out_tok = (sum(r[k] for r in rs) for k in ("in", "cached", "out"))
    p_in, p_cached, p_out = PRICES.get((m, args.window), (float("nan"),) * 3)
    cost = ((tok_in - cached) * p_in + cached * p_cached + out_tok * p_out) / 1e6
    print(f"| {m.split('/')[-1]} | {first}/{n} | {final}/{n} | {fixes} | {breaks} | "
          f"{sum(r['n_turns'] for r in rs) / len(rs):.2f} | {tok_in} | {cached} | {out_tok} | "
          f"${cost:.4f} | {f'${cost / final:.5f}' if final else '-'} |")

print(f"\nper task, all models, out of {len(models) * args.repeats} attempts:")
print("| task | turn 1 wrong | own tests flagged something | own test wrong on correct code | final passes |")
print("|---|---|---|---|---|")
for t in tasks:
    rs = [r for r in rows if r["task"] == t.id]
    print(f"| {t.id} | {sum(not r['pass_first'] for r in rs)} | {sum(r['own_caught_first'] for r in rs)} | "
          f"{sum(r['own_wrong_first'] for r in rs)} | {sum(r['pass_final'] for r in rs)} |")
errors = sum(bool(r["error"]) for r in rows)
truncated = sum(r["truncated_turns"] for r in rows)
print(f"\n{errors} attempts hit an API error, {truncated} turns hit the token limit. responses: {out_path}")
