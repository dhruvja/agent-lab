"""Experiment 2: think longer or verify? Reasoning effort x feedback harness on LiveCodeBench, on Sail's flex window.

Each attempt is up to --turns turns, sent as background Responses requests (flex requires them) and polled.
Every turn is graded on public + hidden tests.
  examples  reply is code only; turn 1 is the single-shot result, later turns feed back public-test failures
  selftest  the model writes its own test cases first, then code; public and own-case failures go back,
            and the model decides from the problem statement whether each failing case or the code is wrong

Run:  uv run run_lcb.py --models Qwen/Qwen3.6-35B-A3B --efforts none,low,high --harness examples [--per-cell 10]
Writes runs/lcb_<timestamp>_<harness>.jsonl, one line per attempt. --resume <file> skips attempts already in it.
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

import lcb

# $ per 1M tokens (input, cached input, output) on flex, from docs.sailresearch.com/pricing on 2026-10-17
FLEX_PRICES = {
    "Qwen/Qwen3.6-35B-A3B": (0.05, 0.02, 0.40),
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.05, 0.004, 0.15),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (0.46, 0.02, 1.39),
    "zai-org/GLM-5.3": (0.40, 0.08, 1.80),
}
EXAMPLES_SUFFIX = "\n\nReply with only one Python code block containing the complete solution."
SELFTEST_SUFFIX = (
    "\n\nFirst write your own test cases in a ```json code block: a JSON list of objects with `input` and "
    "`output`, covering every requirement and edge case in the problem, where {fmt}. Then write the complete "
    "solution in a ```python code block. No explanation."
)

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--efforts", required=True, help="comma list, e.g. none,low,high")
parser.add_argument("--harness", default="examples", choices=["examples", "selftest"])
parser.add_argument("--per-cell", type=int, default=10, help="problems per (platform, medium|hard) cell")
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--turns", type=int, default=3)
parser.add_argument("--max-output", type=int, default=32000)
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--resume")
parser.add_argument("--only-difficulty", choices=["medium", "hard"])
parser.add_argument("--only-truncated-from", help="rerun only the attempts whose turn 1 hit the output cap in this earlier run file")
parser.add_argument("--budget", type=float, default=1.0, help="stop starting new attempts once this run has spent this many USD")
args = parser.parse_args()

if not os.environ.get("SAIL_API_KEY"):
    sys.exit("export SAIL_API_KEY first (create one at https://app.sailresearch.com/api-keys)")
client = OpenAI(base_url="https://api.sailresearch.com/v1", api_key=os.environ["SAIL_API_KEY"], max_retries=5)
problems = [p for p in lcb.select(lcb.load(), per_cell=args.per_cell)
            if not args.only_difficulty or p.difficulty == args.only_difficulty]
spent = 0.0  # USD of finished attempts in this run, updated by the main thread


def call(model, effort, messages):
    """One flex request: submit in the background, poll until it is terminal, return the response."""
    r = client.responses.create(model=model, input=messages, background=True, reasoning={"effort": effort},
                                max_output_tokens=args.max_output, metadata={"completion_window": "flex"})
    delay, deadline = 3, time.time() + 3600
    while r.status not in ("completed", "incomplete", "failed", "cancelled"):
        if time.time() > deadline:
            raise TimeoutError(f"response {r.id} still {r.status} after an hour")
        time.sleep(delay)
        delay = min(delay * 1.5, 20)
        r = client.responses.retrieve(r.id)
    if r.status in ("failed", "cancelled"):
        raise RuntimeError(f"response {r.id} {r.status}: {r.error}")
    return r


def attempt(model, effort, p, rep):
    if spent >= args.budget:
        return None
    suffix = SELFTEST_SUFFIX.format(fmt=lcb.case_format(p)) if args.harness == "selftest" else EXAMPLES_SUFFIX
    messages = [{"role": "user", "content": lcb.prompt(p) + suffix}]
    turns, error = [], None
    for turn in range(1, args.turns + 1):
        t0 = time.time()
        try:
            r = call(model, effort, messages)
        except Exception as e:  # keep the turns we have
            error = repr(e)[:500]
            break
        text, u = r.output_text or "", r.usage
        truncated = r.status == "incomplete"
        code = lcb.extract_code(text)
        public_fails = lcb.run_tests(code, p, p.public)
        if truncated:
            public_fails = ["your reply hit the output token limit before it was complete"] + public_fails
        own_cases, own_fails = None, []
        if args.harness == "selftest":
            own_cases = lcb.extract_cases(text)
            own_fails = (lcb.run_tests(code, p, own_cases, stop_at_first=False)[:8] if own_cases
                         else ["no readable ```json test cases found"])
        hidden_fails = lcb.run_tests(code, p, p.public + p.hidden)
        turns.append({
            "latency_s": round(time.time() - t0, 1),
            "in": u.input_tokens,
            "cached": u.input_tokens_details.cached_tokens or 0,
            "out": u.output_tokens,
            "reasoning": u.output_tokens_details.reasoning_tokens or 0,
            "truncated": truncated,
            "public_fails": public_fails,
            "own_cases": len(own_cases or []),
            "own_fails": own_fails,
            "passed": not hidden_fails,
            "fails": hidden_fails[:2],
            "response": text,
        })
        if not public_fails and not own_fails:
            break
        feedback = []
        if public_fails:
            feedback += ["These public tests from the problem failed (they are correct):"] + [f"- {f}" for f in public_fails]
        if own_fails:
            feedback += ["These of your own test cases failed:"] + [f"- {f}" for f in own_fails]
            feedback += ["For each failing case, decide from the problem statement whether the case or the code is wrong."]
        feedback += ["Reply with corrected test cases in a ```json block and the complete corrected solution in a "
                     "```python block." if args.harness == "selftest" else
                     "Reply with only one Python code block containing the complete corrected solution."]
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": "\n".join(feedback)}]

    p_in, p_cached, p_out = FLEX_PRICES.get(model, (float("nan"),) * 3)
    tok_in, cached, out = (sum(t[k] for t in turns) for k in ("in", "cached", "out"))
    return {
        "model": model, "effort": effort, "harness": args.harness, "problem": p.id, "platform": p.platform,
        "difficulty": p.difficulty, "rep": rep,
        "pass_first": bool(turns) and turns[0]["passed"],
        "pass_final": bool(turns) and turns[-1]["passed"],
        "n_turns": len(turns), "in": tok_in, "cached": cached, "out": out,
        "reasoning": sum(t["reasoning"] for t in turns),
        "cost_usd": ((tok_in - cached) * p_in + cached * p_cached + out * p_out) / 1e6,
        "first_cost_usd": ((turns[0]["in"] - turns[0]["cached"]) * p_in + turns[0]["cached"] * p_cached
                           + turns[0]["out"] * p_out) / 1e6 if turns else 0.0,
        "error": error, "turns": turns,
    }


out_path = Path(args.resume) if args.resume else Path("runs") / f"lcb_{datetime.now():%Y%m%d-%H%M%S}_{args.harness}.jsonl"
out_path.parent.mkdir(exist_ok=True)
done = set()
if out_path.exists():
    for line in open(out_path):
        r = json.loads(line)
        if not r["error"]:
            done.add((r["model"], r["effort"], r["problem"], r["rep"]))
jobs = [(m, e, p, rep) for m in args.models.split(",") for e in args.efforts.split(",") for p in problems
        for rep in range(args.repeats) if (m, e, p.id, rep) not in done]
if args.only_truncated_from:
    earlier = [json.loads(line) for line in open(args.only_truncated_from)]
    capped = {(r["model"], r["effort"], r["problem"], r["rep"]) for r in earlier if r["turns"] and r["turns"][0]["truncated"]}
    jobs = [j for j in jobs if (j[0], j[1], j[2].id, j[3]) in capped]
print(f"{len(jobs)} attempts to run ({len(done)} already done), harness={args.harness}, {len(problems)} problems, "
      f"window=flex -> {out_path}")

rows = []
with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
    for fut in as_completed(pool.submit(attempt, *job) for job in jobs):
        row = fut.result()
        if row is None:
            continue
        rows.append(row)
        spent += row["cost_usd"]
        f.write(json.dumps(row) + "\n")
        f.flush()
        mark = "PASS" if row["pass_final"] else "FAIL"
        print(f"{mark} {row['model'].split('/')[-1]:20s} {row['effort']:6s} {row['problem']:12s} {row['difficulty']:6s} "
              f"turns={row['n_turns']} reasoning={row['reasoning']:6d} ${row['cost_usd']:.4f}"
              + (f"  ERROR {row['error'][:100]}" if row["error"] else ""))

print(f"\n| model | effort | turn 1 | final | median reasoning tok | $ total | $ / solve |")
print("|---|---|---|---|---|---|---|")
for m in args.models.split(","):
    for e in args.efforts.split(","):
        rs = [r for r in rows if r["model"] == m and r["effort"] == e]
        if not rs:
            continue
        final = sum(r["pass_final"] for r in rs)
        cost = sum(r["cost_usd"] for r in rs)
        med = sorted(r["reasoning"] for r in rs)[len(rs) // 2]
        print(f"| {m.split('/')[-1]} | {e} | {sum(r['pass_first'] for r in rs)}/{len(rs)} | {final}/{len(rs)} | "
              f"{med} | ${cost:.3f} | {f'${cost / final:.4f}' if final else '-'} |")
print(f"\nspent ${spent:.3f} of ${args.budget:.2f} budget; {len(jobs) - len(rows)} attempts skipped by the budget")
print(f"{sum(bool(r['error']) for r in rows)} errors, "
      f"{sum(t['truncated'] for r in rows for t in r['turns'])} truncated turns -> {out_path}")
