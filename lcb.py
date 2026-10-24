"""LiveCodeBench problems: loading, prompts, and grading.

Data: the newest LiveCodeBench file (problems from Jan-Apr 2025), fetched with
  uv run lcb.py --download
and stored in data/ (gitignored). `uv run lcb.py` checks the grader against known-good and known-bad code.

Two problem types:
  stdin       (AtCoder)  a full program; each test feeds stdin and compares stdout
  functional  (LeetCode) a `class Solution` method; each test passes JSON arguments and compares the return value
"""
import base64
import io
import json
import pickle
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "lcb_test6.jsonl"
URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"
TIMEOUT_S = 6  # per test, as in LiveCodeBench
PRELUDE = ("from typing import *\nimport sys, math, heapq, bisect, itertools, functools, string, re\n"
           "from collections import *\nfrom functools import *\nfrom itertools import *\nsys.setrecursionlimit(10**6)\n")


@dataclass
class Problem:
    id: str
    title: str
    platform: str
    difficulty: str
    date: str
    content: str
    starter_code: str
    func_name: str | None
    public: list[tuple[str, str]]
    hidden: list[tuple[str, str]]

    @property
    def testtype(self) -> str:
        return "functional" if self.func_name else "stdin"


class _NoGlobals(pickle.Unpickler):
    """Hidden tests are pickled strings. Refuse any class or function reference, so the file cannot run code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def _decode_tests(raw: str) -> list[tuple[str, str]]:
    try:
        cases = json.loads(raw)
    except ValueError:
        cases = json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(raw)))).load())
    return [(c["input"], c["output"]) for c in cases]


def load(path: Path = DATA) -> list[Problem]:
    problems = []
    for line in open(path):
        r = json.loads(line)
        problems.append(Problem(
            id=r["question_id"], title=r["question_title"], platform=r["platform"], difficulty=r["difficulty"],
            date=r["contest_date"][:10], content=r["question_content"], starter_code=r["starter_code"],
            func_name=json.loads(r["metadata"] or "{}").get("func_name"),
            public=_decode_tests(r["public_test_cases"]), hidden=_decode_tests(r["private_test_cases"]),
        ))
    return problems


def select(problems: list[Problem], per_cell: int, seed: int = 0) -> list[Problem]:
    """A fixed sample of `per_cell` problems from each (platform, medium|hard) cell, so reruns use the same set."""
    rng = random.Random(seed)
    chosen = []
    for platform in ("atcoder", "leetcode"):
        for difficulty in ("medium", "hard"):
            cell = sorted((p for p in problems if p.platform == platform and p.difficulty == difficulty), key=lambda p: p.id)
            chosen += rng.sample(cell, min(per_cell, len(cell)))
    return chosen


def prompt(p: Problem) -> str:
    """The problem statement in LiveCodeBench's style, without any reply-format instruction."""
    if p.func_name:
        how = f"Use the following starter code and implement the method:\n```python\n{p.starter_code}\n```"
    else:
        how = "Read the input from stdin, solve the problem, and write the answer to stdout."
    return f"Solve the following competitive programming problem in Python.\n\n{p.content}\n\n{how}"


def case_format(p: Problem) -> str:
    """How one test case is written as JSON, using the problem's first public test as the example."""
    inp, out = p.public[0]
    shape = ("`input` is the method's arguments, one JSON value per line; `output` is the JSON return value"
             if p.func_name else "`input` is the exact stdin text; `output` is the exact expected stdout")
    return f"{shape}. Example: {json.dumps({'input': inp, 'output': out})}"


def _same_stdout(got: str, want: str) -> bool:
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                return False
        except ValueError:
            return False
    return True


FUNCTIONAL_RUNNER = """
import json, signal, sys
sys.path.insert(0, ".")
from solution import *
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
fails = []
for inp, out in json.load(open("cases.json")):
    shown = inp if len(inp) <= 200 else inp[:200] + "..."
    signal.alarm({timeout})
    try:
        got = getattr(Solution(), {func!r})(*[json.loads(a) for a in inp.split("\\n")])
        signal.alarm(0)
    except TimeoutError:
        fails.append(f"input {{shown!r}}: timed out after {timeout}s")
    except Exception as e:
        signal.alarm(0)
        fails.append(f"input {{shown!r}}: raised {{type(e).__name__}}: {{str(e)[:200]}}")
    else:
        got = json.loads(json.dumps(got))
        if got != json.loads(out):
            fails.append(f"input {{shown!r}}: expected {{out[:200]}}, got {{json.dumps(got)[:200]}}")
    if fails and {stop}:
        break
print("\\n" + json.dumps(fails))
"""


def run_tests(code: str, p: Problem, cases: list[tuple[str, str]], stop_at_first: bool = True) -> list[str]:
    """Run code on test cases in an isolated interpreter. Returns failure messages; empty means every case passed."""
    if not cases:
        return []
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(PRELUDE + code)
        if p.func_name:
            Path(d, "cases.json").write_text(json.dumps(cases))
            script = FUNCTIONAL_RUNNER.format(timeout=TIMEOUT_S, func=p.func_name, stop=stop_at_first)
            try:
                r = subprocess.run([sys.executable, "-I", "-c", script], cwd=d, capture_output=True, text=True,
                                   timeout=TIMEOUT_S * len(cases) + 10)
            except subprocess.TimeoutExpired:
                return ["timed out"]
            if r.returncode != 0:
                return [r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit code {r.returncode}"]
            return json.loads(r.stdout.strip().splitlines()[-1])

        fails = []
        for inp, want in cases:
            shown = inp if len(inp) <= 200 else inp[:200] + "..."
            try:
                r = subprocess.run([sys.executable, "-I", "solution.py"], cwd=d, input=inp, capture_output=True,
                                   text=True, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                fails.append(f"input {shown!r}: timed out after {TIMEOUT_S}s")
            else:
                if r.returncode != 0:
                    err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit code {r.returncode}"
                    fails.append(f"input {shown!r}: crashed: {err[:200]}")
                elif not _same_stdout(r.stdout, want):
                    fails.append(f"input {shown!r}: expected {want.strip()[:200]!r}, got {r.stdout.strip()[:200]!r}")
            if fails and stop_at_first:
                break
        return fails


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)\n(.*?)```", text, re.DOTALL) or re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def extract_cases(text: str) -> list[tuple[str, str]] | None:
    """The model's own test cases from its ```json block, or None if it gave none we can read."""
    blocks = re.findall(r"```json\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return None
    try:
        cases = json.loads(blocks[0])
        return [(str(c["input"]), str(c["output"])) for c in cases][:20]
    except (ValueError, KeyError, TypeError):
        return None


if __name__ == "__main__":
    if "--download" in sys.argv:
        DATA.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
        print("downloaded", DATA)
        sys.exit(0)
    problems = load()
    print(f"{len(problems)} problems, {problems[0].date}..{max(p.date for p in problems)}")
    sample = select(problems, per_cell=10)
    print(f"selected {len(sample)}:", {(p.platform, p.difficulty) for p in sample})
    stdin_p = next(p for p in sample if p.testtype == "stdin")
    func_p = next(p for p in sample if p.testtype == "functional")
    # An answer key built from the expected outputs must pass; code that prints nothing must fail.
    echo = "import sys\nANS = " + repr({i: o for i, o in stdin_p.public}) + "\nprint(ANS[sys.stdin.read()], end='')"
    assert run_tests(echo, stdin_p, stdin_p.public) == [], run_tests(echo, stdin_p, stdin_p.public)
    assert run_tests("pass", stdin_p, stdin_p.public)
    lookup = ("import json\nANS = " + repr({i: o for i, o in func_p.public}) + "\nclass Solution:\n"
              f"    def {func_p.func_name}(self, *args):\n"
              "        return json.loads(ANS['\\n'.join(json.dumps(a) for a in args)])")
    print("functional lookup:", run_tests(lookup, func_p, func_p.public)[:1] or "ok")
    assert run_tests("class Solution:\n    pass", func_p, func_p.public)
    print("grader checks ok")
