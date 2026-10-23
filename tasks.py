"""Task set: coding tasks with public examples and hidden tests.

The model sees the spec and the public examples. The harness may feed back
public-example failures. Grading always runs public + hidden tests, so a model
cannot pass by fitting the examples it was shown.

Self-test (no API calls): `uv run tasks.py` checks that every reference passes
everything and that missing code fails.
"""
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

RAISES = "__raises__"


@dataclass
class Task:
    id: str
    spec: str
    public: list[tuple[str, object]]
    hidden: list[tuple[str, object]]
    reference: str
    forbid: list[str] = field(default_factory=list)  # modules or builtins the solution may not use

    @property
    def body(self) -> str:
        """The spec plus the public examples, without any reply-format instruction."""
        examples = "\n".join(
            f"  {expr}  raises {want[1]}" if isinstance(want, tuple) else f"  {expr} == {want!r}"
            for expr, want in self.public
        )
        return f"{self.spec}\n\nExamples:\n{examples}"

    @property
    def prompt(self) -> str:
        return f"{self.body}\n\nReply with only one Python code block. No explanation."


def call(fn: str, *args) -> str:
    return f"{fn}({', '.join(map(repr, args))})"


def raises(name: str = "ValueError") -> tuple[str, str]:
    return (RAISES, name)


def extract_code(text: str) -> str:
    """Return the last ``` block (models put the final version last), or the text itself."""
    blocks = re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def extract_tests_and_code(text: str) -> tuple[str | None, str]:
    """Split a self-test reply into its ```tests block and its ```python block.

    Falls back to first block = tests, last block = code when the model ignores the tags.
    """
    blocks = re.findall(r"```([a-zA-Z0-9]*)\n(.*?)```", text, re.DOTALL)
    tests = [body for lang, body in blocks if lang == "tests"]
    code = [body for lang, body in blocks if lang in ("python", "py")]
    if tests and code:
        return tests[-1], code[-1]
    if len(blocks) >= 2:
        return blocks[0][1], blocks[-1][1]
    return None, extract_code(text)


# Models often write `from pytest import raises` despite being told raises exists, so give them a stand-in.
PYTEST_SHIM = """
import re

class raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match
    def __enter__(self):
        return self
    def __exit__(self, kind, value, tb):
        if kind is None:
            raise AssertionError(f"expected {getattr(self.exc, '__name__', self.exc)} to be raised")
        if not issubclass(kind, self.exc):
            return False
        if self.match and not re.search(self.match, str(value)):
            raise AssertionError(f"exception message did not match {self.match!r}")
        return True
"""

SELF_TEST_SCRIPT = """
import ast, json, sys
sys.path.insert(0, ".")
from solution import *
from pytest import raises

_n, _fails = 0, []
for _node in ast.parse(open("tests.py").read()).body:
    _src = ast.unparse(_node)
    _n += isinstance(_node, (ast.Assert, ast.With))
    try:
        exec(_src, globals())
    except AssertionError as _e:
        _detail = str(_e)
        _t = getattr(_node, "test", None)
        if isinstance(_t, ast.Compare) and len(_t.ops) == 1 and isinstance(_t.ops[0], ast.Eq):
            try:
                _detail = f"left side was {eval(ast.unparse(_t.left))!r}"
            except Exception:
                pass
        _fails.append(f"{_src}  ({_detail})" if _detail else _src)
    except Exception as _e:
        _fails.append(f"{_src}  raised {type(_e).__name__}: {_e}")
for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        _n += 1
        try:
            _fn()
        except AssertionError as _e:
            _fails.append(f"{_name}() failed" + (f": {_e}" if str(_e) else ""))
        except Exception as _e:
            _fails.append(f"{_name}() raised {type(_e).__name__}: {_e}")
print("\\n" + json.dumps({"n": _n, "fails": _fails}))
"""


def run_self_tests(code: str, tests: str, timeout: float = 10.0) -> tuple[int, list[str]]:
    """Run the model's own assert-style tests against its code. Returns (number of tests, failure messages)."""
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/solution.py", "w") as f:
            f.write(code)
        with open(f"{d}/tests.py", "w") as f:
            f.write(tests)
        with open(f"{d}/pytest.py", "w") as f:
            f.write(PYTEST_SHIM)
        try:
            r = subprocess.run([sys.executable, "-I", "-c", SELF_TEST_SCRIPT],
                               cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0, [f"your tests timed out after {timeout:.0f}s"]
    if r.returncode != 0:
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit code {r.returncode}"
        return 0, [f"your tests could not run: {last}"]
    result = json.loads(r.stdout.strip().splitlines()[-1])
    return result["n"], result["fails"]


CHECK_SCRIPT = """
import json, sys
sys.path.insert(0, ".")
from solution import *
_fails = []
for _expr, _want in {checks!r}:
    try:
        _got = eval(_expr)
    except Exception as _e:
        if isinstance(_want, tuple) and _want[1] in [c.__name__ for c in type(_e).__mro__]:
            continue
        _fails.append(f"{{_expr}} raised {{type(_e).__name__}}: {{_e}}")
        continue
    if isinstance(_want, tuple):
        _fails.append(f"{{_expr}} returned {{_got!r}}, expected it to raise {{_want[1]}}")
    elif _got != _want:
        _fails.append(f"{{_expr}} returned {{_got!r}}, expected {{_want!r}}")
print("\\n" + json.dumps(_fails))
"""


def run_checks(code: str, checks: list[tuple[str, object]], forbid: list[str] = (), timeout: float = 10.0) -> list[str]:
    """Run checks against code in an isolated interpreter. Returns failure messages; empty means pass."""
    for name in forbid:
        if re.search(rf"^\s*(import|from)\s+{name}\b|(?<![\w.]){name}\s*\(", code, re.MULTILINE):
            return [f"the solution must not use {name}"]
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/solution.py", "w") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, "-I", "-c", CHECK_SCRIPT.format(checks=checks)],
                               cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [f"timed out after {timeout:.0f}s"]
    if r.returncode != 0:
        return [r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit code {r.returncode}"]
    return json.loads(r.stdout.strip().splitlines()[-1])


TASKS = [
    Task(
        "semver",
        "Write `semver_compare(a: str, b: str) -> int` that returns -1, 0 or 1 when version a has lower, "
        "equal or higher precedence than version b under Semantic Versioning 2.0.0.",
        public=[(call("semver_compare", "1.0.0", "2.0.0"), -1),
                (call("semver_compare", "1.10.0", "1.9.0"), 1),
                (call("semver_compare", "1.0.0-alpha", "1.0.0"), -1)],
        hidden=[(call("semver_compare", "1.0.0-alpha", "1.0.0-alpha.1"), -1),
                (call("semver_compare", "1.0.0-alpha.1", "1.0.0-alpha.beta"), -1),
                (call("semver_compare", "1.0.0-alpha.beta", "1.0.0-beta"), -1),
                (call("semver_compare", "1.0.0-beta.2", "1.0.0-beta.11"), -1),
                (call("semver_compare", "1.0.0-beta.11", "1.0.0-rc.1"), -1),
                (call("semver_compare", "1.0.0", "1.0.0-rc.1"), 1),
                (call("semver_compare", "1.0.0-alpha.10", "1.0.0-alpha.9"), 1),
                (call("semver_compare", "1.0.0-1", "1.0.0-a"), -1),
                (call("semver_compare", "1.0.0+build.5", "1.0.0+build.9"), 0),
                (call("semver_compare", "1.0.0-alpha+x", "1.0.0-alpha"), 0),
                (call("semver_compare", "2.0.0", "2.0.0"), 0)],
        reference="""def semver_compare(a, b):
    def parse(v):
        core, _, pre = v.split("+")[0].partition("-")
        return [int(x) for x in core.split(".")], pre

    (na, pa), (nb, pb) = parse(a), parse(b)
    if na != nb:
        return -1 if na < nb else 1
    if pa == pb:
        return 0
    if not pa or not pb:
        return 1 if not pa else -1
    xs, ys = pa.split("."), pb.split(".")
    for x, y in zip(xs, ys):
        if x == y:
            continue
        if x.isdigit() and y.isdigit():
            return -1 if int(x) < int(y) else 1
        if x.isdigit() or y.isdigit():
            return -1 if x.isdigit() else 1
        return -1 if x < y else 1
    return -1 if len(xs) < len(ys) else 1
""",
    ),
    Task(
        "wrap_text",
        "Write `wrap_text(text: str, width: int) -> list[str]` that word-wraps text greedily. Words are separated "
        "by any whitespace; lines join words with single spaces and never start or end with a space. Put as many "
        "words on each line as fit within `width` characters. A word longer than `width` starts on a new line and "
        "is cut into pieces of exactly `width` characters (the last piece may be shorter); following words may "
        "continue on the line of its last piece. Text with no words returns [].",
        public=[(call("wrap_text", "the quick brown fox", 10), ["the quick", "brown fox"]),
                (call("wrap_text", "", 5), []),
                (call("wrap_text", "abcdefghij", 4), ["abcd", "efgh", "ij"])],
        hidden=[(call("wrap_text", "  a   b  ", 3), ["a b"]),
                (call("wrap_text", "aaa bbb", 3), ["aaa", "bbb"]),
                (call("wrap_text", "one two three", 7), ["one two", "three"]),
                (call("wrap_text", "hi abcdefghij k", 4), ["hi", "abcd", "efgh", "ij k"]),
                (call("wrap_text", "hi abcdefghij ok", 4), ["hi", "abcd", "efgh", "ij", "ok"]),
                (call("wrap_text", "a\nb\tc", 5), ["a b c"]),
                (call("wrap_text", "   ", 4), []),
                (call("wrap_text", "ab c", 1), ["a", "b", "c"]),
                (call("wrap_text", "exactly ten", 11), ["exactly ten"])],
        reference="""def wrap_text(text, width):
    lines, cur = [], ""
    for w in text.split():
        if len(w) > width:
            if cur:
                lines.append(cur)
            pieces = [w[i:i + width] for i in range(0, len(w), width)]
            lines += pieces[:-1]
            cur = pieces[-1]
        elif not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
""",
    ),
    Task(
        "csv_line",
        "Write `parse_csv_line(line: str) -> list[str]` that splits one CSV line into fields separated by commas. "
        "A field that starts with a double quote is quoted: it runs to the matching closing quote, may contain "
        'commas, and "" inside it stands for one literal quote. Any other field is taken exactly as written, '
        "including spaces and quote characters. An empty line is one empty field. Raise ValueError if a quoted "
        "field is never closed, or if its closing quote is followed by anything other than a comma or the end "
        "of the line. Do not use the csv module.",
        public=[(call("parse_csv_line", "a,b,c"), ["a", "b", "c"]),
                (call("parse_csv_line", '"x,y",z'), ["x,y", "z"]),
                (call("parse_csv_line", ""), [""])],
        hidden=[(call("parse_csv_line", "a,,c,"), ["a", "", "c", ""]),
                (call("parse_csv_line", '"say ""hi""",2'), ['say "hi"', "2"]),
                (call("parse_csv_line", " a , b "), [" a ", " b "]),
                (call("parse_csv_line", '""'), [""]),
                (call("parse_csv_line", '"",""'), ["", ""]),
                (call("parse_csv_line", ","), ["", ""]),
                (call("parse_csv_line", 'a"b,c'), ['a"b', "c"]),
                (call("parse_csv_line", '"abc'), raises()),
                (call("parse_csv_line", '"a"b,c'), raises())],
        forbid=["csv"],
        reference="""def parse_csv_line(line):
    fields, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise ValueError("unclosed quote")
                if line[i] == '"' and i + 1 < n and line[i + 1] == '"':
                    buf.append('"')
                    i += 2
                elif line[i] == '"':
                    i += 1
                    break
                else:
                    buf.append(line[i])
                    i += 1
            if i < n and line[i] != ",":
                raise ValueError("text after closing quote")
            fields.append("".join(buf))
        else:
            j = line.find(",", i)
            j = n if j == -1 else j
            fields.append(line[i:j])
            i = j
        if i >= n:
            return fields
        i += 1
""",
    ),
    Task(
        "glob_match",
        "Write `glob_match(pattern: str, s: str) -> bool` that returns True if the whole of s matches pattern. "
        "`?` matches any one character, `*` matches any sequence including the empty one, `[abc]` matches one of "
        "the listed characters, `[a-z]` one character in the range, and `[!...]` one character not in the set. "
        "Every other character matches itself, case-sensitively. Do not use the fnmatch, glob or re modules.",
        public=[(call("glob_match", "*.py", "main.py"), True),
                (call("glob_match", "a?c", "abc"), True),
                (call("glob_match", "[a-c]x", "dx"), False)],
        hidden=[(call("glob_match", "*", ""), True),
                (call("glob_match", "?", ""), False),
                (call("glob_match", "a*b*c", "aXXbYYc"), True),
                (call("glob_match", "a*b*c", "aXXbYY"), False),
                (call("glob_match", "[!0-9]*", "x1"), True),
                (call("glob_match", "[!0-9]*", "1x"), False),
                (call("glob_match", "[abc]", "b"), True),
                (call("glob_match", "*.txt", "notes.txt.bak"), False),
                (call("glob_match", "**a", "bba"), True),
                (call("glob_match", "a", "A"), False),
                (call("glob_match", "[a-z][0-9]?", "q7!"), True),
                (call("glob_match", "*x*", "abc"), False)],
        forbid=["fnmatch", "glob", "re"],
        reference="""from functools import cache

def glob_match(pattern, s):
    def match_class(i, c):
        j = i + 1
        negate = pattern[j] == "!"
        j += negate
        found = False
        while pattern[j] != "]":
            if j + 2 < len(pattern) and pattern[j + 1] == "-" and pattern[j + 2] != "]":
                found |= pattern[j] <= c <= pattern[j + 2]
                j += 3
            else:
                found |= pattern[j] == c
                j += 1
        return found != negate, j + 1

    @cache
    def m(pi, si):
        if pi == len(pattern):
            return si == len(s)
        ch = pattern[pi]
        if ch == "*":
            return m(pi + 1, si) or (si < len(s) and m(pi, si + 1))
        if si == len(s):
            return False
        if ch == "[":
            ok, nxt = match_class(pi, s[si])
            return ok and m(nxt, si + 1)
        return (ch == "?" or ch == s[si]) and m(pi + 1, si + 1)

    return m(0, 0)
""",
    ),
    Task(
        "eval_expr",
        "Write `eval_expr(s: str) -> int` that evaluates an integer expression with +, -, *, /, parentheses, "
        "unary minus and arbitrary spaces. * and / bind tighter than + and -, operators of equal precedence apply "
        "left to right, and / is integer division that truncates toward zero. Do not use eval, exec or compile.",
        public=[(call("eval_expr", "1 + 2 * 3"), 7),
                (call("eval_expr", "(1 + 2) * 3"), 9),
                (call("eval_expr", "7 / 2"), 3)],
        hidden=[(call("eval_expr", "-7 / 2"), -3),
                (call("eval_expr", "7 / -2"), -3),
                (call("eval_expr", "2 * -3"), -6),
                (call("eval_expr", "-(2 + 3) * 2"), -10),
                (call("eval_expr", "1 - -1"), 2),
                (call("eval_expr", "10 - 4 - 3"), 3),
                (call("eval_expr", "100 / 10 / 5"), 2),
                (call("eval_expr", "  42 "), 42),
                (call("eval_expr", "2*(3+(4-1))*2"), 24),
                (call("eval_expr", "--3"), 3),
                (call("eval_expr", "-3 * -3 - -3"), 12)],
        forbid=["eval", "exec", "compile"],
        reference="""def eval_expr(s):
    toks, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            toks.append(int(s[i:j]))
            i = j
        else:
            toks.append(s[i])
            i += 1
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def expr():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v

    def term():
        v = unary()
        while peek() in ("*", "/"):
            if take() == "*":
                v *= unary()
            else:
                d = unary()
                q = abs(v) // abs(d)
                v = q if (v >= 0) == (d > 0) else -q
        return v

    def unary():
        if peek() == "-":
            take()
            return -unary()
        if peek() == "(":
            take()
            v = expr()
            take()
            return v
        return take()

    return expr()
""",
    ),
    Task(
        "normalize_path",
        "Write `normalize_path(path: str) -> str` that normalizes a Unix path as a pure string operation: collapse "
        "repeated slashes, drop '.' segments, and let '..' remove the previous segment. An absolute path stays "
        "absolute, and '..' at the root is dropped. A relative path keeps leading '..' segments that cannot be "
        "resolved. No trailing slash except for the root '/'. A relative path that resolves to nothing is '.'.",
        public=[(call("normalize_path", "/a/./b/../c/"), "/a/c"),
                (call("normalize_path", "a/../.."), ".."),
                (call("normalize_path", ""), ".")],
        hidden=[(call("normalize_path", "//a//b"), "/a/b"),
                (call("normalize_path", "/../x"), "/x"),
                (call("normalize_path", "/"), "/"),
                (call("normalize_path", "///"), "/"),
                (call("normalize_path", "a/b/../../.."), ".."),
                (call("normalize_path", "../../a/./b"), "../../a/b"),
                (call("normalize_path", "a/.."), "."),
                (call("normalize_path", "./"), "."),
                (call("normalize_path", "/a/b/../../.."), "/"),
                (call("normalize_path", "a/.../b"), "a/.../b"),
                (call("normalize_path", "../a/../.."), "../..")],
        reference="""def normalize_path(path):
    absolute = path.startswith("/")
    parts = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg != "..":
            parts.append(seg)
        elif parts and parts[-1] != "..":
            parts.pop()
        elif not absolute:
            parts.append("..")
    if absolute:
        return "/" + "/".join(parts)
    return "/".join(parts) or "."
""",
    ),
    Task(
        "roman_strict",
        "Write `from_roman(s: str) -> int` that converts a Roman numeral for a number from 1 to 3999 into an "
        "integer. Accept only the standard form: uppercase I, V, X, L, C, D, M, subtractive pairs limited to IV, "
        "IX, XL, XC, CD and CM, and the shortest spelling of the number. Raise ValueError for anything else.",
        public=[(call("from_roman", "XIV"), 14),
                (call("from_roman", "MCMXCIV"), 1994),
                (call("from_roman", "IIII"), raises())],
        hidden=[(call("from_roman", "I"), 1),
                (call("from_roman", "XLIX"), 49),
                (call("from_roman", "CDXLIV"), 444),
                (call("from_roman", "MMXXVI"), 2026),
                (call("from_roman", "MMMCMXCIX"), 3999),
                *[(call("from_roman", bad), raises())
                  for bad in ["", "IC", "VX", "IL", "XM", "VV", "MMMM", "IIV", "XIIII", "iv", "IXI", "XCX", "CMCM"]]],
        reference="""def from_roman(s):
    symbols = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
               (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    total, i = 0, 0
    for value, sym in symbols:
        while s.startswith(sym, i):
            total += value
            i += len(sym)
    canonical, n = "", total
    for value, sym in symbols:
        while n >= value:
            canonical += sym
            n -= value
    if i != len(s) or not 1 <= total <= 3999 or canonical != s:
        raise ValueError(s)
    return total
""",
    ),
    Task(
        "format_duration",
        "Write `format_duration(seconds: int) -> str` that renders a non-negative number of seconds in words using "
        "years (365 days), days, hours, minutes and seconds. Omit zero units, use the singular for 1 and the plural "
        "otherwise, join the last two units with ' and ' and earlier ones with ', '. Zero is 'now'.",
        public=[(call("format_duration", 62), "1 minute and 2 seconds"),
                (call("format_duration", 0), "now"),
                (call("format_duration", 3600), "1 hour")],
        hidden=[(call("format_duration", 1), "1 second"),
                (call("format_duration", 120), "2 minutes"),
                (call("format_duration", 3662), "1 hour, 1 minute and 2 seconds"),
                (call("format_duration", 86399), "23 hours, 59 minutes and 59 seconds"),
                (call("format_duration", 90061), "1 day, 1 hour, 1 minute and 1 second"),
                (call("format_duration", 31536001), "1 year and 1 second"),
                (call("format_duration", 63338400), "2 years, 3 days and 2 hours")],
        reference="""def format_duration(seconds):
    if seconds == 0:
        return "now"
    units = [("year", 31536000), ("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for name, size in units:
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{n} {name}" + ("s" if n > 1 else ""))
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
""",
    ),
    Task(
        "topo_sort",
        "Write `topo_sort(deps: dict[str, list[str]]) -> list[str]` where deps maps a node to the nodes it depends "
        "on. Return every node, including nodes that only appear as dependencies, ordered so each node comes after "
        "all of its dependencies. Whenever several nodes are ready, take the alphabetically smallest first. Raise "
        "ValueError if there is a cycle.",
        public=[(call("topo_sort", {"b": ["a"], "a": []}), ["a", "b"]),
                (call("topo_sort", {"c": ["a", "b"]}), ["a", "b", "c"]),
                (call("topo_sort", {"a": ["b"], "b": ["a"]}), raises())],
        hidden=[(call("topo_sort", {}), []),
                (call("topo_sort", {"x": []}), ["x"]),
                (call("topo_sort", {"d": ["b"], "c": ["a"], "b": ["a"]}), ["a", "b", "c", "d"]),
                (call("topo_sort", {"z": [], "a": ["z"]}), ["z", "a"]),
                (call("topo_sort", {"b": [], "a": ["c"], "c": []}), ["b", "c", "a"]),
                (call("topo_sort", {"m": ["k", "k"]}), ["k", "m"]),
                (call("topo_sort", {"a": ["a"]}), raises()),
                (call("topo_sort", {"a": ["b"], "b": ["c"], "c": ["a"], "d": []}), raises())],
        reference="""import heapq

def topo_sort(deps):
    nodes = set(deps) | {d for ds in deps.values() for d in ds}
    waiting = {n: set(deps.get(n, [])) for n in nodes}
    ready = [n for n in nodes if not waiting[n]]
    heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for m in nodes:
            if n in waiting[m]:
                waiting[m].discard(n)
                if not waiting[m]:
                    heapq.heappush(ready, m)
    if len(order) != len(nodes):
        raise ValueError("cycle")
    return order
""",
    ),
    Task(
        "ledger",
        "Write `apply_transfers(balances: dict[str, int], transfers: list[dict]) -> dict` that applies transfers in "
        'order. Each transfer is {"id": str, "from": str, "to": str, "amount": int}. Reject a transfer, leaving '
        "balances unchanged, if either account does not exist, both accounts are the same, the amount is not a "
        "positive int (bools do not count as ints), or the sender's balance would drop below zero. Do not modify the input. Return "
        '{"balances": final balances, "rejected": ids of rejected transfers in order}.',
        public=[(call("apply_transfers", {"a": 10, "b": 0}, [{"id": "t1", "from": "a", "to": "b", "amount": 4}]),
                 {"balances": {"a": 6, "b": 4}, "rejected": []}),
                (call("apply_transfers", {"a": 10, "b": 0}, [{"id": "t1", "from": "a", "to": "b", "amount": 11}]),
                 {"balances": {"a": 10, "b": 0}, "rejected": ["t1"]})],
        hidden=[(call("apply_transfers", {"a": 5, "b": 0}, [{"id": "t1", "from": "a", "to": "b", "amount": 5},
                                                             {"id": "t2", "from": "b", "to": "a", "amount": 3}]),
                 {"balances": {"a": 3, "b": 2}, "rejected": []}),
                (call("apply_transfers", {"a": 10, "b": 0, "c": 0},
                      [{"id": "t1", "from": "a", "to": "b", "amount": 6},
                       {"id": "t2", "from": "a", "to": "c", "amount": 5},
                       {"id": "t3", "from": "b", "to": "c", "amount": 6},
                       {"id": "t4", "from": "c", "to": "a", "amount": 0},
                       {"id": "t5", "from": "x", "to": "a", "amount": 1},
                       {"id": "t6", "from": "a", "to": "a", "amount": 1}]),
                 {"balances": {"a": 4, "b": 0, "c": 6}, "rejected": ["t2", "t4", "t5", "t6"]}),
                *[(call("apply_transfers", {"a": 5, "b": 0}, [{"id": "t", "from": "a", "to": "b", "amount": bad}]),
                   {"balances": {"a": 5, "b": 0}, "rejected": ["t"]}) for bad in [-3, 2.5, "5", True]],
                ("(lambda b: (apply_transfers(b, [{'id': 't', 'from': 'a', 'to': 'b', 'amount': 1}]), b)[1])"
                 "({'a': 1, 'b': 0})", {"a": 1, "b": 0})],
        reference="""def apply_transfers(balances, transfers):
    bal = dict(balances)
    rejected = []
    for t in transfers:
        src, dst, amount = t["from"], t["to"], t["amount"]
        if (src in bal and dst in bal and src != dst
                and type(amount) is int and amount > 0 and bal[src] >= amount):
            bal[src] -= amount
            bal[dst] += amount
        else:
            rejected.append(t["id"])
    return {"balances": bal, "rejected": rejected}
""",
    ),
    Task(
        "query_string",
        "Write `parse_query(qs: str) -> dict[str, list[str]]` that parses a URL query string without the leading "
        "'?'. Pairs are separated by '&' and empty pairs are skipped. Split each pair at the first '='; a pair "
        "without '=' has the value ''. In keys and values, decode '+' as a space and %XX escapes as UTF-8. "
        "Repeated keys collect their values in order.",
        public=[(call("parse_query", "a=1&b=2"), {"a": ["1"], "b": ["2"]}),
                (call("parse_query", "x=1&x=2"), {"x": ["1", "2"]}),
                (call("parse_query", ""), {})],
        hidden=[(call("parse_query", "flag"), {"flag": [""]}),
                (call("parse_query", "a=&b"), {"a": [""], "b": [""]}),
                (call("parse_query", "q=hello+world"), {"q": ["hello world"]}),
                (call("parse_query", "q=a%2Bb"), {"q": ["a+b"]}),
                (call("parse_query", "k=v=w"), {"k": ["v=w"]}),
                (call("parse_query", "a=1&&b=2&"), {"a": ["1"], "b": ["2"]}),
                (call("parse_query", "caf%C3%A9=%E2%9C%93"), {"café": ["✓"]}),
                (call("parse_query", "a+b=c"), {"a b": ["c"]}),
                (call("parse_query", "=v"), {"": ["v"]})],
        reference="""from urllib.parse import unquote_plus

def parse_query(qs):
    out = {}
    for pair in qs.split("&"):
        if pair:
            k, _, v = pair.partition("=")
            out.setdefault(unquote_plus(k), []).append(unquote_plus(v))
    return out
""",
    ),
    Task(
        "number_words",
        "Write `number_to_words(n: int) -> str` for 0 <= n <= 999999 in lowercase American English: hyphenate "
        "twenty-one to ninety-nine, never use 'and', and use 'thousand' for the upper three digits.",
        public=[(call("number_to_words", 0), "zero"),
                (call("number_to_words", 42), "forty-two"),
                (call("number_to_words", 105020), "one hundred five thousand twenty")],
        hidden=[(call("number_to_words", 7), "seven"),
                (call("number_to_words", 13), "thirteen"),
                (call("number_to_words", 20), "twenty"),
                (call("number_to_words", 90), "ninety"),
                (call("number_to_words", 100), "one hundred"),
                (call("number_to_words", 101), "one hundred one"),
                (call("number_to_words", 1000), "one thousand"),
                (call("number_to_words", 1010), "one thousand ten"),
                (call("number_to_words", 12345), "twelve thousand three hundred forty-five"),
                (call("number_to_words", 110000), "one hundred ten thousand"),
                (call("number_to_words", 999999), "nine hundred ninety-nine thousand nine hundred ninety-nine")],
        reference="""ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

def number_to_words(n):
    def below_thousand(n):
        words = []
        if n >= 100:
            words.append(ONES[n // 100] + " hundred")
            n %= 100
        if n >= 20:
            words.append(TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else ""))
        elif n:
            words.append(ONES[n])
        return " ".join(words)

    if n == 0:
        return "zero"
    high, low = divmod(n, 1000)
    words = [below_thousand(high) + " thousand"] if high else []
    if low:
        words.append(below_thousand(low))
    return " ".join(words)
""",
    ),
]


if __name__ == "__main__":
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    broken = []
    for t in TASKS:
        assert t.prompt.startswith(t.body), t.id  # prompts render, so a broken prompt fails here, not mid-run
        fails = run_checks(t.reference, t.public + t.hidden, t.forbid)
        if fails or not run_checks("", t.public + t.hidden, t.forbid):
            broken.append(t.id)
            print(t.id, fails[:3])
    print(f"{len(TASKS) - len(broken)}/{len(TASKS)} tasks ok, "
          f"{sum(len(t.public) + len(t.hidden) for t in TASKS)} checks")
    sys.exit(1 if broken else 0)
