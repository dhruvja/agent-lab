"""Dumbbell chart of turn-1 vs final pass rate per harness, built from run files.

Run:  uv run plot.py runs/A.jsonl runs/B.jsonl ... [-o harnesses.png]
Only models present in every file are counted, so each harness is compared on the same models.
"""
import argparse
import json

import matplotlib.pyplot as plt

LABELS = {
    "examples": "Example failures fed back",
    "selftest": "Own tests + judge each failure",
    "resample": "Own tests + blind retry",
}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TURN1, FINAL = "#86b6ef", "#1c5cab"  # one hue, two shades (validated ordinal pair)

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("-o", "--out", default="harnesses.png")
args = parser.parse_args()

runs = [[json.loads(line) for line in open(p)] for p in args.runs]
models = set.intersection(*({r["model"] for r in rows} for rows in runs))

stats = []
for rows in runs:
    rows = [r for r in rows if r["model"] in models]
    stats.append({
        "harness": rows[0].get("harness", "examples"),
        "n": len(rows),
        "first": sum(r["pass_first"] for r in rows),
        "final": sum(r["pass_final"] for r in rows),
        "breaks": sum(r["pass_first"] and not r["pass_final"] for r in rows),
    })
stats.sort(key=lambda s: s["final"])
for s in stats:
    print(s)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for y, s in enumerate(stats):
    first, final = 100 * s["first"] / s["n"], 100 * s["final"] / s["n"]
    ax.plot([first, final], [y, y], color=FINAL, linewidth=2, zorder=1, solid_capstyle="round")
    ax.scatter([first], [y], s=90, color=TURN1, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.scatter([final], [y], s=90, color=FINAL, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(final + 0.7, y + 0.03, f"{s['final']}/{s['n']}", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
    ax.text(final + 0.7, y - 0.03, f"{s['breaks']} correct broken", va="top", color=TEXT_2, fontsize=9)

ax.set_yticks(range(len(stats)), [LABELS.get(s["harness"], s["harness"]) for s in stats], color=TEXT)
ax.set_xlim(80, 106)
ax.set_xticks(range(80, 101, 5), [f"{x}%" for x in range(80, 101, 5)], color=TEXT_2)
ax.set_ylim(-0.6, len(stats) - 0.4)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)

ax.scatter([], [], s=90, color=TURN1, label="turn 1 (single shot)")
ax.scatter([], [], s=90, color=FINAL, label="after up to 3 turns")
fig.legend(loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=2, frameon=False, fontsize=10, labelcolor=TEXT_2,
           handletextpad=0.3, columnspacing=1.2)
fig.suptitle("Pass rate on hidden tests", x=0.02, ha="left", fontsize=14, fontweight="bold", color=TEXT, y=0.97)
short = ", ".join(sorted(m.split("/")[-1] for m in models))
fig.text(0.02, 0.025, f"{short} · 12 coding tasks × 3 repeats · same-prompt turn-1 noise ≈ ±4 attempts",
         fontsize=8.5, color=TEXT_2)
fig.tight_layout(rect=(0, 0.05, 1, 0.84))
fig.savefig(args.out, facecolor=SURFACE)
print("wrote", args.out)
