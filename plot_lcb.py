"""Experiment 2 chart: pass rate vs cost per attempt, one panel per model, one point per (strategy, effort).

Run:  uv run plot_lcb.py runs/lcb_A.jsonl runs/lcb_B.jsonl ... [--single-shot-redo runs/lcb_C.jsonl] [-o think_vs_verify.png]
--single-shot-redo: a one-turn rerun (e.g. with a larger output cap) whose attempts replace the matching
single-shot points; the feedback strategies are left as they were.
Single shot is turn 1 of the examples harness (its cost is turn 1 only), so it shares attempts with
"example feedback". Errored attempts are left out.
"""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # first three categorical slots, validated all-pairs for scatter
    "single shot": "#2a78d6",
    "example feedback": "#eb6834",
    "self-tests": "#1baf7a",
}
MARKERS = {"single shot": "o", "example feedback": "s", "self-tests": "^"}
# each series labels its points on a different side so nearby points do not collide
LABEL_AT = {"single shot": ((0, -14), "center"), "example feedback": ((-8, 7), "right"), "self-tests": ((8, 7), "left")}

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("-o", "--out", default="think_vs_verify.png")
parser.add_argument("--single-shot-redo")
parser.add_argument("--note", default="", help="extra footnote text")
args = parser.parse_args()

rows = [json.loads(line) for p in args.runs for line in open(p)]
redo = {}
if args.single_shot_redo:
    redo = {(r["model"], r["effort"], r["problem"], r["rep"]): r for r in map(json.loads, open(args.single_shot_redo))}
points = defaultdict(list)  # (model, strategy, effort) -> [(passed, cost)]
for r in rows:
    if r["harness"] == "examples" and r["turns"]:  # single shot needs only a finished turn 1
        s = redo.get((r["model"], r["effort"], r["problem"], r["rep"]), r)
        points[(r["model"], "single shot", r["effort"])].append((s["pass_first"], s["first_cost_usd"]))
    if r["error"]:  # a later turn failed on the API side, so the multi-turn result is unknown
        continue
    if r["harness"] == "examples":
        points[(r["model"], "example feedback", r["effort"])].append((r["pass_final"], r["cost_usd"]))
    else:
        points[(r["model"], "self-tests", r["effort"])].append((r["pass_final"], r["cost_usd"]))

models = sorted({k[0] for k in points})
print("| model | strategy | effort | n | pass | $ / attempt | $ / solve |")
print("|---|---|---|---|---|---|---|")
for key in sorted(points):
    v = points[key]
    passed, cost = sum(p for p, _ in v), sum(c for _, c in v)
    print(f"| {key[0].split('/')[-1]} | {key[1]} | {key[2]} | {len(v)} | {passed}/{len(v)} ({100 * passed / len(v):.0f}%) | "
          f"${cost / len(v):.5f} | {f'${cost / passed:.5f}' if passed else '-'} |")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), dpi=200, squeeze=False)
fig.patch.set_facecolor(SURFACE)
for ax, model in zip(axes[0], models):
    ax.set_facecolor(SURFACE)
    for strategy, color in SERIES.items():
        pts = sorted(((sum(c for _, c in v) / len(v), 100 * sum(p for p, _ in v) / len(v), e)
                      for (m, s, e), v in points.items() if m == model and s == strategy))
        if not pts:
            continue
        xs, ys, efforts = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.5, zorder=1)
        ax.scatter(xs, ys, s=70, color=color, marker=MARKERS[strategy], edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=strategy)
        offset, align = LABEL_AT[strategy]
        for x, y, e in pts:
            ax.annotate(e, (x, y), textcoords="offset points", xytext=offset, ha=align, fontsize=8.5, color=TEXT_2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x * 1000:g}"))
    ax.set_title(model.split("/")[-1], loc="left", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xlabel("$ per 1,000 attempts (log scale)", color=TEXT_2)
    ax.set_ylabel("solved (%)", color=TEXT_2)
    ax.set_ylim(30, 95)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.91), ncol=3, frameon=False, fontsize=9.5,
           labelcolor=TEXT_2)
fig.suptitle("Think longer or verify? LiveCodeBench medium/hard, Sail flex", x=0.01, ha="left", fontsize=13,
             fontweight="bold", color=TEXT, y=0.98)
fig.text(0.01, 0.015, "Labels are reasoning effort. Up-and-left is better: more solved for less money. " + args.note,
         fontsize=8.5, color=TEXT_2)
fig.tight_layout(rect=(0, 0.04, 1, 0.84))
fig.savefig(args.out, facecolor=SURFACE)
print("wrote", args.out)
