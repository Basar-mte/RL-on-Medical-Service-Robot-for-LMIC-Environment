"""Statistics of the repeated Gazebo campaign (results_runs_repeats/, sim/run_campaign.py
--seeds 1-10): per cell (tier x pedestrian level x policy x escape) the success rate with a
Wilson 95 % interval over its seeds, the distributions of time to goal, path, proxemics and
prayer-zone depth, Fisher exact tests of escape on vs off, and data-quality checks.
Writes results_summary/gazebo_repeats.md and paper_figures/fig12_success_rates.{pdf,png},
fig13_proxemics.{pdf,png}.

    python3 analyze_repeats.py [--runs ../results_runs_repeats]

Each seed is an independent run: every walker starts at a different point of its loop
(ped_routes.phases), and the simulator's own timing adds the rest. A success rate here is
a measured frequency over those runs - with ten runs per cell its interval is wide, and the
intervals, not the point estimates, are what a comparison may rest on.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from zone_metrics import zone_metrics   # noqa: E402

POLICIES = ["residual_sf", "classical_sf2", "fixed_equal", "vanilla", "risk_field", "mapless_fixed_equal"]
LABEL = {"residual_sf": "residual_sf", "classical_sf2": "classical SF", "fixed_equal": "fixed_equal",
         "vanilla": "vanilla", "risk_field": "risk_field", "mapless_fixed_equal": "mapless"}
LEVEL = {"busy": "8 walkers", "crowd45": "45 walkers"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, b, c, d):
    """Fisher exact test of the 2x2 table [[a, b], [c, d]] (two-sided, sum of
    the probabilities no larger than the observed one)."""
    n1, n2, k = a + b, c + d, a + c
    N = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p0 = p(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def q(xs, f):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * f
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def med_iqr(xs, nd=1):
    if not xs:
        return "-"
    return f"{q(xs, 0.5):.{nd}f} [{q(xs, 0.25):.{nd}f}, {q(xs, 0.75):.{nd}f}]"


def quality(run_dir):
    """(largest logging gap s, max tilt deg) of a run, from its run.csv."""
    try:
        rows = list(csv.DictReader(open(os.path.join(run_dir, "run.csv"))))
        ts = [float(r["sim_abs"]) for r in rows]
        gap = max((b - a for a, b in zip(ts, ts[1:])), default=0.0)
        tilt = max(float(r["tilt_deg"]) for r in rows) if rows else float("nan")
        return gap, tilt
    except Exception:
        return float("nan"), float("nan")


def load(run_root):
    recs = json.load(open(os.path.join(run_root, "campaign.json")))
    cells = defaultdict(list)
    for r in recs:
        d = os.path.join(run_root, r["tag"])
        g = (r.get("goals") or [{}])[-1]
        ped = r.get("pedestrians") or {}
        zm = zone_metrics(os.path.join(d, "run.csv")) if os.path.exists(os.path.join(d, "run.csv")) else None
        gap, tilt = quality(d)
        cells[(r["tier"], r["peds"], r["preset"], bool(r["escape_enabled"]))].append({
            "seed": r.get("ped_seed"), "outcome": r.get("outcome"),
            "reached": r.get("outcome") == "reached",
            "t": g.get("seconds") if r.get("outcome") == "reached" else None,
            "path": r.get("path_length_m"),
            "contacts": ped.get("contacts"), "touches": ped.get("touches"),
            "pmin": ped.get("min_distance_m"), "pz": ped.get("personal_zone_s"),
            "prayer": zm["prayer"]["max_depth_m"] if zm else None,
            "stall": (r.get("escapes") or {}).get("stall"), "wedge": (r.get("escapes") or {}).get("wedge"),
            "gap": gap, "tilt": tilt, "budget": g.get("seconds"),
        })
    return cells


def table(cells, tier, peds):
    lines = [f"### {tier} ward, {LEVEL.get(peds, peds)}", "",
             "| policy | escape | reached / runs | success (Wilson 95 %) | time to goal s, median [IQR] "
             "| path m | contacts / run | touches / run | personal zone s | prayer depth m |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        for e in (True, False):
            runs = cells.get((tier, peds, p, e), [])
            if not runs:
                continue
            k, n = sum(r["reached"] for r in runs), len(runs)
            lo, hi = wilson(k, n)
            v = lambda key: [r[key] for r in runs if r[key] is not None]
            lines.append(
                f"| {p} | {'on' if e else 'off'} | {k} / {n} | {100 * k / n:.0f} % [{100 * lo:.0f}, {100 * hi:.0f}] "
                f"| {med_iqr(v('t'))} | {med_iqr(v('path'))} | {med_iqr(v('contacts'), 0)} | {med_iqr(v('touches'), 0)} "
                f"| {med_iqr(v('pz'))} | {med_iqr(v('prayer'), 2)} |")
    return "\n".join(lines)


def escape_tests(cells, tier, peds):
    lines = ["| policy | escape on | escape off | Fisher exact p (two-sided) |", "|---|---|---|---|"]
    for p in POLICIES:
        on, off = cells.get((tier, peds, p, True), []), cells.get((tier, peds, p, False), [])
        if not on or not off:
            continue
        a, b = sum(r["reached"] for r in on), sum(not r["reached"] for r in on)
        c, d = sum(r["reached"] for r in off), sum(not r["reached"] for r in off)
        lines.append(f"| {p} | {a} / {a + b} | {c} / {c + d} | {fisher_two_sided(a, b, c, d):.3f} |")
    return "\n".join(lines)


def figures(cells, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    combos = [(t, pl) for pl in ("busy", "crowd45") for t in ("walled", "open")]
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.4), sharey=True)
    for ax, (tier, peds) in zip(axs.flat, combos):
        for j, (e, col) in enumerate(((True, "#2c7bb6"), (False, "#d7191c"))):
            xs, ys, err = [], [], [[], []]
            for i, p in enumerate(POLICIES):
                runs = cells.get((tier, peds, p, e), [])
                if not runs:
                    continue
                k, n = sum(r["reached"] for r in runs), len(runs)
                lo, hi = wilson(k, n)
                xs.append(i + (j - 0.5) * 0.36); ys.append(100 * k / n)
                # clamp: at 0/n or n/n the bound equals the estimate up to rounding
                err[0].append(max(0.0, 100 * (k / n - lo))); err[1].append(max(0.0, 100 * (hi - k / n)))
            ax.bar(xs, ys, width=0.34, color=col, alpha=0.85, label=f"escape {'on' if e else 'off'}")
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
        n_runs = max((len(cells.get((tier, peds, p, True), [])) for p in POLICIES), default=0)
        ax.set_title(f"{tier} ward, {LEVEL[peds]} (n = {n_runs} per bar)", fontsize=7)
        ax.set_xticks(range(len(POLICIES)))
        ax.set_xticklabels([LABEL[p] for p in POLICIES], rotation=25, ha="right", fontsize=6.3)
        ax.set_ylim(0, 105); ax.grid(axis="y", lw=0.3, alpha=0.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("goal reached (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig12_success_rates.{ext}"), dpi=600)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5))
    for ax, key, lab, ttl in ((axs[0], "contacts", "contacts per run", "(a) pedestrian contacts (< 0.5 m)"),
                              (axs[1], "pz", "seconds", "(b) time with a pedestrian within 1.2 m")):
        data, pos, cols, ticks = [], [], [], []
        for i, p in enumerate(POLICIES):
            for j, (pl, col) in enumerate((("busy", "#fdae61"), ("crowd45", "#d7191c"))):
                vals = [r[key] for t in ("walled", "open") for r in cells.get((t, pl, p, True), [])
                        if r[key] is not None]
                if vals:
                    data.append(vals); pos.append(i + (j - 0.5) * 0.36); cols.append(col)
            ticks.append(i)
        if data:
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6),
                            capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
            for b, c in zip(bp["boxes"], cols):
                b.set_facecolor(c); b.set_alpha(0.85)
        ax.set_xticks(ticks); ax.set_xticklabels([LABEL[p] for p in POLICIES], rotation=25, ha="right", fontsize=6.3)
        ax.set_ylabel(lab, fontsize=6.5); ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.set_title(ttl, fontsize=7)
    from matplotlib.patches import Patch
    # escape on, both wards pooled (stated in the caption, paper_figures/README.md)
    fig.legend(handles=[Patch(fc="#fdae61", label="8 walkers"), Patch(fc="#d7191c", label="45 walkers")],
               fontsize=6.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig13_proxemics.{ext}"), dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "results_runs_repeats"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results_summary", "gazebo_repeats.md"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "paper_figures"))
    a = ap.parse_args()
    cells = load(a.runs)
    allruns = [r for v in cells.values() for r in v]
    seeds = sorted({r["seed"] for r in allruns if r["seed"] is not None})
    noresult = sum(r["outcome"] not in ("reached", "timeout") for r in allruns)
    gaps = [r for r in allruns if r["gap"] == r["gap"] and r["gap"] > 1.0]
    over = [r for r in allruns if r["outcome"] == "timeout" and (r["budget"] or 0) > 301.0]
    tilt = max((r["tilt"] for r in allruns if r["tilt"] == r["tilt"]), default=float("nan"))
    out = ["# Gazebo repeated campaign - success rates over pedestrian seeds", "",
           f"{len(allruns)} runs in {len(cells)} cells (tier x pedestrian level x policy x escape), seeds "
           f"{seeds[0] if seeds else '-'}-{seeds[-1] if seeds else '-'}. Task: nurse desk (37, 15) -> emergency room "
           "(5, 7.5), `reached` = within 0.4 m inside 300 s simulation time. Each seed starts every walker at a "
           "different point of its loop (ped_routes.phases), so repeated runs meet different crowds. Success rates "
           "are frequencies over those runs with Wilson 95 % intervals; with ten runs per cell the intervals are "
           "wide (10 / 10 is [72, 100] %), and only non-overlapping intervals - or the Fisher tests below - support a "
           "comparison. Distributions are median [interquartile range].", "",
           "## Data quality", "",
           f"- runs without a result: {noresult}",
           f"- runs with a pose/logging gap > 1 s: {len(gaps)}"
           + (" (" + ", ".join(sorted({str(r['seed']) for r in gaps})) + ")" if gaps else ""),
           f"- timeouts recorded after more than 301 s: {len(over)}",
           f"- largest chassis tilt in any run: {tilt:.1f} deg (> 45 would be a knock-down)", ""]
    for peds in ("busy", "crowd45"):
        for tier in ("walled", "open"):
            if any(k[0] == tier and k[1] == peds for k in cells):
                out += [table(cells, tier, peds), "", "Escape on vs off:", "", escape_tests(cells, tier, peds), ""]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    os.makedirs(a.fig_dir, exist_ok=True)
    figures(cells, a.fig_dir)
    print(f"wrote {os.path.relpath(a.out, ROOT)} and fig12/fig13 ({len(allruns)} runs, {len(cells)} cells)")


if __name__ == "__main__":
    main()
