"""Generate the manuscript's result tables (manuscript/tables/*.tex) from
runs_v5/results — the single source of truth. Also computes two derived columns
from the per-episode records: mean time-to-goal over successes, and the
deadline-met rate (fraction of EPISODES finishing within the task deadline).

    .venv\\Scripts\\python.exe scripts\\make_paper_tables.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import campaign_v5_stable as C  # noqa: E402

OUT = ROOT / "manuscript" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
RES = C.RUNS / "results"
DEADLINE = dict(STAT_medicine=45, nurse_call=60, specimen=90, restock=180, idle_return=300)
AG6 = ["cwn", "fixed_equal", "nav_heavy", "attention", "vanilla", "he2025"]
TEXNAME = {"cwn": "CWN", "fixed_equal": "Fixed-equal", "nav_heavy": "Nav-heavy", "attention": "Attention",
           "vanilla": "Vanilla", "he2025": "Risk-field", "residual_sf": "Residual (ours)",
           "safe_heavy": "Safety-heavy", "td3_equal": "TD3", "ppo_equal": "PPO"}


def table(group):
    return json.load(open(RES / group / "table.json"))


def pooled_records(group, agent, tag):
    """Pool per-episode records over all seeds of (group, agent) for eval tag."""
    spec = C.GROUPS[group]
    files = {"final": lambda rd: rd.parent / "eval_final.json", "shield": lambda rd: rd.parent / "eval_shield.json",
             "shield2": lambda rd: rd.parent / "eval_shield2.json", "mid": lambda rd: rd / "eval_mid.json"}
    recs = []
    for s in spec["seeds"]:
        f = files[tag](C.run_dir((group, agent, s)))
        if f.exists():
            recs += json.load(open(f))["records"]
    return recs


def classical_records(name):
    f = RES / "tier2" / "classical" / f"{name}.json"
    return json.load(open(f))["records"] if f.exists() else []


def tstats(recs):
    """(time-to-goal over successes, deadline-met rate over all episodes)."""
    if not recs:
        return float("nan"), float("nan")
    ct = [r["steps"] * 0.1 for r in recs if r["success"]]
    met = [r["success"] and r["steps"] * 0.1 <= DEADLINE[r["task_id"]] for r in recs]
    return (float(np.mean(ct)) if ct else float("nan")), float(np.mean(met))


def f1(x, mul=100):
    return "--" if x != x else f"{mul * x:.1f}"


# ---------------------------------------------------------------- tab:main (two tiers)
rows = []
for agent in AG6:
    cells = [TEXNAME[agent]]
    for tier in ("tier1", "tier2"):
        t = table(tier)[agent]
        cells += [f"${f1(t['final']['sr'])}\\pm{f1(t['final']['sr_std'])}$", f1(t['final']['collision']),
                  f"{t['final']['cvr']:.3f}", f"${f1(t['shield2']['sr'])}\\pm{f1(t['shield2']['sr_std'])}$",
                  f1(t['shield2']['timeout'])]
    rows.append(" & ".join(cells) + r" \\")
(OUT / "tab_main.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

# ------------------------------------------------- tab:drivers (Tier-2 driver study)
drv = []


def add(name, kind, plain_summary, plain_recs, v2_summary, v2_recs, note=""):
    tg, dl = tstats(plain_recs)
    drv.append(" & ".join([
        name, kind,
        f1(plain_summary.get("sr", float("nan"))), f1(plain_summary.get("collision", float("nan"))),
        f1(v2_summary.get("sr", float("nan"))), f1(v2_summary.get("timeout", float("nan"))),
        "--" if tg != tg else f"{tg:.0f}", f1(dl)]) + r" \\" + (f" % {note}" if note else ""))


def sm(recs):
    if not recs:
        return {}
    return dict(sr=float(np.mean([r["success"] for r in recs])),
                collision=float(np.mean([r["end"] == "collision" for r in recs])),
                timeout=float(np.mean([r["end"] == "timeout" for r in recs])))


# mapless anchors
mp = table("tier2_mapless")
mp_recs = pooled_records("tier2_mapless", "fixed_equal", "final")
add("SAC (mapless)", "learned", mp["fixed_equal"]["final"], mp_recs, mp["fixed_equal"]["shield2"],
    pooled_records("tier2_mapless", "fixed_equal", "shield2"))
# classical family
for name, label in (("sf", "Social force (stock)"), ("astar_pp", "A$^*$ + pure pursuit"),
                    ("planner_pp", "Planner + pure pursuit"), ("planner_orca", "Planner + ORCA"),
                    ("planner_sf2_tuned", "Planner + tuned SF")):
    plain = classical_records(name)
    v2 = classical_records(("planner_sf2_tunedb_shield2" if name == "planner_sf2_tuned" else name + "_shield2"))
    add(label, "classical", sm(plain), plain, sm(v2), v2)
# learned hierarchical (best fixed-weight representative + algorithms)
t2 = table("tier2")
add("Planner + SAC (CARS)", "learned", t2["fixed_equal"]["final"], pooled_records("tier2", "fixed_equal", "final"),
    t2["fixed_equal"]["shield2"], pooled_records("tier2", "fixed_equal", "shield2"))
td3 = table("tier2_td3")["td3_equal"]
add("Planner + TD3", "learned", td3["final"], pooled_records("tier2_td3", "td3_equal", "final"),
    td3["shield2"], pooled_records("tier2_td3", "td3_equal", "shield2"))
ppo = table("tier2_ppo")["ppo_equal"]
add("Planner + PPO (defaults)", "learned", ppo["final"], pooled_records("tier2_ppo", "ppo_equal", "final"),
    ppo["shield2"], pooled_records("tier2_ppo", "ppo_equal", "shield2"))
sh = table("tier2_safe")["safe_heavy"]
add("Planner + SAC (safety-heavy)", "learned", sh["final"], pooled_records("tier2_safe", "safe_heavy", "final"),
    sh["shield2"], pooled_records("tier2_safe", "safe_heavy", "shield2"))
rs = table("tier2_residual")["residual_sf"]
add(r"\textbf{Planner + residual SAC}", "hybrid", rs["final"], pooled_records("tier2_residual", "residual_sf", "final"),
    rs["shield2"], pooled_records("tier2_residual", "residual_sf", "shield2"))
(OUT / "tab_drivers.tex").write_text("\n".join(drv) + "\n", encoding="utf-8")

# ------------------------------------------------------------- tab:ablation (2x2)
pt_main = json.load(open(RES / "tier2" / "per_task.json"))
pt_abl = json.load(open(RES / "tier2_planner_ablation" / "per_task.json"))
t_abl = table("tier2_planner_ablation")
cells = {
    ("CARS", "cult"): (pt_main["fixed_equal"]["final"]["STAT_medicine"]["cvr"], t2["fixed_equal"]["final"]["cvr"],
                       t2["fixed_equal"]["final"]["sr"]),
    ("CARS", "geo"): (pt_abl["fixed_equal"]["final"]["STAT_medicine"]["cvr"], t_abl["fixed_equal"]["final"]["cvr"],
                      t_abl["fixed_equal"]["final"]["sr"]),
    ("blind", "cult"): (pt_abl["vanilla"]["final"]["STAT_medicine"]["cvr"], t_abl["vanilla"]["final"]["cvr"],
                        t_abl["vanilla"]["final"]["sr"]),
    ("blind", "geo"): (pt_main["vanilla"]["final"]["STAT_medicine"]["cvr"], t2["vanilla"]["final"]["cvr"],
                       t2["vanilla"]["final"]["sr"]),
}
lines = []
for rw, label in (("CARS", "CARS reward (fixed-equal)"), ("blind", "Culture-blind reward (vanilla)")):
    a = cells[(rw, "geo")]; b = cells[(rw, "cult")]
    lines.append(f"{label} & {a[0]:.3f} & {a[1]:.3f} & {f1(a[2])} & {b[0]:.3f} & {b[1]:.3f} & {f1(b[2])} \\\\")
(OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ------------------------------------------------------------- tab:robustness
RB = RES / "tier2" / "robustness"
CL = RES / "tier2" / "classical"


def rob(cond, drv_key, sh):
    if drv_key == "sf2":
        if cond == "unmapped":
            recs = []
            for i in range(10):
                f = CL / f"planner_sf2__unmapped_L{i}__sf2{'_shield2' if sh else ''}.json"
                if f.exists():
                    recs += json.load(open(f))["records"]
        else:
            f = CL / f"planner_sf2__{cond}__sf2{'_shield2' if sh else ''}.json"
            recs = json.load(open(f))["records"] if f.exists() else []
    else:
        if cond == "unmapped":
            recs = []
            for i in range(10):
                f = RB / f"unmapped_L{i}__{drv_key}{'_shield2' if sh else ''}.json"
                if f.exists():
                    recs += json.load(open(f))["records"]
        else:
            f = RB / f"{cond}__{drv_key}{'_shield2' if sh else ''}.json"
            recs = json.load(open(f))["records"] if f.exists() else []
    s = sm(recs)
    return f"{f1(s.get('sr', float('nan')))}/{f1(s.get('collision', float('nan')))}"


lines = []
for cond, label in (("unmapped", "Unmapped obstacles"), ("noyield", "Non-yielding crowd"),
                    ("lidar", r"LiDAR noise (5\,\% + 2\,\% drop)"), ("act", r"Actuation noise (10\,\%)")):
    row = [label]
    for drv_key in ("fe", "van", "sf2"):
        row += [rob(cond, drv_key, False), rob(cond, drv_key, True)]
    lines.append(" & ".join(row) + r" \\")
(OUT / "tab_robustness.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ------------------------------------------------------------- tab:shield (v0/v1/v2)
lines = []
for label, plain, v1, v2 in (
    ("Planner + SAC (fixed-equal)", t2["fixed_equal"]["final"], t2["fixed_equal"]["shield"], t2["fixed_equal"]["shield2"]),
    ("Planner + SAC (vanilla)", t2["vanilla"]["final"], t2["vanilla"]["shield"], t2["vanilla"]["shield2"]),
    ("Planner + residual SAC", rs["final"], rs["shield"], rs["shield2"]),
):
    cells = [label]
    for r in (plain, v1, v2):
        cells += [f1(r["sr"]), f1(r["collision"]), f1(r["timeout"])]
    lines.append(" & ".join(cells) + r" \\")
# classical row (unbiased tuned)
pl = sm(classical_records("planner_sf2_tuned")); v1c = sm(classical_records("planner_sf2_tuned_shield"))
v2c = sm(classical_records("planner_sf2_tunedb_shield2"))
cells = ["Planner + tuned SF"]
for r in (pl, v1c, v2c):
    cells += [f1(r.get("sr")), f1(r.get("collision")), f1(r.get("timeout"))]
lines.append(" & ".join(cells) + r" \\")
(OUT / "tab_shield.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")
for f in sorted(OUT.glob("*.tex")):
    print(f"--- {f.name}:")
    print(f.read_text(encoding="utf-8"))
