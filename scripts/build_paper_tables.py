"""Build the five results tables (Tables IV to VIII) of the journal article.

Reads repository result files only (runs_v5/, ros_sim/paper_data/) and writes only into the
folder that holds this script:
    tab_main.tex        Table IV   (tab:main)        six SAC reward designs x two tiers
    tab_drivers.tex     Table V    (tab:drivers)     Tier-2 local drivers, no / cone / swept filter
    tab_compliance.tex  Table VI   (tab:compliance)  2 x 2 reward x planner attribution
    tab_robust.tex      Table VII  (tab:robust)      assumption stress tests
    tab_gazebo.tex      Table VIII (tab:gazebo)      480-run Gazebo study

Conventions (design.md parts e, f, f.1):
  * "+-" is the SAMPLE standard deviation over training seeds (ddof = 1), from per_seed.json.
  * Time to goal of learned drivers is pooled over seeds as in scripts/make_paper_tables.py
    (mean of steps x 0.1 s over all successful episodes of all seeds).
  * Tuned SF with the swept filter uses planner_sf2_tunedb_shield2.json (97.8 %).
  * One decimal for percentages, three for CVR and SVR.
It also prints a check of every number against the curated fact sheets.

usage:  python -B build_tables.py
Standard library only.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_v5"
RES = RUNS / "results"
CL2 = RES / "tier2" / "classical"
ROB = RES / "tier2" / "robustness"
GZ = ROOT / "ros_sim" / "paper_data" / "runs.csv"
OUT = ROOT / "paper_tables"
OUT.mkdir(exist_ok=True)

CHECKS: list[tuple[str, str, str]] = []   # (what, computed, expected)


# ----------------------------------------------------------------------------- helpers
def load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def table(group):
    return load(RES / group / "table.json")


def per_seed(group):
    return load(RES / group / "per_seed.json")


def seeds_of(group, agent):
    return sorted(int(p.name.rsplit("_s", 1)[1]) for p in (RUNS / group).glob(f"{agent}_s[0-9]*") if p.is_dir())


def pooled(group, agent, tag):
    """Per-episode records of all seeds of (group, agent); tag in final, shield, shield2."""
    recs = []
    for s in seeds_of(group, agent):
        f = RUNS / group / f"{agent}_s{s}" / f"eval_{tag}.json"
        if f.exists():
            recs += load(f)["records"]
    return recs


def summ(recs):
    n = len(recs)
    return dict(n=n,
                sr=sum(r["success"] for r in recs) / n,
                coll=sum(r["end"] == "collision" for r in recs) / n,
                to=sum(r["end"] == "timeout" for r in recs) / n,
                svr=sum(r["svr"] for r in recs) / n,
                cvr=sum(r["cvr"] for r in recs) / n)


def tgoal(recs):
    """make_paper_tables.py rule: mean of steps*0.1 over successful episodes (pooled)."""
    ct = [r["steps"] * 0.1 for r in recs if r["success"]]
    return sum(ct) / len(ct) if ct else float("nan")


def sd1(values):
    return st.stdev(values) if len(values) > 1 else float("nan")


def p1(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f}"


def p3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def pm(mean, sd):
    return f"${100 * mean:.1f}\\pm{100 * sd:.1f}$"


def pm_small(mean, sd):
    """Mean with a smaller '+- sd' (Table V, where width is tight)."""
    return f"{100 * mean:.1f}" + r"{\tiny$\,\pm$" + f"{100 * sd:.1f}" + "}"


def chk(what, computed: str, expected: str):
    CHECKS.append((what, computed, expected))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fisher2(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables with p <= p_obs)."""
    r1, c1, n = a + b, a + c, a + b + c + d
    def hp(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = hp(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return min(1.0, sum(hp(x) for x in range(lo, hi + 1) if hp(x) <= p0 * (1 + 1e-7)))


def note(width, text):
    return ("\n\\vspace{2pt}\n"
            f"\\begin{{minipage}}{{{width}}}\n\\footnotesize\n{text.strip()}\n\\end{{minipage}}\n")


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"wrote {OUT / name}")


# ============================================================================ Table IV
AG6 = [("fixed_equal", "SAC-CARS"), ("nav_heavy", "SAC-CARS-nav"), ("cwn", "SAC-CWN"),
       ("attention", "SAC-Attn"), ("vanilla", "SAC-blind"), ("he2025", "SAC-risk")]
FACT_MAIN = {  # facts_sim_curated.md section 2: SR, coll, CVR, SR v2, T/O v2, popSD SR, popSD SR v2
    "tier2": {"cwn": (80.4, 19.6, 0.109, 94.8, 5.2, 1.3, 1.4), "fixed_equal": (80.4, 19.6, 0.110, 95.4, 4.6, 1.5, 1.7),
              "nav_heavy": (80.9, 19.1, 0.112, 95.2, 4.8, 3.2, 0.9), "attention": (79.9, 20.1, 0.107, 94.6, 5.4, 1.4, 1.2),
              "vanilla": (84.2, 15.8, 0.147, 96.1, 3.9, 2.4, 1.3), "he2025": (84.5, 15.5, 0.143, 95.7, 4.3, 1.3, 0.4)},
    "tier1": {"cwn": (77.0, 23.0, 0.097, 93.3, 6.7, 3.9, 2.3), "fixed_equal": (80.3, 19.7, 0.099, 95.4, 4.6, 1.8, 0.7),
              "nav_heavy": (81.9, 18.1, 0.108, 95.1, 4.9, 2.4, 1.5), "attention": (81.0, 19.0, 0.101, 94.7, 5.3, 2.3, 1.1),
              "vanilla": (83.2, 16.8, 0.123, 96.1, 3.9, 0.5, 0.6), "he2025": (84.5, 15.5, 0.123, 95.6, 4.4, 1.4, 0.9)},
}


def build_main():
    rows = []
    v2_coll_total, v2_eps = 0, 0
    for i, (ag, name) in enumerate(AG6):
        if i == 4:
            rows.append("\\midrule")
        cells = [name]
        for tier in ("tier1", "tier2"):
            T, P = table(tier)[ag], per_seed(tier)[ag]
            fin, sh2 = T["final"], T["shield2"]
            sr_seeds = [r["sr"] for r in P["final"]]
            v2_seeds = [r["sr"] for r in P["shield2"]]
            assert len(sr_seeds) == 5 and len(v2_seeds) == 5
            assert abs(st.mean(sr_seeds) - fin["sr"]) < 1e-12
            sd_f, sd_v = sd1(sr_seeds), sd1(v2_seeds)
            # population SD in table.json must equal sample SD * sqrt((n-1)/n)
            assert abs(sd_f * math.sqrt(4 / 5) - fin["sr_std"]) < 1e-9
            assert abs(sd_v * math.sqrt(4 / 5) - sh2["sr_std"]) < 1e-9
            v2_coll_total += round(sh2["collision"] * 2500)
            v2_eps += 2500
            cells += [pm(fin["sr"], sd_f), p1(fin["collision"]), p3(fin["cvr"]), pm(sh2["sr"], sd_v), p1(sh2["timeout"])]
            fx = FACT_MAIN[tier][ag]
            chk(f"IV {tier} {name} SR/coll/CVR/SRv2/TOv2",
                f"{p1(fin['sr'])}/{p1(fin['collision'])}/{p3(fin['cvr'])}/{p1(sh2['sr'])}/{p1(sh2['timeout'])}",
                f"{fx[0]:.1f}/{fx[1]:.1f}/{fx[2]:.3f}/{fx[3]:.1f}/{fx[4]:.1f}")
            chk(f"IV {tier} {name} popSD SR / SRv2 (facts) vs table.json",
                f"{100 * fin['sr_std']:.1f}/{100 * sh2['sr_std']:.1f}", f"{fx[5]:.1f}/{fx[6]:.1f}")
            assert fin["timeout"] == 0.0, "note says no-filter runs never time out"
        rows.append(" & ".join(cells) + " \\\\")
    chk("IV swept-filter collisions over 12 agent-tiers x 5 seeds", f"{v2_coll_total} in {v2_eps}", "0 in 30000")
    body = r"""%% Table IV (tab:main). Generated by build_tables.py. Do not edit by hand.
\begin{table*}[!t]
\centering
\caption{Six SAC Reward Designs on Both Tiers, Without a Filter and With the Swept Filter (5 Seeds $\times$ 500 Episodes per Row and Tier)}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{4.2pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{}l ccc cc ccc cc@{}}
\toprule
& \multicolumn{5}{c}{Tier 1, open ward floor} & \multicolumn{5}{c}{Tier 2, walled ward}\\
\cmidrule(lr){2-6}\cmidrule(l){7-11}
& \multicolumn{3}{c}{No filter} & \multicolumn{2}{c}{Swept filter} & \multicolumn{3}{c}{No filter} & \multicolumn{2}{c}{Swept filter}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}
Driver & SR (\%) & Coll (\%) & CVR & SR (\%) & T/O (\%) & SR (\%) & Coll (\%) & CVR & SR (\%) & T/O (\%)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""" + note(r"\textwidth", r"""
All drivers are SAC with the planner, at the final 2M-step checkpoint. The CARS rows use the cultural planner ($\kappa=3$), and SAC-blind and SAC-risk the geometric planner ($\kappa=0$). SR, Coll and T/O are the shares of episodes ending in success, contact or timeout, averaged over 5 seeds of 500 episodes. The value after $\pm$ is the sample standard deviation over seeds. CVR is the share of steps that break a cultural rule. It cannot fall much below 0.10 because two of the five tasks end inside a gendered ward. No-filter runs never time out, and the swept filter gave 0 contacts in all 30,000 of its episodes here.
""") + "\\end{table*}\n"
    write("tab_main.tex", body)


# ============================================================================ Table V
FACT_DRV = {  # facts_sim_curated.md section 5 and section 7 / tab_shield: SR, coll, cone SR, cone coll, v2 SR, v2 T/O, t_goal, SVR
    "stock_mapless": (4.6, 0.0, None, None, 4.4, 95.6, 47.6, 0.029),
    "tuned_mapless": (58.2, 5.8, None, None, None, None, 37.2, 0.011),
    "mapless_sac": (48.2, 51.8, None, None, 56.5, 43.5, 26.5, 0.037),
    "astar": (11.4, 84.4, None, None, 42.4, 57.6, 21.0, 0.039),
    "stock_planner": (6.0, 0.2, None, None, 6.4, 93.6, 53.0, 0.033),
    "pp": (37.0, 63.0, None, None, 79.6, 20.4, 23.6, 0.055),
    "orca": (66.8, 33.2, None, None, 74.2, 25.8, 43.2, 0.125),
    "tuned": (95.4, 4.6, 93.0, 0.0, 97.8, 2.2, 34.3, 0.014),
    "sac": (80.4, 19.6, 90.4, 6.0, 95.4, 4.6, 27.7, 0.042),
    "blind": (84.2, 15.8, 92.6, 4.4, 96.1, 3.9, 27.8, 0.041),
    "safe": (82.8, 17.2, None, None, 94.6, 5.4, 28.0, 0.037),
    "td3": (65.0, 32.0, None, None, 67.2, 32.8, 51.5, 0.041),
    "ppo": (0.0, 56.9, None, None, 0.0, 100.0, None, 0.058),
    "res": (94.0, 6.0, 95.7, 0.0, 99.4, 0.6, 30.0, 0.0145),
}


def classical_row(plain, cone, swept):
    def rec(name):
        return load(CL2 / f"{name}.json")["records"] if name and (CL2 / f"{name}.json").exists() else None
    pr, cr, sr = rec(plain), rec(cone), rec(swept)
    P = summ(pr)
    C = summ(cr) if cr else None
    S = summ(sr) if sr else None
    return dict(sr=P["sr"], coll=P["coll"], sd=None,
                csr=C["sr"] if C else None, ccoll=C["coll"] if C else None,
                ssr=S["sr"] if S else None, sto=S["to"] if S else None, ssd=None, scoll_n=round(S["coll"] * S["n"]) if S else None,
                tg=tgoal(pr), svr=P["svr"], n=P["n"])


def learned_row(group, agent):
    T, P = table(group)[agent], per_seed(group)[agent]
    fin, sh, sh2 = T["final"], T.get("shield"), T["shield2"]
    recs = pooled(group, agent, "final")
    recs2 = pooled(group, agent, "shield2")
    assert abs(summ(recs)["sr"] - fin["sr"]) < 1e-9 and abs(summ(recs2)["sr"] - sh2["sr"]) < 1e-9
    return dict(sr=fin["sr"], coll=fin["collision"], sd=sd1([r["sr"] for r in P["final"]]),
                csr=sh["sr"] if sh else None, ccoll=sh["collision"] if sh else None,
                ssr=sh2["sr"], sto=sh2["timeout"], ssd=sd1([r["sr"] for r in P["shield2"]]),
                scoll_n=sum(r["end"] == "collision" for r in recs2),
                tg=tgoal(recs), svr=fin["svr"], n=len(recs), seeds=len(P["final"]),
                tg_seedmean=st.mean([tgoal([r for r in load(RUNS / group / f"{agent}_s{s}" / "eval_final.json")["records"]])
                                     for s in seeds_of(group, agent)]) if fin["sr"] > 0 else float("nan"))


def build_drivers():
    D = {
        "stock_mapless": classical_row("sf", "sf_shield", "sf_shield2"),
        "tuned_mapless": classical_row("sf2", None, None),
        "mapless_sac": learned_row("tier2_mapless", "fixed_equal"),
        "astar": classical_row("astar_pp", "astar_pp_shield", "astar_pp_shield2"),
        "stock_planner": classical_row("planner_sf", "planner_sf_shield", "planner_sf_shield2"),
        "pp": classical_row("planner_pp", "planner_pp_shield", "planner_pp_shield2"),
        "orca": classical_row("planner_orca", "planner_orca_shield", "planner_orca_shield2"),
        "tuned": classical_row("planner_sf2_tuned", "planner_sf2_tuned_shield", "planner_sf2_tunedb_shield2"),
        "sac": learned_row("tier2", "fixed_equal"),
        "blind": learned_row("tier2", "vanilla"),
        "safe": learned_row("tier2_safe", "safe_heavy"),
        "td3": learned_row("tier2_td3", "td3_equal"),
        "ppo": learned_row("tier2_ppo", "ppo_equal"),
        "res": learned_row("tier2_residual", "residual_sf"),
    }
    names = {"stock_mapless": r"Stock SF$^{\dagger}$", "tuned_mapless": "Tuned SF", "mapless_sac": "Mapless SAC-CARS",
             "astar": r"A$^{*}$ + pursuit$^{\dagger}$", "stock_planner": r"Stock SF$^{\dagger}$",
             "pp": r"Pure pursuit$^{\dagger}$", "orca": r"ORCA$^{\ddagger}$", "tuned": "Tuned SF", "sac": "SAC-CARS",
             "blind": "SAC-blind", "safe": "SAC-CARS-safe", "td3": "TD3-CARS", "ppo": "PPO-CARS", "res": "Residual"}

    def cell_sr(x, sd):
        if x is None:
            return "n/a"
        return pm_small(x, sd) if sd is not None and sd == sd else p1(x)

    def line(k):
        r = D[k]
        tg = "n/a" if r["tg"] != r["tg"] else f"{r['tg']:.1f}"
        cells = [names[k], cell_sr(r["sr"], r["sd"]), p1(r["coll"]), p1(r["csr"]), p1(r["ccoll"]),
                 cell_sr(r["ssr"], r["ssd"]), p1(r["sto"]), tg, p3(r["svr"])]
        fx = FACT_DRV[k]
        def e1(v):
            return "n/a" if v is None else f"{v:.1f}"
        chk(f"V {k} SR/coll/cone SR/cone coll/v2 SR/v2 TO/t_goal/SVR",
            "/".join([p1(r["sr"]), p1(r["coll"]), p1(r["csr"]), p1(r["ccoll"]), p1(r["ssr"]), p1(r["sto"]), tg,
                      f"{r['svr']:.4f}"]),
            "/".join([e1(fx[0]), e1(fx[1]), e1(fx[2]) if fx[2] is not None else "(not in facts)",
                      e1(fx[3]) if fx[3] is not None else "(not in facts)", e1(fx[4]), e1(fx[5]), e1(fx[6]),
                      f"{fx[7]}"]))
        return " & ".join(cells) + " \\\\"

    td3 = D["td3"]
    chk("V TD3 t_goal: pooled (make_paper_tables rule) vs mean of seed means",
        f"pooled {td3['tg']:.2f} s", f"facts_sim_curated 51.5 s = mean of seed means {td3['tg_seedmean']:.2f} s")
    chk("V swept-filter contacts Mapless SAC-CARS / PPO-CARS", f"{D['mapless_sac']['scoll_n']} / {D['ppo']['scoll_n']}",
        "1 / 1 (census)")
    for k in ("tuned", "sac", "blind", "safe", "td3", "res", "astar", "stock_planner", "pp", "orca", "stock_mapless"):
        if D[k]["scoll_n"]:
            chk(f"V swept-filter contacts {k}", str(D[k]["scoll_n"]), "0")
    chk("V seed SD (ddof=1): SAC-CARS / mapless / residual v2 / TD3",
        f"{100 * D['sac']['sd']:.1f} / {100 * D['mapless_sac']['sd']:.1f} / {100 * D['res']['ssd']:.1f} / {100 * D['td3']['sd']:.1f}",
        "1.7 / 3.0 / 0.5 / 13.8 (corrections item 28)")

    def sub(text):
        return f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{text}}}}}\\\\"

    rows = [sub("No route planner"), line("stock_mapless"), line("tuned_mapless"), line("mapless_sac"),
            "\\midrule", sub("Own A$^{*}$ path"), line("astar"),
            "\\midrule", sub("Route planner"), line("stock_planner"), line("pp"), line("orca"), line("tuned"),
            line("sac"), line("blind"), line("safe"), line("td3"), line("ppo"), line("res")]
    body = r"""%% Table V (tab:drivers). Generated by build_tables.py. Do not edit by hand.
%% Provenance: the no-filter and cone-filter files of Stock SF, A* + pursuit and Pure pursuit predate
%% seed_base logging (facts_sim_corrections.md section 14); their swept-filter runs record seed_base 1000.
%% TD3-CARS t_goal uses the make_paper_tables.py rule (pooled over seeds); facts_sim_curated.md 51.5 s is the mean of seed means.
\begin{table}[!t]
\centering
\caption{Local Drivers on Tier 2 Without a Filter, With the Cone Filter and With the Swept Filter (500 Episodes per Seed)}
\label{tab:drivers}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{}l cc cc cc cc@{}}
\toprule
& \multicolumn{2}{c}{No filter} & \multicolumn{2}{c}{Cone filter} & \multicolumn{2}{c}{Swept filter} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Driver & SR & Coll & SR & Coll & SR & T/O & $t_{\mathrm{goal}}$ & SVR\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""" + note(r"\columnwidth", r"""
SR, Coll and T/O are the shares of episodes, in \%, that end in success, contact or timeout. All rows use the same 500 episode seeds. Learned rows pool 5 seeds, except 3 for SAC-CARS-safe, TD3-CARS and Residual. The value after $\pm$ is the sample standard deviation over seeds. Classical drivers are deterministic, with one run each. $t_{\mathrm{goal}}$ is the mean time in seconds over successful episodes and SVR the share of steps with a person closer than 0.8\,m, both without a filter. The planner uses $\kappa=3$, or $\kappa=0$ for SAC-blind. Under the swept filter only Mapless SAC-CARS and PPO-CARS made contact, once each in 2,500 episodes. n/a: not run, or no successful episode. $^{\dagger}$Stock settings, not tuned. $^{\ddagger}$Lightly tuned: time horizon and minimum speed chosen from four settings on seeds 5000+.
""") + "\\end{table}\n"
    write("tab_drivers.tex", body)


# ============================================================================ Table VI
def build_compliance():
    t2, pt2 = table("tier2"), load(RES / "tier2" / "per_task.json")
    ab, ptab = table("tier2_planner_ablation"), load(RES / "tier2_planner_ablation" / "per_task.json")
    cell = {
        ("cars", "geo"): (ptab["fixed_equal"]["final"]["STAT_medicine"]["cvr"], ab["fixed_equal"]["final"]["cvr"],
                          ab["fixed_equal"]["final"]["sr"], len(per_seed("tier2_planner_ablation")["fixed_equal"]["final"])),
        ("cars", "cult"): (pt2["fixed_equal"]["final"]["STAT_medicine"]["cvr"], t2["fixed_equal"]["final"]["cvr"],
                           t2["fixed_equal"]["final"]["sr"], len(per_seed("tier2")["fixed_equal"]["final"])),
        ("blind", "geo"): (pt2["vanilla"]["final"]["STAT_medicine"]["cvr"], t2["vanilla"]["final"]["cvr"],
                           t2["vanilla"]["final"]["sr"], len(per_seed("tier2")["vanilla"]["final"])),
        ("blind", "cult"): (ptab["vanilla"]["final"]["STAT_medicine"]["cvr"], ab["vanilla"]["final"]["cvr"],
                            ab["vanilla"]["final"]["sr"], len(per_seed("tier2_planner_ablation")["vanilla"]["final"])),
    }
    fact = {("cars", "cult"): (0.0008, 0.110, 80.4, 5), ("cars", "geo"): (0.0806, 0.138, 80.4, 3),
            ("blind", "cult"): (0.0026, 0.113, 81.1, 3), ("blind", "geo"): (0.0797, 0.147, 84.2, 5)}
    for k, v in cell.items():
        f = fact[k]
        chk(f"VI {k} CVR_STAT/CVR/SR/seeds", f"{v[0]:.4f}/{v[1]:.3f}/{100 * v[2]:.1f}/{v[3]}",
            f"{f[0]:.4f}/{f[1]:.3f}/{f[2]:.1f}/{f[3]}")
    rows = []
    for rk, label in (("cars", "CARS"), ("blind", "Culture-blind")):
        g, c = cell[(rk, "geo")], cell[(rk, "cult")]
        rows.append(f"{label} & {p3(g[0])} & {p3(g[1])} & {p1(g[2])} & {p3(c[0])} & {p3(c[1])} & {p1(c[2])} \\\\")
    body = r"""%% Table VI (tab:compliance). Generated by build_tables.py. Do not edit by hand.
\begin{table}[!t]
\centering
\caption{Driver Reward Versus Planner Price on Tier 2 (No Filter, 5 or 3 Seeds per Cell, 500 Episodes per Seed)}
\label{tab:compliance}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}l ccc ccc@{}}
\toprule
& \multicolumn{3}{c}{Geometric planner, $\kappa=0$} & \multicolumn{3}{c}{Cultural planner, $\kappa=3$}\\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Driver reward & CVR\textsubscript{STAT} & CVR & SR (\%) & CVR\textsubscript{STAT} & CVR & SR (\%)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""" + note(r"\columnwidth", r"""
The rows are SAC-CARS and SAC-blind. CVR\textsubscript{STAT} is the CVR on the 100 STAT-task episodes per seed, whose shortest route from the south-corridor spawn crosses the prayer room. CVR covers all five tasks. The matched cells, CARS with $\kappa=3$ and culture-blind with $\kappa=0$, come from the main campaign with 5 seeds. The swapped cells come from the planner ablation with 3 seeds. Under either reward, the cultural planner lowers CVR\textsubscript{STAT} from about 0.08 to 0.003 or less. Switching the reward changes it by 0.002 at most.
""") + "\\end{table}\n"
    write("tab_compliance.tex", body)
    # statements in the note: check them
    geo = [cell[(r, "geo")][0] for r in ("cars", "blind")]
    cult = [cell[(r, "cult")][0] for r in ("cars", "blind")]
    assert all(0.075 < g < 0.085 for g in geo) and all(c < 0.0035 for c in cult)
    d_reward = max(abs(geo[0] - geo[1]), abs(cult[0] - cult[1]))
    assert round(d_reward, 3) <= 0.002
    chk("VI note: CVR_STAT geo / cult / max reward effect", f"{geo[0]:.4f},{geo[1]:.4f} / {cult[0]:.4f},{cult[1]:.4f} / {d_reward:.4f}",
        "about 0.08 / <= 0.003 / <= 0.002")


# ============================================================================ Table VII
FACT_ROB = {  # facts_sim_curated.md section 8: (SR, coll) fe plain, fe v2, van plain, van v2, sf2 plain, sf2 v2
    "clean": ((80.0, 20.0), (94.0, 0.0), (84.2, 15.8), (96.0, 0.0), (95.4, 4.6), (97.8, 0.0)),
    "clean50": ((84.0, None), (100.0, None), (88.0, None), (94.0, None), (92.0, None), (96.0, None)),
    "unmapped": ((71.0, 28.8), (85.6, 0.0), (76.0, 23.8), (88.0, 0.0), (85.6, 8.4), (88.6, 0.0)),
    "lidar": ((78.4, 21.6), (97.2, 0.2), (81.4, 18.6), (97.4, 0.0), (96.4, 3.6), (98.8, 0.0)),
    "act": ((78.8, 21.2), (97.4, 1.8), (82.2, 17.8), (95.8, 1.4), (93.4, 6.6), (97.0, 3.0)),
    "noyield": ((48.8, 51.2), (56.8, 42.4), (48.8, 51.2), (55.6, 43.6), (65.4, 34.6), (60.0, 39.4)),
}


def rob_recs(cond, drv, shield):
    suf = "_shield2" if shield else ""
    if cond in ("clean", "clean50"):
        if drv == "sf2":
            recs = load(CL2 / ("planner_sf2_tunedb_shield2.json" if shield else "planner_sf2_tuned.json"))["records"]
        else:
            g, a, s = {"fe": ("tier2", "fixed_equal", 1), "van": ("tier2", "vanilla", 0)}[drv]
            recs = load(RUNS / g / f"{a}_s{s}" / ("eval_shield2.json" if shield else "eval_final.json"))["records"]
        if cond == "clean50":
            recs = recs[:50]   # records are stored in episode order (learned records carry no "ep" key)
            assert all(r.get("ep", i) == i for i, r in enumerate(recs))
        return recs
    if drv == "sf2":
        base = [CL2 / f"planner_sf2__unmapped_L{i}__sf2{suf}.json" for i in range(10)] if cond == "unmapped" \
            else [CL2 / f"planner_sf2__{cond}__sf2{suf}.json"]
    else:
        base = [ROB / f"unmapped_L{i}__{drv}{suf}.json" for i in range(10)] if cond == "unmapped" \
            else [ROB / f"{cond}__{drv}{suf}.json"]
    recs = []
    for f in base:
        recs += load(f)["records"]
    return recs


def build_robust():
    labels = [("clean", "Clean"), ("clean50", "Clean, eps.\ 0-49"),
              ("unmapped", "Unmapped (A1)"), ("lidar", "LiDAR noise (A4)"),
              ("act", "Actuation noise (A2)"), ("noyield", "Non-yielding (A3)")]
    rows = []
    for cond, label in labels:
        cells = [label]
        j = 0
        for drv in ("fe", "van", "sf2"):
            for shield in (False, True):
                s = summ(rob_recs(cond, drv, shield))
                cells.append(f"{p1(s['sr'])}/{p1(s['coll'])}")
                fx = FACT_ROB[cond][j]
                chk(f"VII {cond} {drv} {'v2' if shield else 'plain'} SR/coll (n={s['n']})",
                    f"{p1(s['sr'])}/{p1(s['coll'])}",
                    f"{fx[0]:.1f}/" + ("(not in facts)" if fx[1] is None else f"{fx[1]:.1f}"))
                j += 1
        rows.append(" & ".join(cells) + " \\\\")
        if cond == "clean50":
            rows.append("\\midrule")
    body = r"""%% Table VII (tab:robust). Generated by build_tables.py. Do not edit by hand.
\begin{table}[!t]
\centering
\caption{Breaking Each Assumption on Purpose on Tier 2 (SR/Coll in \%, Single Seeds)}
\label{tab:robust}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}l cc cc cc@{}}
\toprule
& \multicolumn{2}{c}{SAC-CARS} & \multicolumn{2}{c}{SAC-blind} & \multicolumn{2}{c}{Tuned SF}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Condition & None & Swept & None & Swept & None & Swept\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""" + note(r"\columnwidth", r"""
Each cell gives SR/Coll without a filter (None) and with the swept filter (Swept), over 500 episodes, or 50 in the second row. SAC-CARS is seed 1 and SAC-blind is seed 0. All drivers use the planner. A1 adds six boxes unknown to the planner to each of ten layouts, run on episodes 0-49, so the second row is its clean baseline. A4 adds 5\% range noise and 2\% dropped rays for driver and filter. A2 adds Gaussian noise with a standard deviation of 10\% of the command range after the filter. In A3, pedestrians ignore the robot.
""") + "\\end{table}\n"
    write("tab_robust.tex", body)


# ============================================================================ Table VIII
GZ_NAMES = [("fixed_equal", "SAC-CARS"), ("residual_sf", "Residual"), ("classical_sf2", "Tuned SF"),
            ("vanilla", "SAC-blind"), ("risk_field", "SAC-risk"), ("mapless_fixed_equal", "Mapless SAC-CARS")]
FACT_GZ = {  # facts_gazebo.md 4.3 (on|off per cell: walled 8, walled 45, open 8, open 45), 4.7 depth, 4.6 touch runs
    "residual_sf": ("10/0", "7/0", "10/10", "10/9", 0.58, 3), "classical_sf2": ("9/0", "4/1", "10/10", "10/10", 0.45, 10),
    "fixed_equal": ("10/8", "10/10", "10/9", "10/10", 0.62, 9), "vanilla": ("10/9", "10/8", "10/7", "10/6", 3.98, 2),
    "risk_field": ("10/7", "10/8", "10/3", "10/6", 3.98, 4), "mapless_fixed_equal": ("8/7", "8/0", "10/6", "10/7", 1.95, 7),
}


def build_gazebo():
    with open(GZ, encoding="utf-8", newline="") as f:
        R = [r for r in csv.DictReader(f) if r["campaign"] == "repeats"]
    assert len(R) == 480
    cells = [("walled", "busy"), ("walled", "crowd45"), ("open", "busy"), ("open", "crowd45")]

    def cnt(pol, tier, peds, esc):
        rr = [r for r in R if r["policy"] == pol and r["tier"] == tier and r["pedestrians"] == peds and r["escape"] == esc]
        assert len(rr) == 10, (pol, tier, peds, esc, len(rr))
        return sum(int(r["reached"]) for r in rr)

    rows, tests = [], []
    tot = {c: [0, 0] for c in cells}
    touch_total, touch_open = 0, 0
    for pol, name in GZ_NAMES:
        line = [name]
        got = []
        for tier, peds in cells:
            on, off = cnt(pol, tier, peds, "on"), cnt(pol, tier, peds, "off")
            tot[(tier, peds)][0] += on
            tot[(tier, peds)][1] += off
            tests.append(((tier, peds, pol), fisher2(on, 10 - on, off, 10 - off)))
            line.append(f"{on}/{off}")
            got.append(f"{on}/{off}")
        opn = [float(r["prayer_max_depth_m"]) for r in R if r["policy"] == pol and r["tier"] == "open"]
        assert len(opn) == 40
        depth = st.median(opn)
        prr = [r for r in R if r["policy"] == pol]
        trun = sum(int(float(r["ped_touches"])) > 0 for r in prr)
        touch_total += trun
        touch_open += sum(int(float(r["ped_touches"])) > 0 for r in prr if r["tier"] == "open")
        line += [f"{depth:.2f}", str(trun)]
        if pol in ("classical_sf2", "risk_field"):
            rows.append(" & ".join(line) + " \\\\")
            rows.append("\\midrule")
        else:
            rows.append(" & ".join(line) + " \\\\")
        fx = FACT_GZ[pol]
        chk(f"VIII {name} cells/depth/touch runs", "/".join(got) + f" | {depth:.2f} | {trun}",
            "/".join(fx[:4]) + f" | {fx[4]:.2f} | {fx[5]}")
    rows.append("\\midrule")
    rows.append("All drivers (of 60) & " + " & ".join(f"{tot[c][0]}/{tot[c][1]}" for c in cells)
                + f" & & {touch_total} \\\\")
    chk("VIII totals per cell on/off", " ".join(f"{tot[c][0]}/{tot[c][1]}" for c in cells), "57/31 49/27 60/45 60/48")
    chk("VIII touch runs total / in open ward", f"{touch_total} / {touch_open}", "35 / 0")
    on_all = sum(v[0] for v in tot.values())
    off_all = sum(v[1] for v in tot.values())
    chk("VIII recovery on vs off pooled", f"{on_all}/240 vs {off_all}/240, Fisher p = {fisher2(on_all, 240 - on_all, off_all, 240 - off_all):.2g}",
        "226/240 vs 151/240, p = 9.5e-18")
    # Holm over the 24 per-cell recovery tests
    tests.sort(key=lambda t: t[1])
    m = len(tests)
    holm, running = [], 0.0
    for i, (k, p) in enumerate(tests):
        running = max(running, min(1.0, (m - i) * p))
        holm.append((k, p, running))
    nominal = [h for h in holm if h[1] < 0.05]
    surv = [h for h in holm if h[2] < 0.05]
    chk("VIII per-cell recovery tests: nominal p<0.05 / Holm survivors", f"{len(nominal)} / {len(surv)}: "
        + "; ".join(f"{k[0]} {k[1]} {k[2]} p={p:.2g} holm={hp:.2g}" for k, p, hp in nominal), "5 / 3 (facts 4.4)")
    p_sac_sf = fisher2(10, 0, 4, 6)
    chk("VIII walled 45 with recovery SAC-CARS 10/10 vs Tuned SF 4/10", f"p = {p_sac_sf:.2g}", "p = 0.011")
    for k, n in ((10, 10), (7, 10), (4, 10), (0, 10)):
        lo, hi = wilson(k, n)
        chk(f"VIII Wilson {k}/{n}", f"[{100 * lo:.1f}, {100 * hi:.1f}]", {10: "[72.2, 100]", 7: "[39.7, 89.2]",
                                                                         4: "[16.8, 68.7]", 0: "[0, 27.8]"}[k])
    body = r"""%% Table VIII (tab:gazebo). Generated by build_tables.py from ros_sim/paper_data/runs.csv (campaign = repeats). Do not edit by hand.
\begin{table}[!t]
\centering
\caption{Goal-Reached Counts, Prayer-Room Depth and Touches in the 480-Run Gazebo Study (10 Runs per Cell)}
\label{tab:gazebo}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}l cc cc cc@{}}
\toprule
& \multicolumn{2}{c}{Walled ward} & \multicolumn{2}{c}{Open ward} & Prayer & Touch\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Driver & 8 & 45 & 8 & 45 & depth (m) & runs\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""" + note(r"\columnwidth", r"""
Each cell counts the runs, out of 10 pedestrian seeds, that reached the goal within 300\,s with/without stall recovery, an intervention the benchmark did not have. Heads 8 and 45 give the walker count. Wilson 95\% intervals are [72, 100]\% for 10/10, [40, 89]\% for 7/10, [17, 69]\% for 4/10 and [0, 28]\% for 0/10. Fisher tests of with against without recovery give $p<0.05$ in 5 of the 24 cells. Three survive a Holm correction: Residual and Tuned SF with 8 walkers and Mapless SAC-CARS with 45, all in the walled ward. There, with 45 walkers and recovery, SAC-CARS reached 10/10 and Tuned SF 4/10 ($p=0.011$). Prayer depth is the median over the 40 open-ward runs of the deepest point inside the prayer room, from its nearest edge. The room centre lies 4.0\,m deep. Touch runs count the runs, out of 80, with a walker centre within 0.35\,m of the robot centre, all in the walled ward. Localisation is ground truth on a prebuilt map, and all runs share one route.
""") + "\\end{table}\n"
    write("tab_gazebo.tex", body)


# ============================================================================ main
if __name__ == "__main__":
    build_main()
    build_drivers()
    build_compliance()
    build_robust()
    build_gazebo()
    print("\n==== number check against facts_sim_curated.md / facts_gazebo.md ====")
    bad = 0
    for what, got, exp in CHECKS:
        flag = "" if got == exp else "   <-- differs"
        bad += bool(flag)
        print(f"{what}\n    computed: {got}\n    facts   : {exp}{flag}")
    print(f"\n{len(CHECKS)} checks, {bad} differ textually (review each above)")
