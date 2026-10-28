"""Markdown tables for the manuscript, written to tables/*.md."""
import glob, os
import numpy as np, pandas as pd
from analyze import load, pairwise, friedman, ORDER
from core import make_instance, VMS_PER_SCALE, SCALES
from evaluate import inst_seed

os.makedirs("tables", exist_ok=True)
FMT = {"M": "{:.2f} ± {:.2f}", "E": "{:.4f} ± {:.4f}", "C": "{:.3f} ± {:.3f}",
       "LB": "{:.4f} ± {:.4f}", "F": "{:.4f} ± {:.4f}"}
SCALE_C = 1000.0  # cost reported in 10^-3 USD per task


def md(df):
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out


def metric_table(df, k, order=ORDER):
    rows = []
    for meth in order:
        g = df[df.method == meth]
        if g.empty:
            continue
        row = {"Algorithm": meth}
        best = {}
        for n in sorted(df.scale.unique()):
            x = g[g.scale == n][k] * (SCALE_C if k == "C" else 1)
            row[f"{n} tasks"] = FMT[k].format(x.mean(), x.std())
        rows.append(row)
    t = pd.DataFrame(rows)
    # bold the lowest mean per column
    for n in sorted(df.scale.unique()):
        means = {meth: df[(df.method == meth) & (df.scale == n)][k].mean() for meth in t.Algorithm}
        bm = min(means, key=means.get)
        t.loc[t.Algorithm == bm, f"{n} tasks"] = "**" + t.loc[t.Algorithm == bm, f"{n} tasks"] + "**"
    return md(t)


def vm_table():
    rows = []
    for n in SCALES:
        m = VMS_PER_SCALE[n]
        I = [make_instance(n, m, inst_seed(n, r, "uniform")) for r in range(30)]
        rows.append({"Tasks": n, "VMs (m)": m, "Tasks per VM": n // m,
                     "MIPS range": "500–2,000", "Busy power (W)": "100–220",
                     "Price (10⁻³ USD/s)": "0.025–0.40",
                     "Makespan lower bound (s)": f"{np.mean([i['bounds'][0] for i in I]):.1f}"})
    return md(pd.DataFrame(rows))


def composite(df):
    fr = friedman(df)
    rows = []
    for meth in ORDER:
        if meth not in df.method.unique():
            continue
        row = {"Algorithm": meth}
        for n in sorted(df.scale.unique()):
            x = df[(df.method == meth) & (df.scale == n)].F
            rk = fr[fr.scale == n][f"rank::{meth}"].values[0]
            row[f"{n} tasks"] = f"{x.mean():.4f} ({rk:.2f})"
        rows.append(row)
    t = md(pd.DataFrame(rows))
    t += "\nFriedman: " + "; ".join(f"{int(r.scale)} tasks χ² = {r.chi2:.1f}, p = {r.p:.1e}"
                                     for _, r in fr.iterrows()) + "\n"
    return t


def wtl(pw, metric=None):
    t = pw if metric is None else pw[pw.metric == metric]
    s = t.groupby(["baseline", "outcome"]).size().unstack(fill_value=0)
    for c in ["win", "tie", "loss"]:
        if c not in s:
            s[c] = 0
    s = s[["win", "tie", "loss"]].reindex([m for m in ORDER if m in s.index])
    s.insert(0, "Baseline", s.index)
    return md(s.reset_index(drop=True))


def key_pairs(pw):
    keep = ["Min-Min + LS", "Memetic GA", "Fixed-ACO (default)", "Fixed-ACO (tuned)", "Random-ACO", "DQN-once"]
    t = pw[(pw.metric == "F") & pw.baseline.isin(keep)].copy()
    t["Δ F (HL, 95% CI) ×10⁻⁴"] = t.apply(lambda r: f"{r.HL*1e4:.2f} [{r.CI_lo*1e4:.2f}, {r.CI_hi*1e4:.2f}]", axis=1)
    t["Rel. diff (%)"] = t.rel_diff_pct.map(lambda v: f"{v:.2f}")
    t["A12"] = t.A12.map(lambda v: f"{v:.2f}")
    t["W"] = t.W.map(lambda v: "–" if pd.isna(v) else f"{v:.0f}")
    t["p (Holm)"] = t.p_holm.map(lambda v: f"{v:.3g}")
    t = t.rename(columns={"scale": "Tasks", "baseline": "DRL-ACO vs", "outcome": "Outcome"})
    return md(t[["Tasks", "DRL-ACO vs", "Rel. diff (%)", "Δ F (HL, 95% CI) ×10⁻⁴", "W", "p (Holm)", "A12", "Outcome"]])


def runtime(df):
    g = df[df.scale == 1000].groupby("method").time_s.mean()
    rows = [{"Algorithm": m, "Scheduling time per instance at 1,000 tasks (s)": f"{g[m]:.3f}"}
            for m in ORDER if m in g.index]
    return md(pd.DataFrame(rows))


def ablation():
    fs = sorted(glob.glob("results/ablation_uniform_*.csv"))
    if not fs:
        return ""
    df = pd.concat([pd.read_csv(f) for f in fs])
    out = ""
    for n, g in df.groupby("scale"):
        ref = g[g.method == "DRL-ACO (full)"].sort_values("rep").F.values
        rows = []
        from scipy import stats
        meths = list(g.groupby("method").F.mean().sort_values().index)
        ps = []
        for meth in meths:
            x = g[g.method == meth].sort_values("rep").F.values
            p = 1.0 if meth == "DRL-ACO (full)" or np.allclose(x, ref) else stats.wilcoxon(ref, x).pvalue
            ps.append(p)
        from analyze import holm
        adj = holm(ps)
        for meth, pa in zip(meths, adj):
            x = g[g.method == meth]
            rows.append({"Variant": meth, "F": f"{x.F.mean():.4f} ± {x.F.std():.4f}",
                         "Makespan (s)": f"{x.M.mean():.2f}", "LB": f"{x.LB.mean():.4f}",
                         "Δ vs full (%)": f"{100*(x.F.mean()-ref.mean())/ref.mean():+.2f}",
                         "p (Holm) vs full": "–" if meth == "DRL-ACO (full)" else f"{pa:.3g}"})
        out += f"\n**{n} tasks**\n\n" + md(pd.DataFrame(rows))
    return out


def weights():
    f = "results/weights_uniform_300.csv"
    if not os.path.exists(f):
        return ""
    df = pd.read_csv(f)
    names = ["Equal (0.25 each)", "Makespan priority (0.50, 0.20, 0.15, 0.15)",
             "Energy priority (0.20, 0.50, 0.15, 0.15)", "Cost priority (0.15, 0.20, 0.50, 0.15)",
             "LB priority (0.15, 0.15, 0.20, 0.50)"]
    rows = []
    for wi, g in df.groupby("weights"):
        piv = g.pivot(index="rep", columns="method", values="F")
        rk = piv.rank(axis=1).mean()
        mean = piv.mean()
        order = rk.sort_values()
        rows.append({"Weights (w1–w4)": names[wi], "DRL-ACO F": f"{mean['DRL-ACO']:.4f}",
                     "DRL-ACO avg. rank (of 11)": f"{rk['DRL-ACO']:.2f}",
                     "Best method (avg. rank)": f"{order.index[0]} ({order.iloc[0]:.2f})",
                     "Fixed-ACO (tuned) F": f"{mean['Fixed-ACO (tuned)']:.4f}"})
    return md(pd.DataFrame(rows))


def tuning():
    rows = []
    for n in SCALES:
        d = pd.read_csv(f"results/tune_{n}.csv")
        b = d.loc[d.meanF.idxmin()]; w = d.loc[d.meanF.idxmax()]
        dflt = d[(d.alpha == 1.0) & (d.beta == 2.0) & (d.rho == 0.3)].meanF.values[0]
        rows.append({"Tasks": n, "Best (α, β, ρ)": f"({b.alpha:g}, {b.beta:g}, {b.rho:g})",
                     "F best": f"{b.meanF:.4f}", "F default (1, 2, 0.3)": f"{dflt:.4f}",
                     "F worst": f"{w.meanF:.4f}",
                     "Spread best→worst (%)": f"{100*(w.meanF-b.meanF)/b.meanF:.1f}"})
    return md(pd.DataFrame(rows))


if __name__ == "__main__":
    df = load()
    pw = pairwise(df)
    parts = {"vm": vm_table(), "M": metric_table(df, "M"), "E": metric_table(df, "E"),
             "C": metric_table(df, "C"), "LB": metric_table(df, "LB"), "F": composite(df),
             "wtl_all": wtl(pw), "wtl_F": wtl(pw, "F"), "keypairs": key_pairs(pw),
             "runtime": runtime(df), "ablation": ablation(), "weights": weights(),
             "tuning": tuning()}
    hv = load("main", "heavy")
    if hv is not None:
        parts["heavyF"] = composite(hv)
        parts["heavy_keypairs"] = key_pairs(pairwise(hv))
    for k, v in parts.items():
        open(f"tables/{k}.md", "w").write(v)
    print("\n\n".join(f"## {k}\n{v}" for k, v in parts.items()))
