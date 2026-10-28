"""All result figures of the paper (Figs. 2-11) (left: trends with +-1 SD
bands; right: mean and 95% CI at 1,000 tasks), from results/*.csv.
Run from the repository root: python src/make_figures.py"""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

ORDER = ["FCFS", "Min-Min", "Min-Min + LS", "Memetic GA", "Memetic PSO", "NSGA-II", "DRL-only",
         "Fixed-ACO (default)", "Fixed-ACO (tuned)", "Random-ACO", "DQN-once", "DRL-ACO"]
COL = {"FCFS": "#e74c3c", "Min-Min": "#e67e22", "Min-Min + LS": "#f1c40f", "Memetic GA": "#2ecc71",
       "Memetic PSO": "#1abc9c", "NSGA-II": "#3498db", "DRL-only": "#9b59b6",
       "Fixed-ACO (default)": "#8e44ad", "Fixed-ACO (tuned)": "#16a085", "Random-ACO": "#2980b9",
       "DQN-once": "#7f8c8d", "DRL-ACO": "#2c3e50"}
MK = {"FCFS": "o", "Min-Min": "s", "Min-Min + LS": "D", "Memetic GA": "^", "Memetic PSO": "v",
      "NSGA-II": "<", "DRL-only": ">", "Fixed-ACO (default)": "x", "Fixed-ACO (tuned)": "P",
      "Random-ACO": "*", "DQN-once": "h", "DRL-ACO": "o"}
SCALE = {"M": 1, "E": 1000, "C": 1000, "LB": 100}
LAB = {"M": "Makespan (s)", "E": "Energy (Wh)", "C": r"Cost ($10^{-3}$ USD/task)",
       "LB": r"Load imbalance ($\times10^{-2}$)"}

df = pd.concat([pd.read_csv(f"results/main_uniform_{n}.csv") for n in [100, 300, 500, 700, 1000]])


def metric_fig(k, fn, skip_left=("FCFS",), skip_right=("FCFS",)):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for m in ORDER:
        if m in skip_left:
            continue
        g = df[df.method == m].groupby("scale")[k].agg(["mean", "std"]) * SCALE[k]
        main = m == "DRL-ACO"
        a1.plot(g.index, g["mean"], marker=MK[m], ms=6 if main else 4, lw=2.8 if main else 1.1,
                ls="-" if main else "--", color=COL[m], label=m, zorder=5 if main else 2)
        a1.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COL[m], alpha=0.08)
    a1.set_xticks([100, 300, 500, 700, 1000]); a1.set_xlabel("Number of Tasks"); a1.set_ylabel(LAB[k])
    a1.grid(ls="--", alpha=0.4)
    meths = [m for m in ORDER if m not in skip_right]
    g = df[df.scale == 1000].groupby("method")[k].agg(["mean", "std", "count"]) * 1
    mu = g["mean"] * SCALE[k]; ci = 1.96 * g["std"] * SCALE[k] / np.sqrt(g["count"])
    y = np.arange(len(meths))[::-1]
    for yy, m in zip(y, meths):
        a2.errorbar(mu[m], yy, xerr=ci[m], fmt=MK[m], color=COL[m], ms=8 if m == "DRL-ACO" else 6,
                    capsize=3, lw=1.5)
    a2.set_yticks(y); a2.set_yticklabels(meths, fontsize=8.5)
    a2.set_xlabel(LAB[k] + " at 1,000 tasks"); a2.grid(axis="x", ls="--", alpha=0.4)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{OUT}/{fn}", dpi=200); plt.close(fig)


metric_fig("M", "fig1_makespan.png")
metric_fig("E", "fig2_energy.png")
metric_fig("C", "fig3_cost.png")
metric_fig("LB", "fig4_lb.png", skip_right=("FCFS", "Min-Min"))


# ---------- ablation (two panels, F with 95% CI) ----------
ab = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/ablation_uniform_*.csv"))])
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (n, g) in zip(axs, ab.groupby("scale")):
    s = g.groupby("method").F.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    ci = 1.96 * s["std"] / np.sqrt(s["count"])
    ax.barh(range(len(s)), s["mean"], xerr=ci, capsize=2.5,
            color=["#2c3e50" if "full" in k else "#95a5a6" for k in s.index])
    ax.set_yticks(range(len(s))); ax.set_yticklabels([k.replace("alpha", "α").replace("beta", "β").replace("rho", "ρ") for k in s.index], fontsize=8.5)
    lo, hi = (s["mean"] - ci).min(), (s["mean"] + ci).max()
    ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_xlabel("Composite objective F (mean, 95% CI)"); ax.set_title(f"{n:,} tasks")
    ax.grid(axis="x", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_ablation.png", dpi=200); plt.close(fig)


# ---------- relative F vs Min-Min + LS ----------
base = df[df.method == "Min-Min + LS"].set_index(["scale", "rep"]).F
meths = [m for m in ORDER if m not in ("FCFS", "Min-Min", "Min-Min + LS", "DRL-only", "Memetic PSO", "NSGA-II")]
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(5); wd = 0.8 / len(meths)
for i, m in enumerate(meths):
    s = df[df.method == m].set_index(["scale", "rep"]).F
    r = (100 * (s - base) / base).groupby("scale").agg(["mean", "std", "count"])
    ax.bar(x + (i - len(meths) / 2 + 0.5) * wd, r["mean"], wd, yerr=1.96 * r["std"] / np.sqrt(r["count"]),
           capsize=2, color=COL[m], label=m, error_kw=dict(lw=0.8))
ax.axhline(0, color="grey", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["100", "300", "500", "700", "1,000"])
ax.set_xlabel("Number of Tasks"); ax.set_ylabel("Change in F vs Min-Min + LS (%)")
ax.legend(fontsize=8, ncol=4, frameon=False, loc="lower right"); ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_relative_F.png", dpi=200); plt.close(fig)


# ---------- trade-off front, one instance ----------
z = np.load("results/front_300_0.npz"); A, B = z["A"], z["B"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, (i, j, li, lj) in zip(axs, [(0, 2, "Normalized makespan", "Normalized cost"),
                                    (1, 3, "Normalized energy", "Normalized load imbalance")]):
    ax.scatter(B[:, i], B[:, j], s=14, color="#3498db", alpha=0.6, label="NSGA-II (final population)")
    ax.scatter(A[:, i], A[:, j], s=70, color="#2c3e50", marker="*", label="DRL-ACO (10 weight vectors)")
    ax.set_xlabel(li); ax.set_ylabel(lj); ax.grid(ls="--", alpha=0.4)
axs[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig10_front.png", dpi=200); plt.close(fig)

# ---------- tuning grid, per-VM completion times, training curves ----------
import figures_extra as FX
FX.tuning_heat(f"{OUT}/fig6_tuning.png")
FX.ct_boxplots(f"{OUT}/fig8_ct.png")
FX.training(f"{OUT}/fig9_training.png")
print("ok")
