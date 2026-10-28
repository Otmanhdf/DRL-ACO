"""Figures for the revised paper."""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import load, ORDER

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
STY = {  # distinct marker + line style per method (Reviewer 3, minor 1)
    "FCFS": ("o", ":"), "Min-Min": ("s", ":"), "Min-Min + LS": ("D", "--"),
    "Memetic GA": ("^", "--"), "Memetic PSO": ("v", "--"), "NSGA-II": ("<", "--"),
    "DRL-only": (">", "-."), "Fixed-ACO (default)": ("x", "-."), "Fixed-ACO (tuned)": ("P", "-"),
    "Random-ACO": ("*", "-."), "DQN-once": ("h", "-"), "DRL-ACO": ("o", "-")}
LAB = {"M": "Makespan (s)", "E": "Energy (kWh)", "C": "Cost (USD/task)", "LB": "Load imbalance (CV)",
       "F": "Composite objective F"}


def trends(df, fn, metrics=("M", "E", "C", "LB"), skip=("FCFS",)):
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.0))
    for ax, k in zip(axs, metrics):
        for meth in ORDER:
            if meth in skip or meth not in df.method.unique():
                continue
            g = df[df.method == meth].groupby("scale")[k].agg(["mean", "std"])
            mk, ls = STY[meth]
            lw = 2.2 if meth == "DRL-ACO" else 1.0
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker=mk, ls=ls, lw=lw, ms=4,
                        capsize=2, label=meth, color="black" if meth == "DRL-ACO" else None)
        ax.set_xlabel("Number of tasks"); ax.set_ylabel(LAB[k])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(fn, dpi=200); plt.close(fig)


def rel_F(df, fn):
    """F relative to Min-Min + LS (paired), mean and 95% CI, per scale."""
    base = df[df.method == "Min-Min + LS"].set_index(["scale", "rep"]).F
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    meths = [m for m in ORDER if m not in ("FCFS", "Min-Min", "Min-Min + LS", "DRL-only") and m in df.method.unique()]
    scales = sorted(df.scale.unique()); x = np.arange(len(scales)); wdt = 0.8 / len(meths)
    for i, meth in enumerate(meths):
        s = df[df.method == meth].set_index(["scale", "rep"]).F
        r = 100 * (s - base) / base
        g = r.groupby("scale").agg(["mean", "std", "count"])
        ci = 1.96 * g["std"] / np.sqrt(g["count"])
        ax.bar(x + (i - len(meths) / 2 + 0.5) * wdt, g["mean"], wdt, yerr=ci, capsize=1.5,
               label=meth, color="black" if meth == "DRL-ACO" else None, error_kw=dict(lw=0.7))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Change in F vs Min-Min + LS (%)")
    ax.legend(fontsize=6.5, ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


def ct_boxplots(fn):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    for ax, n in zip(axs, [700, 1000]):
        f = f"results/ct_uniform_{n}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        keys = ["Min-Min", "Min-Min_+_LS", "Fixed-ACO_(tuned)", "DRL-ACO"]
        data = [(d[k] / d[k].max(1, keepdims=True)).ravel() for k in keys]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels([k.replace("_", " ") for k in keys], rotation=20, fontsize=7)
        ax.set_ylabel("VM completion time / makespan"); ax.set_title(f"{n} tasks (30 instances)")
    fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


def training(fn):
    fs = sorted(glob.glob("results/trainlog_drlaco_s*.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    curves, losses = [], []
    for f in fs:
        d = pd.read_csv(f)
        curves.append(d.improvement_pct.rolling(20, min_periods=1).mean().values)
        losses.append(d.loss.rolling(20, min_periods=1).mean().values)
    for arr, ax, yl in [(curves, axs[0], "Improvement over start (%), 20-ep. mean"),
                        (losses, axs[1], "TD loss (Huber), 20-ep. mean")]:
        A = np.array(arr); mu, sd = A.mean(0), A.std(0)
        ax.plot(mu, color="black"); ax.fill_between(range(len(mu)), mu - sd, mu + sd, alpha=0.25)
        ax.set_xlabel("Training episode"); ax.set_ylabel(yl)
    axs[0].set_title(f"mean ± SD over {len(fs)} training seeds", fontsize=8)
    fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


def tuning_heat(fn):
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
    for ax, n in zip(axs, [100, 300, 500, 700, 1000]):
        d = pd.read_csv(f"results/tune_{n}.csv")
        piv = d.groupby(["beta", "rho"]).meanF.min().unstack()
        im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
        b = d.loc[d.meanF.idxmin()]
        ax.set_title(f"{n} tasks: best ({b.alpha:g}, {b.beta:g}, {b.rho:g})", fontsize=8)
        ax.set_xlabel("rho")
        plt.colorbar(im, ax=ax, fraction=0.046)
    axs[0].set_ylabel("beta (best alpha per cell)")
    fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


def ablation_bars(fn):
    fs = sorted(glob.glob("results/ablation_uniform_*.csv"))
    if not fs:
        return
    df = pd.concat([pd.read_csv(f) for f in fs])
    fig, axs = plt.subplots(1, len(fs), figsize=(5 * len(fs), 3.6))
    axs = np.atleast_1d(axs)
    for ax, (n, g) in zip(axs, df.groupby("scale")):
        s = g.groupby("method").F.agg(["mean", "std", "count"]).sort_values("mean")
        ci = 1.96 * s["std"] / np.sqrt(s["count"])
        ax.barh(range(len(s)), s["mean"], xerr=ci, capsize=2,
                color=["black" if "full" in k else "0.6" for k in s.index])
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
        lo = (s["mean"] - ci).min(); hi = (s["mean"] + ci).max()
        ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_xlabel("F (mean, 95% CI)"); ax.set_title(f"{n} tasks")
    fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)


if __name__ == "__main__":  # legacy figure set (not used in the paper)
    df = load()
    trends(df, "figures/fig2_metrics_by_scale.png")
    rel_F(df, "figures/fig3_relative_F.png")
    ct_boxplots("figures/fig4_ct_distribution.png")
    training("figures/fig6_training.png")
    tuning_heat("figures/fig1b_tuning_grid.png")
    ablation_bars("figures/fig5_ablation.png")
    print("ok")
