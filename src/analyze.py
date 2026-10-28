"""Statistics, tables and figures from results/*.csv."""
import glob, os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True); os.makedirs("tables", exist_ok=True)
METRICS = ["M", "E", "C", "LB", "F"]
REF = "DRL-ACO"
ORDER = ["FCFS", "Min-Min", "Min-Min + LS", "Memetic GA", "Memetic PSO", "NSGA-II", "DRL-only",
         "Fixed-ACO (default)", "Fixed-ACO (tuned)", "Random-ACO", "DQN-once", "DRL-ACO"]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); k = len(p)
    adj = np.empty(k); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (k - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def a12(x, y):
    """P(X < Y) + 0.5 P(X = Y): probability DRL-ACO (x) is lower (better)."""
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def hodges_lehmann(d, conf=0.95):
    d = np.asarray(d); n = len(d)
    wa = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    N = len(wa)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    return np.median(wa), wa[k], wa[N - 1 - k]


def load(mode="main", workload="uniform"):
    fs = sorted(glob.glob(f"results/{mode}_{workload}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


def pairwise(df, ref=REF):
    rows = []
    for n, g in df.groupby("scale"):
        R = g[g.method == ref].sort_values("rep")
        for meth in g.method.unique():
            if meth == ref:
                continue
            B = g[g.method == meth].sort_values("rep")
            for k in METRICS:
                x, y = R[k].values, B[k].values
                d = x - y
                if np.allclose(d, 0):
                    W, p = np.nan, 1.0
                else:
                    res = stats.wilcoxon(x, y, zero_method="wilcox")
                    W, p = res.statistic, res.pvalue
                hl, lo, hi = hodges_lehmann(d)
                rows.append(dict(scale=n, baseline=meth, metric=k, W=W, p=p,
                                 A12=a12(x, y), HL=hl, CI_lo=lo, CI_hi=hi,
                                 rel_diff_pct=100 * (x.mean() - y.mean()) / y.mean()))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    t["outcome"] = np.where(t.p_holm >= 0.05, "tie", np.where(t.HL < 0, "win", "loss"))
    return t


def friedman(df):
    out = []
    for n, g in df.groupby("scale"):
        piv = g.pivot(index="rep", columns="method", values="F")
        piv = piv[[m for m in ORDER if m in piv.columns]]
        st, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
        ranks = piv.rank(axis=1).mean()
        out.append(dict(scale=n, chi2=st, p=p, **{f"rank::{k}": v for k, v in ranks.items()}))
    return pd.DataFrame(out)


def summary(df):
    s = df.groupby(["method", "scale"])[METRICS + ["time_s"]].agg(["mean", "std"])
    return s


if __name__ == "__main__":
    df = load()
    summary(df).to_csv("tables/summary_main.csv")
    pw = pairwise(df); pw.to_csv("tables/S1_pairwise_main.csv", index=False)
    fr = friedman(df); fr.to_csv("tables/friedman_main.csv", index=False)
    print(fr.T)
    print(pw.groupby(["baseline", "outcome"]).size().unstack(fill_value=0))
