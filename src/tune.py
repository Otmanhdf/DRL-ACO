"""Exhaustive tuning of Fixed-ACO over the 80-action grid on validation
instances (seeds disjoint from training and evaluation).
Usage: python tune.py <scale> ; writes results/tune_<scale>.csv"""
import sys, os, csv, time
import numpy as np
from core import make_instance, VMS_PER_SCALE
from agent import ACTION_SETS, run_aco, fixed_controller

W = np.full(4, 0.25)
N_VAL = 5


def main(n):
    os.makedirs("results", exist_ok=True)
    insts = [make_instance(n, VMS_PER_SCALE[n], 20000 + n * 10 + k) for k in range(N_VAL)]
    rows = []
    t0 = time.time()
    for ai, (a, b, r) in enumerate(ACTION_SETS["A80"]):
        Fs = [run_aco(inst, W, 777 + k, fixed_controller((a, b, r))).gF
              for k, inst in enumerate(insts)]
        rows.append(dict(scale=n, alpha=a, beta=b, rho=r, meanF=np.mean(Fs),
                         **{f"F{k}": f for k, f in enumerate(Fs)}))
    with open(f"results/tune_{n}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    best = min(rows, key=lambda d: d["meanF"])
    print(n, "best", best["alpha"], best["beta"], best["rho"], round(best["meanF"], 5),
          "time", round(time.time() - t0, 1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
