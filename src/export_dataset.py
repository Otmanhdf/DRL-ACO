"""Export the benchmark instances used in the paper to data/.

Every instance is fully determined by (n, m, seed, workload) through
core.make_instance; this script writes the generated values so that they can
be used without this code base.

    python src/export_dataset.py          # write data/
    python src/export_dataset.py --check  # regenerate and compare with data/

Files (one per workload x size x split):
    data/<split>_<workload>_<n>.npz with arrays
        seed  (R,)        instance seeds
        L     (R, n)      task lengths [MI]
        mips  (R, m)      VM capacities [MIPS]
        P     (R, m)      busy power [W]
        Pidle (R, m)      idle power [W]
        price (R, m)      VM price [USD/s]
        h     (R, n, m)   task-VM affinity factors
        (execution times are ET = L[:, None] / mips[None, :] * h  [s])
        bounds(R, 8)      normalisation bounds
                          [M_lo, M_hi, E_lo, E_hi, C_lo, C_hi, LB_lo, LB_hi]
    data/instances_index.csv  one row per instance (split, workload, n, m, rep, seed)
    data/example_csv/         the first 100-task test instance as plain CSV files
"""
import os, sys, csv
import numpy as np
from core import make_instance, VMS_PER_SCALE, SCALES
from evaluate import inst_seed

OUT = "data"
SPLITS = []
for n in SCALES:
    SPLITS.append(("test", "uniform", n, [inst_seed(n, r, "uniform") for r in range(30)]))
    SPLITS.append(("validation", "uniform", n, [20000 + n * 10 + k for k in range(5)]))
for n in (300, 1000):
    SPLITS.append(("test", "heavy", n, [inst_seed(n, r, "heavy") for r in range(30)]))


def build(n, workload, seeds):
    m = VMS_PER_SCALE[n]
    I = [make_instance(n, m, s, workload) for s in seeds]
    return dict(seed=np.array(seeds), L=np.stack([i["L"] for i in I]), mips=np.stack([i["mips"] for i in I]),
                P=np.stack([i["P"] for i in I]), Pidle=np.stack([i["Pi"] for i in I]),
                price=np.stack([i["c"] for i in I]), h=np.stack([i["h"] for i in I]),
                bounds=np.stack([i["bounds"] for i in I]))


def main(check=False):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split, wl, n, seeds in SPLITS:
        fn = f"{OUT}/{split}_{wl}_{n}.npz"
        d = build(n, wl, seeds)
        if check:
            z = np.load(fn)
            ok = all(np.array_equal(z[k], d[k]) for k in d)
            print(fn, "OK" if ok else "MISMATCH")
            if not ok:
                sys.exit(1)
        else:
            np.savez_compressed(fn, **d)
            print("wrote", fn)
        for r, s in enumerate(seeds):
            rows.append(dict(split=split, workload=wl, n=n, m=VMS_PER_SCALE[n], rep=r, seed=s, file=os.path.basename(fn)))
    if not check:
        with open(f"{OUT}/instances_index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        example_csv()


def example_csv():
    import pandas as pd
    d = f"{OUT}/example_csv"; os.makedirs(d, exist_ok=True)
    I = load_instance(f"{OUT}/test_uniform_100.npz", 0)
    pd.DataFrame(dict(task=range(I["n"]), length_MI=I["L"])).to_csv(f"{d}/tasks.csv", index=False)
    pd.DataFrame(dict(vm=range(I["m"]), mips=I["mips"], busy_power_W=I["P"], idle_power_W=I["Pi"],
                      price_USD_per_s=I["c"])).to_csv(f"{d}/vms.csv", index=False)
    pd.DataFrame(I["h"], columns=[f"vm{j}" for j in range(I["m"])]).to_csv(f"{d}/affinity_h.csv", index_label="task")
    pd.DataFrame(I["ET"], columns=[f"vm{j}" for j in range(I["m"])]).to_csv(f"{d}/exec_time_s.csv", index_label="task")
    pd.DataFrame([I["bounds"]], columns=["M_lo", "M_hi", "E_lo_kWh", "E_hi_kWh", "C_lo", "C_hi", "LB_lo", "LB_hi"]
                 ).to_csv(f"{d}/bounds.csv", index=False)


def load_instance(fn, rep):
    """Rebuild the instance dict used by the algorithms from an exported file."""
    z = np.load(fn)
    L, mips, h = z["L"][rep], z["mips"][rep], z["h"][rep]
    ET = L[:, None] / mips[None, :] * h
    return dict(n=L.shape[0], m=mips.shape[0], mips=mips, L=L, P=z["P"][rep], Pi=z["Pidle"][rep],
                c=z["price"][rep], h=h, ET=ET, order=np.argsort(-L).astype(np.int64),
                bounds=z["bounds"][rep], seed=int(z["seed"][rep]))


if __name__ == "__main__":
    main(check="--check" in sys.argv)
