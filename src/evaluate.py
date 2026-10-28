"""Evaluation on the 30 paired test instances.
python evaluate.py main <scale> [--workload uniform|heavy]
python evaluate.py ablation <scale>
python evaluate.py weights <scale>
Writes results/<mode>_<workload>_<scale>.csv (one row per method x replication)."""
import sys, os, csv, json, time, argparse
import numpy as np
from core import (make_instance, VMS_PER_SCALE, evaluate, fcfs, min_min, args)
from agent import (ACTION_SETS, restricted_actions, run_aco, fixed_controller,
                   random_controller, dqn_controller, DQNAgent, ITERS)
import baselines as BL
from train import load_tuned

REPS = 30
W0 = np.full(4, 0.25)


def inst_seed(n, r, workload):
    return 30000 + n * 100 + r + (500000 if workload == "heavy" else 0)


def load_policy(name, seed=0):
    cfg = json.load(open(f"policies/{name}_s{seed}.json"))
    acts = restricted_actions(cfg["free"], None) if cfg["free"] else ACTION_SETS[cfg["actions"]]
    ag = DQNAgent(len(acts), 0)
    ag.load(f"policies/{name}_s{seed}.npz")
    return ag, acts, cfg


def aco_methods(tuned_n, kind):
    ms = {}
    ag, acts, _ = load_policy("drlaco")
    if kind in ("main", "heavy", "weights"):
        ms["Fixed-ACO (default)"] = lambda: fixed_controller((1.0, 2.0, 0.3)), {}
        ms["Fixed-ACO (tuned)"] = lambda: fixed_controller(tuned_n), {}
        ms["Random-ACO"] = lambda: random_controller(ACTION_SETS["A80"], 99), {}
        ms["DQN-once"] = lambda: dqn_controller(ag, acts, once=True), {}
        ms["DRL-ACO"] = lambda: dqn_controller(ag, acts), {}
    if kind == "ablation":
        ms["Fixed-ACO (tuned)"] = lambda: fixed_controller(tuned_n), {}
        ms["DQN-once"] = lambda: dqn_controller(ag, acts, once=True), {}
        ms["DRL-ACO (full)"] = lambda: dqn_controller(ag, acts), {}
        for name, label in [("free_a", "Control alpha only"), ("free_b", "Control beta only"),
                            ("free_r", "Control rho only"), ("free_ab", "Control alpha, beta"),
                            ("free_ar", "Control alpha, rho"), ("free_br", "Control beta, rho"),
                            ("act27", "27 actions"), ("act252", "252 actions"),
                            ("negF", "Reward -F"), ("K5", "Block K = 5"), ("K20", "Block K = 20"),
                            ("small", "Trained on 100-300 tasks only")]:
            if not os.path.exists(f"policies/{name}_s0.npz"):
                continue
            a2, ac2, cfg = load_policy(name)
            ms[label] = (lambda a2=a2, ac2=ac2: dqn_controller(a2, ac2, tuned=tuned_n)), {"K": cfg["K"]}
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode"); p.add_argument("scale", type=int)
    p.add_argument("--workload", default="uniform")
    p.add_argument("--reps", type=int, default=REPS)
    a = p.parse_args()
    n, m = a.scale, VMS_PER_SCALE[a.scale]
    tuned = load_tuned()[n]
    rows = []
    disp = None
    weights = [W0]
    if a.mode == "weights":
        weights = [np.array(x) for x in [(0.25, 0.25, 0.25, 0.25), (0.5, 0.2, 0.15, 0.15),
                                         (0.2, 0.5, 0.15, 0.15), (0.15, 0.2, 0.5, 0.15),
                                         (0.15, 0.15, 0.2, 0.5)]]
    if a.mode in ("main",):
        disp = BL.train_dispatcher(0)
    kind = "heavy" if a.workload == "heavy" else a.mode
    ms = aco_methods(tuned, kind)
    cts = {}
    for r in range(a.reps):
        inst = make_instance(n, m, inst_seed(n, r, a.workload), a.workload)
        for wi, w in enumerate(weights):
            res = {}
            if a.mode != "ablation":
                t = time.time(); res["FCFS"] = (fcfs(inst), time.time() - t)
                t = time.time(); res["Min-Min"] = (min_min(inst["ET"], m), time.time() - t)
                t = time.time(); res["Min-Min + LS"] = (BL.minmin_ls(inst, w), time.time() - t)
                t = time.time(); res["Memetic GA"] = (BL.ga(inst, w, 7000 + r), time.time() - t)
                t = time.time(); res["Memetic PSO"] = (BL.pso(inst, w, 7000 + r), time.time() - t)
                t = time.time(); res["NSGA-II"] = (BL.nsga2_best(inst, w, 7000 + r), time.time() - t)
                if disp is not None:
                    t = time.time(); res["DRL-only"] = (BL.drl_only(inst, w, disp), time.time() - t)
            for name, (mk, kw) in ms.items():
                t = time.time()
                ep = run_aco(inst, w, 7000 + r, mk(), **kw)
                res[name] = (ep.gbest.copy(), time.time() - t)
            for name, (sig, dt) in res.items():
                ev = evaluate(sig, inst, w)
                rows.append(dict(mode=a.mode, workload=a.workload, scale=n, vms=m, rep=r,
                                 weights=wi, method=name, **ev, time_s=dt))
                if a.mode == "main" and n in (700, 1000) and wi == 0:
                    CT = np.bincount(sig, weights=inst["ET"][np.arange(n), sig], minlength=m)
                    cts.setdefault(name, []).append(CT)
        print(a.mode, a.workload, n, "rep", r, "done", flush=True)
    os.makedirs("results", exist_ok=True)
    fn = f"results/{a.mode}_{a.workload}_{n}.csv"
    with open(fn, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    if cts:
        np.savez(f"results/ct_{a.workload}_{n}.npz", **{k.replace(" ", "_"): np.array(v) for k, v in cts.items()})
    print("wrote", fn)


if __name__ == "__main__":
    main()
