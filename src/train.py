"""Train a DQN parameter controller.
python train.py <name> <seed> [--actions A80|A27|A252] [--free abr] [--K 10]
                 [--reward impr|negF] [--episodes 300] [--scales 100,300,...]
Writes policies/<name>_s<seed>.npz and results/trainlog_<name>_s<seed>.csv"""
import argparse, os, csv, json, time
import numpy as np
from core import make_instance, VMS_PER_SCALE, SCALES
from agent import (ACTION_SETS, restricted_actions, resolve, ACOEpisode, DQNAgent,
                   ITERS)

W = np.full(4, 0.25)


def load_tuned():
    t = {}
    for n in SCALES:
        import pandas as pd
        d = pd.read_csv(f"results/tune_{n}.csv")
        r = d.loc[d.meanF.idxmin()]
        t[n] = (float(r.alpha), float(r.beta), float(r.rho))
    return t


def actions_for(a):
    if a.free:
        return restricted_actions(a.free, None)
    return ACTION_SETS[a.actions]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name"); p.add_argument("seed", type=int)
    p.add_argument("--actions", default="A80"); p.add_argument("--free", default="")
    p.add_argument("--K", type=int, default=10); p.add_argument("--reward", default="impr")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--scales", default="100,300,500,700,1000")
    a = p.parse_args()
    os.makedirs("policies", exist_ok=True); os.makedirs("results", exist_ok=True)
    acts = actions_for(a)
    tuned = load_tuned() if a.free else None
    scales = [int(x) for x in a.scales.split(",")]
    ag = DQNAgent(len(acts), a.seed, decay=0.01)
    rng = np.random.default_rng(a.seed)
    log = []
    t0 = time.time()
    for e in range(a.episodes):
        n = int(rng.choice(scales))
        inst = make_instance(n, VMS_PER_SCALE[n], 10000 + 1000 * a.seed + e)
        ep = ACOEpisode(inst, W, 5000 + e, K=a.K)
        s = ep.state(); done = False; ret = 0.0; losses = []
        eps = ag.eps(e)
        while not done:
            ai = ag.act(s, eps)
            al, be, rh = resolve(acts[ai], tuned[n] if tuned else None)
            r, done = ep.step(al, be, rh)
            if a.reward == "negF":
                r = -100.0 * ep.gF / ep.F0 + 100.0
            s2 = ep.state()
            ag.remember(s, ai, r, s2, float(done))
            l = ag.learn()
            if l is not None:
                losses.append(l)
            ret += r; s = s2
        log.append(dict(episode=e, scale=n, eps=eps, ret=ret,
                        improvement_pct=100 * (ep.F0 - ep.gF) / ep.F0,
                        loss=np.mean(losses) if losses else np.nan,
                        time_s=time.time() - t0))
    ag.save(f"policies/{a.name}_s{a.seed}.npz")
    with open(f"results/trainlog_{a.name}_s{a.seed}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0])); wr.writeheader(); wr.writerows(log)
    json.dump(vars(a), open(f"policies/{a.name}_s{a.seed}.json", "w"))
    print(a.name, a.seed, "done", round(time.time() - t0, 1), "s",
          "last50 impr", round(np.mean([x["improvement_pct"] for x in log[-50:]]), 3))


if __name__ == "__main__":
    main()
