"""Trade-off front: DRL-ACO over 10 weight vectors vs 4-objective NSGA-II with
the same total evaluation budget (10 x 2,000). Hypervolume (Monte Carlo) in
the normalised objective space, reference point 1.1 per objective.
python front.py <scale> <reps>"""
import sys, csv
import numpy as np
from core import make_instance, VMS_PER_SCALE
from agent import run_aco, dqn_controller
from evaluate import load_policy, inst_seed
import baselines as BL

WV = [np.array(x) / sum(x) for x in [(1, 1, 1, 1), (4, 1, 1, 1), (1, 4, 1, 1), (1, 1, 4, 1),
                                     (1, 1, 1, 4), (2, 2, 1, 1), (1, 2, 2, 1), (1, 1, 2, 2),
                                     (2, 1, 1, 2), (2, 1, 2, 1)]]
REFP = 1.1


def nd(Fm):
    keep = []
    for i in range(len(Fm)):
        if not any((Fm[j] <= Fm[i]).all() and (Fm[j] < Fm[i]).any() for j in range(len(Fm)) if j != i):
            keep.append(i)
    return Fm[keep]


def hv(Fm, lo, rng, N=200000):
    S = rng.uniform(lo, REFP, (N, Fm.shape[1]))
    dom = np.zeros(N, bool)
    for f in Fm:
        dom |= (S >= f).all(1)
    return dom.mean() * np.prod(REFP - lo)


def main(n, reps):
    ag, acts, _ = load_policy("drlaco")
    rows = []
    for r in range(reps):
        inst = make_instance(n, VMS_PER_SCALE[n], inst_seed(n, r, "uniform"))
        A = []
        for k, w in enumerate(WV):
            ep = run_aco(inst, w, 7000 + r, dqn_controller(ag, acts))
            A.append(BL.objs_norm(ep.gbest, inst))
        A = nd(np.array(A))
        X, Fm = BL.nsga2(inst, WV[0], 7000 + r, pop=100, gens=200)
        B = nd(Fm)
        lo = np.minimum(A.min(0), B.min(0)) - 1e-9
        rng = np.random.default_rng(r)
        ha, hb = hv(A, lo, rng), hv(B, lo, rng)
        U = nd(np.vstack([A, B]))
        # share of the combined non-dominated set contributed by each method
        ca = np.mean([any(np.allclose(u, a) for a in A) for u in U])
        rows.append(dict(scale=n, rep=r, HV_DRLACO=ha, HV_NSGA2=hb, n_DRLACO=len(A), n_NSGA2=len(B),
                         share_DRLACO=ca, share_NSGA2=1 - ca))
        print(rows[-1], flush=True)
        np.savez(f"results/front_{n}_{r}.npz", A=A, B=B)
    with open(f"results/front_{n}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
