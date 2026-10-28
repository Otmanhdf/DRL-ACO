"""Worked example (Appendix A): 6 tasks, 3 VMs, H=3 blocks of K=2 iterations, 4 ants,
trained DRL-ACO policy (seed 0)."""
import numpy as np, json
from core import make_instance, objectives, normalized, F_of, args
from agent import ACOEpisode, ACTION_SETS
from evaluate import load_policy
W = np.full(4, .25)
inst = make_instance(6, 3, 1012)
ag, acts, _ = load_policy("drlaco")
ep = ACOEpisode(inst, W, 7, iters=6, K=2, ants=4)
out = {"L": inst["L"].round(0).tolist(), "mips": inst["mips"].round(0).tolist(), "P": inst["P"].round(1).tolist(),
       "c": (inst["c"]*1e3).round(4).tolist(), "ET": inst["ET"].round(2).tolist(), "bounds": inst["bounds"].round(5).tolist(),
       "steps": []}
def snap(tag, a=None, r=None):
    M,E,C,LB = objectives(ep.gbest, *args(inst), inst["m"])
    out["steps"].append(dict(t=ep.t, tag=tag, action=a, reward=r, sigma=(ep.gbest+1).tolist(),
        M=round(M,2), E_Wh=round(E*1000,3), C_mUSD=round(C*1000,4), LB=round(LB,4), F=round(ep.gF,4),
        state=np.round(ep.state(),3).tolist()))
snap("start")
done=False
while not done:
    a = acts[ag.act(ep.state(), 0.0)]
    r, done = ep.step(*a)
    snap("block", list(a), round(r,3))
json.dump(out, open("results/worked_example.json","w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
