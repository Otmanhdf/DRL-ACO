"""Baselines minimising the same objective F (Eq. 8) with the same budget as
ACO (20 x 100 = 2,000 solution evaluations, local search on the best solution
of each iteration, start from the Min-Min + local search solution)."""
import numpy as np
from core import (args, min_min, ls_moves, F_of, objectives, fval, normalized,
                  seed_rng, greedy_mct)
from agent import LS_PASSES, MLP, STATE_DIM, DQNAgent

POP, GENS = 20, 100


def _start(inst, w):
    ET, P, Pi, c = args(inst)
    m, b = inst["m"], inst["bounds"]
    sig = min_min(ET, m)
    ls_moves(sig, ET, P, Pi, c, m, b, w, LS_PASSES)
    return sig


def minmin_ls(inst, w):
    return _start(inst, w)


# ------------------------------------------------------------------ memetic GA
def ga(inst, w, seed, pop=POP, gens=GENS, pc=0.9, pm=None):
    rng = np.random.default_rng(seed); seed_rng(seed)
    ET, P, Pi, c = args(inst); n, m, b = inst["n"], inst["m"], inst["bounds"]
    pm = pm or 2.0 / n
    X = rng.integers(0, m, (pop, n)).astype(np.int64)
    X[0] = _start(inst, w)
    fit = np.array([F_of(x, ET, P, Pi, c, m, b, w) for x in X])
    for _ in range(gens):
        def tour():
            i, j = rng.integers(0, pop, 2)
            return X[i] if fit[i] < fit[j] else X[j]
        Y = np.empty_like(X)
        Y[0] = X[np.argmin(fit)]                       # elitism
        for k in range(1, pop):
            p1, p2 = tour(), tour()
            child = np.where(rng.random(n) < 0.5, p1, p2) if rng.random() < pc else p1.copy()
            mut = rng.random(n) < pm
            child[mut] = rng.integers(0, m, mut.sum())
            Y[k] = child
        fy = np.array([F_of(y, ET, P, Pi, c, m, b, w) for y in Y])
        ib = 1 + np.argmin(fy[1:])
        fy[ib] = ls_moves(Y[ib], ET, P, Pi, c, m, b, w, LS_PASSES)
        X, fit = Y, fy
    return X[np.argmin(fit)].copy()


# ------------------------------------------------------------------ memetic PSO
def pso(inst, w, seed, pop=POP, iters=GENS, wi=0.7, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed); seed_rng(seed)
    ET, P, Pi, c = args(inst); n, m, b = inst["n"], inst["m"], inst["bounds"]
    X = rng.uniform(0, m, (pop, n))
    s0 = _start(inst, w)
    X[0] = s0 + 0.5
    V = rng.uniform(-1, 1, (pop, n))
    dec = lambda x: np.clip(x.astype(np.int64), 0, m - 1)
    fit = np.array([F_of(dec(x), ET, P, Pi, c, m, b, w) for x in X])
    pb, pf = X.copy(), fit.copy()
    g = np.argmin(pf); gb, gf = pb[g].copy(), pf[g]
    vmax = 0.5 * m
    for _ in range(iters):
        r1, r2 = rng.random((pop, n)), rng.random((pop, n))
        V = np.clip(wi * V + c1 * r1 * (pb - X) + c2 * r2 * (gb - X), -vmax, vmax)
        X = np.mod(X + V, m)
        fit = np.array([F_of(dec(x), ET, P, Pi, c, m, b, w) for x in X])
        ib = np.argmin(fit)
        s = dec(X[ib]); fit[ib] = ls_moves(s, ET, P, Pi, c, m, b, w, LS_PASSES)
        X[ib] = s + 0.5
        imp = fit < pf
        pb[imp], pf[imp] = X[imp], fit[imp]
        g = np.argmin(pf)
        if pf[g] < gf:
            gb, gf = pb[g].copy(), pf[g]
    return dec(gb)


# ------------------------------------------------------------------ NSGA-II
def objs_norm(sig, inst):
    M, E, C, LB = objectives(sig, *args(inst), inst["m"])
    return np.array(normalized(dict(M=M, E=E, C=C, LB=LB), inst["bounds"]))


def nondom_sort(Fm):
    N = len(Fm)
    dom = (Fm[:, None, :] <= Fm[None, :, :]).all(2) & (Fm[:, None, :] < Fm[None, :, :]).any(2)
    cnt = dom.sum(0)            # number of solutions dominating j
    rank = np.full(N, -1); r = 0
    cur = np.where(cnt == 0)[0]
    while len(cur):
        rank[cur] = r
        for i in cur:
            cnt[dom[i]] -= 1
        cnt[cur] = -1
        cur = np.where(cnt == 0)[0]; r += 1
    return rank


def crowding(Fm):
    N, k = Fm.shape
    d = np.zeros(N)
    for j in range(k):
        o = np.argsort(Fm[:, j]); d[o[[0, -1]]] = np.inf
        rngj = Fm[o[-1], j] - Fm[o[0], j] + 1e-12
        d[o[1:-1]] += (Fm[o[2:], j] - Fm[o[:-2], j]) / rngj
    return d


def nsga2(inst, w, seed, pop=100, gens=20, pm=None, seeds_extra=()):
    """4-objective NSGA-II; returns final population (assignments, objectives)."""
    rng = np.random.default_rng(seed); seed_rng(seed)
    n, m = inst["n"], inst["m"]
    pm = pm or 2.0 / n
    X = rng.integers(0, m, (pop, n)).astype(np.int64)
    X[0] = _start(inst, w); X[1] = min_min(inst["ET"], m); X[2] = greedy_mct(inst["ET"], inst["order"], m)
    for k, s in enumerate(seeds_extra):
        X[3 + k] = s
    Fm = np.array([objs_norm(x, inst) for x in X])
    for _ in range(gens):
        rank, cd = nondom_sort(Fm), crowding(Fm)
        def tour():
            i, j = rng.integers(0, pop, 2)
            if rank[i] != rank[j]:
                return X[i] if rank[i] < rank[j] else X[j]
            return X[i] if cd[i] > cd[j] else X[j]
        Y = np.empty_like(X)
        for k in range(pop):
            p1, p2 = tour(), tour()
            ch = np.where(rng.random(n) < 0.5, p1, p2)
            mut = rng.random(n) < pm
            ch[mut] = rng.integers(0, m, mut.sum())
            Y[k] = ch
        FY = np.array([objs_norm(y, inst) for y in Y])
        XX, FF = np.vstack([X, Y]), np.vstack([Fm, FY])
        rank = nondom_sort(FF); cd = crowding(FF)
        order = np.lexsort((-cd, rank))[:pop]
        X, Fm = XX[order], FF[order]
    return X, Fm


def nsga2_best(inst, w, seed, **kw):
    X, Fm = nsga2(inst, w, seed, **kw)
    k = np.argmin(Fm @ w)
    return X[k].copy()


# ------------------------------------------------------------------ DRL-only
RULES = 5


def _dispatch_state(k, n, i, inst, CT, b):
    L = inst["L"]
    s = np.zeros(STATE_DIM)
    s[0] = k / n; s[1] = L[i] / L.max()
    u = CT / b[0]
    s[2], s[3], s[4], s[5] = u.mean(), u.std(), u.max(), u.min()
    s[6] = inst["n"] / 1000.0; s[7] = inst["m"] / 50.0
    return s


def _rule_vm(rule, i, inst, CT):
    ET, P, Pi, c = args(inst)
    if rule == 0:
        return int(np.argmin(CT + ET[i]))            # minimum completion time
    if rule == 1:
        return int(np.argmin(ET[i]))                 # fastest / best affinity
    if rule == 2:
        return int(np.argmin((P - Pi) * ET[i]))      # least marginal energy
    if rule == 3:
        return int(np.argmin(c * ET[i]))             # cheapest
    return int(np.argmin(CT))                        # least loaded


def dispatch_episode(inst, w, agent, eps, learn):
    ET, P, Pi, c = args(inst); n, m, b = inst["n"], inst["m"], inst["bounds"]
    CT = np.zeros(m); sig = np.full(n, -1, np.int64)
    prevF = 0.0; ret = 0.0
    s = None
    for k, i in enumerate(inst["order"]):
        s = _dispatch_state(k, n, i, inst, CT, b)
        a = agent.act(s, eps)
        j = _rule_vm(a, i, inst, CT)
        sig[i] = j; CT[j] += ET[i, j]
        done = sig >= 0
        M = CT.max(); mu = CT.mean()
        E = (P * CT + Pi * (M - CT)).sum() / 3.6e6
        C = (c[sig[done]] * ET[np.where(done)[0], sig[done]]).sum() / n
        LB = CT.std() / mu
        Fk = fval(M, E, C, LB, b, w)
        r = -100.0 * (Fk - prevF); prevF = Fk; ret += r
        if learn:
            s2 = _dispatch_state(min(k + 1, n - 1), n, inst["order"][min(k + 1, n - 1)], inst, CT, b)
            agent.remember(s, a, r, s2, float(k == n - 1))
            if k % 4 == 0:
                agent.learn()
    return sig, ret


def train_dispatcher(seed, episodes=150, scales=(100, 300, 500, 700, 1000)):
    from core import make_instance, VMS_PER_SCALE
    ag = DQNAgent(RULES, seed, gamma=0.99, decay=0.02)
    rng = np.random.default_rng(seed)
    W = np.full(4, 0.25)
    for e in range(episodes):
        n = int(rng.choice(scales))
        inst = make_instance(n, VMS_PER_SCALE[n], 50000 + 1000 * seed + e)
        dispatch_episode(inst, W, ag, ag.eps(e), True)
    return ag


def drl_only(inst, w, agent):
    sig, _ = dispatch_episode(inst, w, agent, 0.0, False)
    ls_moves(sig, *args(inst), inst["m"], inst["bounds"], w, LS_PASSES)
    return sig
