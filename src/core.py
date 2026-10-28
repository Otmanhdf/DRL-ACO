"""Core simulator, objectives and ACO for the DRL-ACO study.

Model (independent, non-preemptive tasks; space-shared VMs, as in CloudSim):
  ET_ij = L_i / MIPS_j * h_ij       (h_ij: task-VM affinity, inconsistent ETC)
  CT_j  = sum of ET_ij over tasks assigned to VM j
  M     = max_j CT_j                                              [s]
  E     = sum_j [P_j CT_j + Pidle_j (M - CT_j)] / 3.6e6           [kWh]
  C     = (1/n) sum_i c_sigma(i) * ET_i,sigma(i)                  [USD/task]
  LB    = std(CT) / mean(CT)                                      [-]
"""
import numpy as np
from numba import njit

# Workload size -> number of VMs (Table 3 of the revised paper)
VMS_PER_SCALE = {100: 10, 300: 20, 500: 30, 700: 40, 1000: 50}
SCALES = [100, 300, 500, 700, 1000]


# ---------------------------------------------------------------- instances
def make_instance(n, m, seed, workload="uniform"):
    rng = np.random.default_rng(seed)
    if workload == "uniform":
        L = rng.uniform(1000.0, 20000.0, n)
    elif workload == "heavy":
        # heavy-tailed mix: 70% short, 25% medium, 5% very long (log-normal bodies)
        k = rng.choice(3, size=n, p=[0.70, 0.25, 0.05])
        mu = np.array([np.log(3000), np.log(15000), np.log(80000)])
        L = np.clip(np.exp(rng.normal(mu[k], 0.35)), 500.0, 300000.0)
    else:
        raise ValueError(workload)
    mips = rng.uniform(500.0, 2000.0, m)
    P = 60.0 + 0.08 * mips                  # W, busy power
    Pi = 0.6 * P                            # W, idle power
    c = 1.0e-4 * (mips / 1000.0) ** 2       # USD/s
    h = rng.uniform(0.6, 1.4, (n, m))       # task-VM affinity
    ET = L[:, None] / mips[None, :] * h
    order = np.argsort(-L).astype(np.int64) # LPT construction order
    inst = dict(n=n, m=m, L=L, mips=mips, P=P, Pi=Pi, c=c, h=h, ET=ET, order=order,
                seed=seed, workload=workload)
    inst["bounds"] = bounds(inst, rng)
    return inst


def args(inst):
    return inst["ET"], inst["P"], inst["Pi"], inst["c"]


def bounds(inst, rng):
    """Instance-level normalisation bounds, computable before scheduling and
    independent of the algorithms compared.
      lo: M = max(sum_i min_j ET_ij / m, max_i min_j ET_ij); E = sum_i min_j P_j ET_ij;
          C = (1/n) sum_i min_j c_j ET_ij; LB = 0
      hi: M, E, LB = mean over 20 uniformly random assignments;
          C = (1/n) sum_i max_j c_j ET_ij"""
    ET, P, Pi, c = args(inst)
    n, m = inst["n"], inst["m"]
    Mlo = max(ET.min(1).sum() / m, ET.min(1).max())
    Elo = (P[None, :] * ET).min(1).sum() / 3.6e6
    q = c[None, :] * ET / n
    Clo, Chi = q.min(1).sum(), q.max(1).sum()
    Ms, Es, LBs = [], [], []
    for _ in range(20):
        sig = rng.integers(0, m, n).astype(np.int64)
        M, E, C, LB = objectives(sig, ET, P, Pi, c, m)
        Ms.append(M); Es.append(E); LBs.append(LB)
    return np.array([Mlo, np.mean(Ms), Elo, np.mean(Es), Clo, Chi, 0.0, np.mean(LBs)])


# ---------------------------------------------------------------- objectives
@njit(cache=True)
def objectives(sig, ET, P, Pi, c, m):
    n = sig.shape[0]
    CT = np.zeros(m)
    C = 0.0
    for i in range(n):
        j = sig[i]
        t = ET[i, j]
        CT[j] += t
        C += c[j] * t
    M = CT.max()
    E = 0.0
    for j in range(m):
        E += P[j] * CT[j] + Pi[j] * (M - CT[j])
    mu = CT.mean()
    LB = np.sqrt(((CT - mu) ** 2).mean()) / mu
    return M, E / 3.6e6, C / n, LB


@njit(cache=True)
def fval(M, E, C, LB, b, w):
    return (w[0] * (M - b[0]) / (b[1] - b[0]) + w[1] * (E - b[2]) / (b[3] - b[2])
            + w[2] * (C - b[4]) / (b[5] - b[4]) + w[3] * (LB - b[6]) / (b[7] - b[6]))


@njit(cache=True)
def F_of(sig, ET, P, Pi, c, m, b, w):
    M, E, C, LB = objectives(sig, ET, P, Pi, c, m)
    return fval(M, E, C, LB, b, w)


def evaluate(sig, inst, w):
    M, E, C, LB = objectives(sig, *args(inst), inst["m"])
    return dict(M=M, E=E, C=C, LB=LB, F=fval(M, E, C, LB, inst["bounds"], w))


def normalized(ev, b):
    return [(ev["M"] - b[0]) / (b[1] - b[0]), (ev["E"] - b[2]) / (b[3] - b[2]),
            (ev["C"] - b[4]) / (b[5] - b[4]), (ev["LB"] - b[6]) / (b[7] - b[6])]


# ---------------------------------------------------------------- heuristics
def fcfs(inst):
    n, m = inst["n"], inst["m"]
    return (np.arange(n) % m).astype(np.int64)   # arrival order, round robin


@njit(cache=True)
def min_min(ET, m):
    n = ET.shape[0]
    CT = np.zeros(m)
    done = np.zeros(n, np.bool_)
    sig = np.zeros(n, np.int64)
    for _ in range(n):
        bi, bj, bv = -1, -1, 1e300
        for i in range(n):
            if done[i]:
                continue
            for j in range(m):
                v = CT[j] + ET[i, j]
                if v < bv:
                    bv, bi, bj = v, i, j
        done[bi] = True
        sig[bi] = bj
        CT[bj] += ET[bi, bj]
    return sig


@njit(cache=True)
def unit_costs(ET, P, Pi, c, b):
    """Normalised marginal energy (busy minus idle power) and cost of task i on VM j."""
    n, m = ET.shape
    e_n = np.empty((n, m))
    q_n = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            e_n[i, j] = (P[j] - Pi[j]) * ET[i, j] / 3.6e6 / (b[3] - b[2])
            q_n[i, j] = c[j] * ET[i, j] / n / (b[5] - b[4])
    return e_n, q_n


@njit(cache=True)
def idle_norm(Pi, b):
    return Pi.sum() / 3.6e6 / (b[3] - b[2])


@njit(cache=True)
def incr_cost(i, j, CT, ET, e_n, q_n, idn, b, w):
    """Heuristic incremental cost of assigning task i to VM j: marginal energy
    and cost, plus a penalty for the part of the task that lies beyond the
    makespan lower bound (makespan, idle energy and imbalance all grow there)."""
    over = CT[j] + ET[i, j] - b[0]
    if over < 0.0:
        over = 0.0
    elif over > ET[i, j]:
        over = ET[i, j]
    return ((w[0] / (b[1] - b[0]) + w[3] / (b[0] * (b[7] - b[6])) + w[1] * idn) * over
            + w[1] * e_n[i, j] + w[2] * q_n[i, j])


@njit(cache=True)
def greedy_composite(ET, P, Pi, c, order, m, b, w):
    """Deterministic construction used to initialise every ACO variant."""
    n = ET.shape[0]
    e_n, q_n = unit_costs(ET, P, Pi, c, b)
    idn = idle_norm(Pi, b)
    CT = np.zeros(m)
    sig = np.zeros(n, np.int64)
    for k in range(n):
        i = order[k]
        bv, bj = 1e300, 0
        for j in range(m):
            v = incr_cost(i, j, CT, ET, e_n, q_n, idn, b, w)
            if v < bv:
                bv, bj = v, j
        sig[i] = bj
        CT[bj] += ET[i, bj]
    return sig


@njit(cache=True)
def greedy_mct(ET, order, m):
    """LPT order, each task to the VM with minimum completion time (the
    deterministic limit of the ACO construction rule)."""
    n = ET.shape[0]
    CT = np.zeros(m)
    sig = np.zeros(n, np.int64)
    for k in range(n):
        i = order[k]
        bv, bj = 1e300, 0
        for j in range(m):
            v = CT[j] + ET[i, j]
            if v < bv:
                bv, bj = v, j
        sig[i] = bj
        CT[bj] += ET[i, bj]
    return sig


@njit(cache=True)
def ls_moves(sig, ET, P, Pi, c, m, b, w, passes):
    """First-improvement single-task move local search on F with O(m)
    incremental evaluation. Modifies sig in place; returns new F."""
    n = sig.shape[0]
    CT = np.zeros(m)
    Csum = 0.0
    for i in range(n):
        CT[sig[i]] += ET[i, sig[i]]
        Csum += c[sig[i]] * ET[i, sig[i]]
    sP = 0.0
    for j in range(m):
        sP += Pi[j]
    rM = w[0] / (b[1] - b[0]); rE = w[1] / (b[3] - b[2]) / 3.6e6
    rC = w[2] / (b[5] - b[4]) / n; rL = w[3] / (b[7] - b[6])

    def fcur(CT, Csum):
        M = 0.0; s1 = 0.0; s2 = 0.0; Eb = 0.0; Ei = 0.0
        for j in range(m):
            v = CT[j]
            if v > M:
                M = v
            s1 += v; s2 += v * v
            Eb += P[j] * v; Ei += Pi[j] * v
        mu = s1 / m
        var = s2 / m - mu * mu
        if var < 0.0:
            var = 0.0
        LB = np.sqrt(var) / mu
        E = Eb + sP * M - Ei
        return rM * (M - b[0]) + rE * (E - b[2] * 3.6e6) + rC * (Csum - b[4] * n) + rL * LB

    Fb = fcur(CT, Csum)
    for _ in range(passes):
        imp = False
        for i in range(n):
            a = sig[i]
            ta = ET[i, a]
            for j in range(m):
                if j == a:
                    continue
                tj = ET[i, j]
                CT[a] -= ta; CT[j] += tj
                Cn = Csum - c[a] * ta + c[j] * tj
                f = fcur(CT, Cn)
                if f < Fb - 1e-12:
                    Fb = f; Csum = Cn; sig[i] = j; a = j; ta = tj; imp = True
                else:
                    CT[a] += ta; CT[j] -= tj
        if not imp:
            break
    return Fb


# ---------------------------------------------------------------- ACO (MMAS)
@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@njit(cache=True)
def _powb(x, beta2):
    # x ** (beta2 / 2), beta in steps of 0.5
    r = 1.0
    for _ in range(beta2 // 2):
        r *= x
    if beta2 % 2 == 1:
        r *= np.sqrt(x)
    return r


@njit(cache=True)
def aco_block(tau, ET, P, Pi, c, order, m, b, w, alpha, beta2, rho, K, A,
              gbest, gF, ls_passes=20, q0=0.9, hmode=2, dep_gb=1):
    """K iterations of MAX-MIN ACO with fixed (alpha, beta=beta2/2, rho).
    tau (n x m) and gbest are modified in place.
    Returns (global-best F, mean ant F in last iteration, best F per iteration)."""
    n = ET.shape[0]
    tau_a = np.empty_like(tau)
    sol = np.empty((A, n), np.int64)
    Fs = np.empty(A)
    hist = np.empty(K)
    CT = np.empty(m)
    pr = np.empty(m)
    meanF = 0.0
    for it in range(K):
        for i in range(n):
            for j in range(m):
                tau_a[i, j] = tau[i, j] ** alpha
        for a in range(A):
            CT[:] = 0.0
            for k in range(n):
                i = order[k]
                s = 0.0
                vb = -1.0
                jb = 0
                for j in range(m):
                    if hmode == 0:
                        hv = 1.0 / (CT[j] + ET[i, j])
                    elif hmode == 1:
                        hv = 1.0 / ET[i, j]
                    else:
                        hv = 1.0 / (ET[i, j] * (1.0 + CT[j] / b[0]))
                    v = tau_a[i, j] * _powb(hv, beta2)
                    pr[j] = v
                    s += v
                    if v > vb:
                        vb = v
                        jb = j
                if np.random.random() < q0:
                    # ACS pseudo-random-proportional rule: exploit
                    jj = jb
                else:
                    r = np.random.random() * s
                    acc = 0.0
                    jj = m - 1
                    for j in range(m):
                        acc += pr[j]
                        if acc >= r:
                            jj = j
                            break
                sol[a, i] = jj
                CT[jj] += ET[i, jj]
            Fs[a] = F_of(sol[a], ET, P, Pi, c, m, b, w)
        ib = np.argmin(Fs)
        meanF = Fs.mean()
        if ls_passes > 0:
            Fs[ib] = ls_moves(sol[ib], ET, P, Pi, c, m, b, w, ls_passes)
        hist[it] = Fs[ib]
        if Fs[ib] < gF:
            gF = Fs[ib]
            gbest[:] = sol[ib]
        # MMAS update: evaporation, iteration-best deposit, trail limits
        tmax = 1.0 / (rho * max(gF, 1e-6))
        tmin = tmax / (2.0 * m)
        if dep_gb == 1:
            dsol = gbest
            dep = 1.0 / max(gF, 1e-6)
        else:
            dsol = sol[ib]
            dep = 1.0 / max(Fs[ib], 1e-6)
        for i in range(n):
            for j in range(m):
                t = (1.0 - rho) * tau[i, j]
                if dsol[i] == j:
                    t += dep
                if t > tmax:
                    t = tmax
                elif t < tmin:
                    t = tmin
                tau[i, j] = t
    return gF, meanF, hist


def pheromone_entropy(tau):
    p = tau / tau.sum(1, keepdims=True)
    h = -(p * np.log(p + 1e-300)).sum(1)
    return float(h.mean() / np.log(tau.shape[1]))


@njit(cache=True)
def local_search(sig, ET, P, Pi, c, m, b, w, passes):
    """First-improvement single-move local search on F (reference only)."""
    n = sig.shape[0]
    sig = sig.copy()
    Fb = F_of(sig, ET, P, Pi, c, m, b, w)
    for _ in range(passes):
        imp = False
        for i in range(n):
            j0 = sig[i]
            for j in range(m):
                if j == j0:
                    continue
                sig[i] = j
                f = F_of(sig, ET, P, Pi, c, m, b, w)
                if f < Fb - 1e-12:
                    Fb = f
                    j0 = j
                    imp = True
                else:
                    sig[i] = j0
            sig[i] = j0
        if not imp:
            break
    return sig, Fb
