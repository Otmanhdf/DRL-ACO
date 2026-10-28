"""DQN parameter controller for ACO (DRL-ACO) and the ACO-based variants."""
import itertools
import numpy as np
from core import (args, min_min, ls_moves, F_of, aco_block, seed_rng,
                  objectives, normalized, pheromone_entropy)

# ------------------------------------------------------------------ settings
ANTS = 20          # A
ITERS = 100        # I  (ACO iterations per instance)
K_BLOCK = 10       # K  (iterations per DQN step) -> H = I/K = 10 steps
LS_PASSES = 20

ALPHA4 = [0.5, 1.0, 1.5, 2.0]
BETA4 = [1.0, 2.0, 3.0, 4.0]
RHO5 = [0.1, 0.3, 0.5, 0.7, 0.9]


def grid(al, be, rh):
    return [tuple(x) for x in itertools.product(al, be, rh)]


ACTION_SETS = {
    "A80": grid(ALPHA4, BETA4, RHO5),
    "A27": grid([0.5, 1.25, 2.0], [1.0, 2.5, 4.0], [0.1, 0.5, 0.9]),
    "A252": grid([0.5, 0.8, 1.1, 1.4, 1.7, 2.0], [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                 [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]),
}


def restricted_actions(free, tuned):
    """Action set where only the parameters in `free` ('a','b','r') vary; the
    others are taken from `tuned` (alpha, beta, rho), resolved per instance."""
    al = ALPHA4 if "a" in free else [None]
    be = BETA4 if "b" in free else [None]
    rh = RHO5 if "r" in free else [None]
    return [tuple(x) for x in itertools.product(al, be, rh)]


def resolve(action, tuned):
    a, b, r = action
    return (tuned[0] if a is None else a, tuned[1] if b is None else b,
            tuned[2] if r is None else r)


# ------------------------------------------------------------------ episode
STATE_DIM = 17


class ACOEpisode:
    """One ACO run on one instance, split into H blocks of K iterations."""

    def __init__(self, inst, w, seed, iters=ITERS, K=K_BLOCK, ants=ANTS):
        self.inst, self.w = inst, np.asarray(w, float)
        self.ET, self.P, self.Pi, self.c = args(inst)
        self.m, self.b = inst["m"], inst["bounds"]
        self.K, self.H, self.A = K, iters // K, ants
        seed_rng(seed)
        # sigma*(0): Min-Min followed by local search (same for every variant)
        sig = min_min(self.ET, self.m)
        ls_moves(sig, self.ET, self.P, self.Pi, self.c, self.m, self.b, self.w, LS_PASSES)
        self.gbest = sig
        self.gF = F_of(sig, self.ET, self.P, self.Pi, self.c, self.m, self.b, self.w)
        self.F0 = self.gF
        self.tau = np.full((inst["n"], self.m), 1.0 / (0.3 * self.gF))
        self.t = 0
        self.prev_gF = self.gF
        self.meanF = self.gF
        self.stag = 0
        self.last_imp = 0.0
        self.prev_action = (1.0, 2.0, 0.3)
        self.trace = [self.gF]
        L = inst["L"]
        self.static = [inst["n"] / 1000.0, self.m / 50.0, L.std() / L.mean()]

    def state(self):
        M, E, C, LB = objectives(self.gbest, self.ET, self.P, self.Pi, self.c, self.m)
        nm = normalized(dict(M=M, E=E, C=C, LB=LB), self.b)
        CT = np.bincount(self.gbest, weights=self.ET[np.arange(len(self.gbest)), self.gbest],
                         minlength=self.m)
        u = CT / M
        a, bb, r = self.prev_action
        return np.array([self.t / self.H, *nm, self.last_imp * 10.0, self.stag / self.H,
                         pheromone_entropy(self.tau), u.mean(), u.std(),
                         (self.meanF - self.gF) / self.gF, *self.static,
                         (a - 0.5) / 1.5, (bb - 1.0) / 3.0, r], dtype=np.float64)

    def step(self, alpha, beta, rho):
        before = self.gF
        gF, meanF, hist = aco_block(self.tau, self.ET, self.P, self.Pi, self.c,
                                    self.inst["order"], self.m, self.b, self.w,
                                    float(alpha), int(round(beta * 2)), float(rho),
                                    self.K, self.A, self.gbest, self.gF, LS_PASSES)
        self.gF, self.meanF = gF, meanF
        self.trace.extend(list(np.minimum.accumulate(np.minimum(hist, before))))
        self.last_imp = (before - gF) / before
        self.stag = 0 if gF < before - 1e-12 else self.stag + 1
        self.prev_action = (alpha, beta, rho)
        self.t += 1
        reward = 100.0 * (before - gF) / self.F0      # relative improvement, %
        return reward, self.t >= self.H


# ------------------------------------------------------------------ DQN (numpy)
class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.m = [np.zeros_like(x) for x in self.W + self.b]
        self.v = [np.zeros_like(x) for x in self.W + self.b]
        self.k = 0

    def forward(self, x, keep=False):
        hs = [x]
        for i in range(len(self.W)):
            x = x @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            hs.append(x)
        return (x, hs) if keep else x

    def copy_from(self, o):
        self.W = [x.copy() for x in o.W]
        self.b = [x.copy() for x in o.b]

    def train_step(self, x, a, y, lr=1e-3):
        q, hs = self.forward(x, keep=True)
        B = x.shape[0]
        err = q[np.arange(B), a] - y
        # Huber loss gradient
        g = np.zeros_like(q)
        g[np.arange(B), a] = np.clip(err, -1.0, 1.0) / B
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (hs[i] > 0)
        self.k += 1
        params = self.W + self.b
        grads = gW + gb
        for p, gr, mm, vv in zip(params, grads, self.m, self.v):
            mm *= 0.9; mm += 0.1 * gr
            vv *= 0.999; vv += 0.001 * gr * gr
            mh = mm / (1 - 0.9 ** self.k)
            vh = vv / (1 - 0.999 ** self.k)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean(np.minimum(np.abs(err), 1.0) * np.abs(err)))


class DQNAgent:
    def __init__(self, n_actions, seed, hidden=(128, 64), gamma=0.9, lr=1e-3,
                 buffer=10000, batch=64, target_every=50, eps0=1.0, eps_min=0.1,
                 decay=0.01):
        self.rng = np.random.default_rng(seed)
        self.nA = n_actions
        self.q = MLP([STATE_DIM, *hidden, n_actions], self.rng)
        self.qt = MLP([STATE_DIM, *hidden, n_actions], self.rng)
        self.qt.copy_from(self.q)
        self.gamma, self.lr, self.batch, self.target_every = gamma, lr, batch, target_every
        self.cap = buffer
        self.S = np.zeros((buffer, STATE_DIM)); self.S2 = np.zeros((buffer, STATE_DIM))
        self.Aa = np.zeros(buffer, np.int64); self.R = np.zeros(buffer); self.D = np.zeros(buffer)
        self.ptr = 0; self.size = 0; self.steps = 0
        self.eps0, self.eps_min, self.decay = eps0, eps_min, decay

    def eps(self, episode):
        return self.eps_min + (self.eps0 - self.eps_min) * np.exp(-self.decay * episode)

    def act(self, s, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        return int(np.argmax(self.q.forward(s[None, :])[0]))

    def remember(self, s, a, r, s2, d):
        i = self.ptr
        self.S[i], self.Aa[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def learn(self):
        if self.size < self.batch:
            return None
        idx = self.rng.integers(0, self.size, self.batch)
        q2 = self.qt.forward(self.S2[idx]).max(1)
        y = self.R[idx] + self.gamma * (1.0 - self.D[idx]) * q2
        loss = self.q.train_step(self.S[idx], self.Aa[idx], y, self.lr)
        self.steps += 1
        if self.steps % self.target_every == 0:
            self.qt.copy_from(self.q)
        return loss

    def save(self, path):
        np.savez(path, *self.q.W, *self.q.b)

    def load(self, path):
        d = np.load(path)
        k = len(self.q.W)
        self.q.W = [d[f"arr_{i}"] for i in range(k)]
        self.q.b = [d[f"arr_{i}"] for i in range(k, 2 * k)]


# ------------------------------------------------------------------ controllers
def run_aco(inst, w, seed, controller, **kw):
    """controller(ep) -> (alpha, beta, rho) for the next block."""
    ep = ACOEpisode(inst, w, seed, **kw)
    done = False
    while not done:
        a, b, r = controller(ep)
        _, done = ep.step(a, b, r)
    return ep


def fixed_controller(params):
    return lambda ep: params


def random_controller(actions, seed, tuned=None):
    rng = np.random.default_rng(seed)
    return lambda ep: resolve(actions[rng.integers(len(actions))], tuned or (1, 2, .3))


def dqn_controller(agent, actions, tuned=None, once=False):
    memo = {}

    def ctl(ep):
        if once and "a" in memo:
            return memo["a"]
        a = resolve(actions[agent.act(ep.state(), 0.0)], tuned or (1, 2, .3))
        memo["a"] = a
        return a
    return ctl
