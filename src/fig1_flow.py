"""Fig. 1: DRL-ACO control loop."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(8, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, t, fc="#f2f2f2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
def arr(x1, y1, x2, y2, t="", dx=0.05, dy=0.12, ha="center"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9))
    if t: ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, t, fontsize=6.5, ha=ha)
box(0.2, 4.6, 2.6, 1.0, "Instance (n tasks, m VMs)\nbounds of Eq. (7)")
box(0.2, 2.9, 2.6, 1.1, "σ*(0) = Min-Min + local search\nτ = 1 / (0.3 F(σ*(0)))")
box(3.6, 4.4, 2.8, 1.3, "DQN  Q(s, a)\n17 → 128 → 64 → 80\nchoose a(t) = (α, β, ρ)", "#dde8f5")
box(3.6, 1.6, 2.8, 2.1, "K = 10 ACO iterations\n20 ants, ACS rule (q0 = 0.9),\nheuristic of Eq. (9)\nlocal search on iteration best\nMAX–MIN update, Eq. (10)", "#e8f2e0")
box(7.2, 2.9, 2.6, 1.1, "r(t) = 100·ΔF / F(σ*(0))\nstate s(t+1)")
box(7.2, 4.6, 2.6, 1.0, "Training only:\nreplay buffer, DQN update")
box(7.2, 0.4, 2.6, 1.0, "after H = 10 blocks:\nreturn σ*(H)")
arr(1.5, 4.6, 1.5, 4.0)
arr(2.8, 3.45, 3.6, 3.0, "s(0)")
arr(5.0, 4.4, 5.0, 3.7, "α, β, ρ", dx=0.12, dy=-0.05, ha="left")
arr(6.4, 2.9, 7.2, 3.3, "σ*, τ, ants", dx=0.0, dy=-0.35)
arr(8.5, 4.0, 8.5, 4.6)
arr(7.2, 5.1, 6.4, 5.1, "s(t+1)")
arr(8.5, 2.9, 8.5, 1.4, "t = H", dx=0.12, dy=0.0, ha="left")
fig.tight_layout(); fig.savefig("figures/fig0_control_loop.png", dpi=220)
