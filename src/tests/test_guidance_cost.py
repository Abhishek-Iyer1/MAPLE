"""
Unit test + diagnostic plot for obstacle_cost and its gradient.

Tests cost and grad over a sweep of distances from the sphere centre,
including the boundary (dist == r) and interior (dist < r) regions.

Usage:
    conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/tests/test_guidance_cost.py"

Output:
    plots/guidance_cost_diagnostic.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.guidance import obstacle_cost

# ── Config ────────────────────────────────────────────────────────────────────
RADIUS = 0.2
CENTER = torch.tensor([0.0, 0.0, 0.0])
N_POINTS = 500          # number of distances to sweep
DIST_RANGE = (0.001, 2 * RADIUS)   # start slightly above 0 to avoid division-by-zero

# ── Sweep distances along +x axis ─────────────────────────────────────────────
distances = torch.linspace(DIST_RANGE[0], DIST_RANGE[1], N_POINTS)

costs      = []
grad_mags  = []
grad_signs = []   # sign of x-component (should always be negative = pointing away from centre for repulsion)

for d in distances:
    # position along +x at distance d from centre
    pos = torch.tensor([[d.item(), 0.0, 0.0]], requires_grad=True)   # (1, 3)

    cost = obstacle_cost(pos, CENTER, RADIUS)
    costs.append(cost.item())

    if cost.item() > 0:
        grad = torch.autograd.grad(cost, pos)[0]   # (1, 3)
        gx = grad[0, 0].item()
        grad_mags.append(abs(gx))
        grad_signs.append(np.sign(gx))
    else:
        grad_mags.append(0.0)
        grad_signs.append(0.0)

costs     = np.array(costs)
grad_mags = np.array(grad_mags)
distances_np = distances.numpy()

# ── Print a few spot-check values ──────────────────────────────────────────────
print("=== Spot-check values ===")
check_dists = [0.001, RADIUS * 0.25, RADIUS * 0.5, RADIUS * 0.99, RADIUS, RADIUS * 1.01, RADIUS * 1.5]
for d_val in check_dists:
    pos = torch.tensor([[d_val, 0.0, 0.0]], requires_grad=True)
    cost = obstacle_cost(pos, CENTER, RADIUS)
    if cost.item() > 0:
        grad = torch.autograd.grad(cost, pos)[0]
        gx = grad[0, 0].item()
        print(f"  dist={d_val:.4f}  cost={cost.item():.6f}  grad_x={gx:.6f}  grad_mag={abs(gx):.6f}")
    else:
        print(f"  dist={d_val:.4f}  cost={cost.item():.6f}  grad=0 (outside sphere)")

# ── Check at exact centre (dist = 0) ──────────────────────────────────────────
print("\n=== At exact centre (dist = 0) ===")
pos_zero = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
cost_zero = obstacle_cost(pos_zero, CENTER, RADIUS)
try:
    grad_zero = torch.autograd.grad(cost_zero, pos_zero)[0]
    print(f"  cost={cost_zero.item():.6f}  grad={grad_zero}")
except Exception as e:
    print(f"  cost={cost_zero.item():.6f}  grad ERROR: {e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
Path("plots").mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax1.plot(distances_np, costs, color="tomato", linewidth=2, label="cost = relu(r - dist)")
ax1.axvline(RADIUS, color="gray", linestyle="--", linewidth=1, label=f"r = {RADIUS}")
ax1.axvspan(0, RADIUS, alpha=0.08, color="red", label="inside sphere")
ax1.set_ylabel("Cost")
ax1.set_title("Obstacle cost and gradient vs. distance from sphere centre")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(distances_np, grad_mags, color="steelblue", linewidth=2, label="|∂cost/∂position_x|")
ax2.axvline(RADIUS, color="gray", linestyle="--", linewidth=1, label=f"r = {RADIUS}")
ax2.axvspan(0, RADIUS, alpha=0.08, color="red", label="inside sphere")
ax2.set_xlabel("Distance from sphere centre (m)")
ax2.set_ylabel("Gradient magnitude")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = Path("plots/guidance_cost_diagnostic.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved to: {out.resolve()}")
