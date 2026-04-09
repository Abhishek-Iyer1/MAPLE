"""
Visualize base camera observations sampled from the training dataloader.

Usage:
    conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/tools/visualize_dataset_obs.py \
        --demo_path ~/.maniskill/demos/PickCube-SideView-Left-v1/motionplanning/trajectory_left_rgb_ee.h5 \
        --env_id PickCube-SideView-Left-v1 \
        --control_mode pd_ee_delta_pos \
        --num_demos 10 \
        --output_path obs_visualization.png"
"""

import argparse
import os
from functools import partial

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import mani_skill.envs
from diffusion_policy.utils import build_state_obs_extractor, convert_obs, load_demo_dataset
from src.envs.pick_cube_side import PickCubeSideViewEnv  # noqa — registers all SideView envs
from src.args import Args
from src.training.datasets import SmallDemoDataset_DiffusionPolicy
from src.training.training_utils import reorder_keys


def make_dataset(demo_path, env_id, control_mode, num_demos, obs_mode="rgb"):
    env_kwargs = dict(
        control_mode=control_mode,
        reward_mode="sparse",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=100,
    )
    tmp_env = gym.make(env_id, **env_kwargs)
    obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(env_id),
        depth=include_depth,
    )

    args = Args()
    args.obs_horizon = 2
    args.pred_horizon = 16
    args.act_horizon = 8

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=torch.device("cpu"),
        num_traj=num_demos,
        control_mode=control_mode,
        args=args,
    )
    return dataset


def slice_to_traj_position(dataset, idx):
    """Return how far into the trajectory this slice is (0.0 = start, 1.0 = end)."""
    traj_idx, start, end = dataset.slices[idx]
    L = dataset.trajectories["actions"][traj_idx].shape[0]
    effective_start = max(0, start)
    return effective_start / L, L, effective_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", required=True)
    parser.add_argument("--env_id", default="PickCube-SideView-Left-v1")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--output_path", default="obs_visualization.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset from {args.demo_path}...")
    dataset = make_dataset(args.demo_path, args.env_id, args.control_mode, args.num_demos)
    print(f"Dataset size: {len(dataset)} slices")

    # ── Sample strategies ──────────────────────────────────────────────────────
    # 1. Random (what you see with a normal dataloader)
    # 2. Forced early (start ≤ 5% into trajectory) — should show block on table
    # 3. Forced late  (start ≥ 80% into trajectory) — should show block in hand

    def indices_by_position(dataset, min_frac, max_frac, n=6):
        candidates = [
            i for i in range(len(dataset))
            if min_frac <= slice_to_traj_position(dataset, i)[0] <= max_frac
        ]
        chosen = np.random.choice(candidates, min(n, len(candidates)), replace=False)
        return chosen.tolist()

    random_indices  = np.random.choice(len(dataset), 6, replace=False).tolist()
    early_indices   = indices_by_position(dataset, 0.0, 0.10, n=6)
    late_indices    = indices_by_position(dataset, 0.80, 1.00, n=6)

    sections = [
        ("Random (dataloader default)",      random_indices),
        ("Early slices  (0–10% of traj)",    early_indices),
        ("Late slices   (80–100% of traj)",  late_indices),
    ]

    n_rows = len(sections)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3))
    fig.suptitle("Base camera obs — latest frame in obs_horizon window", fontsize=13)

    for row, (title, idxs) in enumerate(sections):
        for col, idx in enumerate(idxs):
            ax = axes[row][col]
            sample = dataset[idx]
            # rgb stored as (obs_horizon, C, H, W) after transpose_fn
            # Take the LAST frame in the obs window, convert back to (H, W, C)
            rgb = sample["observations"]["rgb"][-1]  # (C, H, W)
            img = rgb.permute(1, 2, 0).numpy().astype(np.uint8)

            frac, L, eff_start = slice_to_traj_position(dataset, idx)
            traj_idx, start, _ = dataset.slices[idx]

            ax.imshow(img)
            ax.set_title(f"traj={traj_idx} t={eff_start}/{L}\n({frac*100:.0f}%)", fontsize=7)
            ax.axis("off")
        axes[row][0].set_ylabel(title, fontsize=8, labelpad=4)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {args.output_path}")

    # ── Print distribution stats ───────────────────────────────────────────────
    print("\n── Slice position distribution ──")
    fracs = [slice_to_traj_position(dataset, i)[0] for i in range(len(dataset))]
    fracs = np.array(fracs)
    bins = [0, 0.1, 0.25, 0.5, 0.75, 1.01]
    labels = ["0–10%", "10–25%", "25–50%", "50–75%", "75–100%"]
    for lo, hi, label in zip(bins, bins[1:], labels):
        count = np.sum((fracs >= lo) & (fracs < hi))
        pct = count / len(fracs) * 100
        print(f"  {label}: {count:5d} slices ({pct:.1f}%)")


if __name__ == "__main__":
    main()
