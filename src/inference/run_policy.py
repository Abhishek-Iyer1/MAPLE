"""
Standalone inference script for a trained diffusion policy checkpoint.

Usage:
    (Single robot example)
    conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
        --checkpoint_path checkpoints/best_eval_success_at_end.pt \
        --num_episodes 3 --max_episode_steps 100"

    (Two robot example)
    conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
        --checkpoint_path checkpoints/best_eval_success_at_end.pt \
        --env_id PickCube-SideView-TwoRobot-v1 \
        --num_episodes 5 \
        --max_episode_steps 100"
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from src.agent import Agent
from src.envs.pick_cube_side import PickCubeSideViewEnv  # noqa — registers PickCube-SideView-v1
from src.envs.pick_cube_side import PickCubeSideViewTwoRobotEnv  # noqa — registers PickCube-SideView-TwoRobot-v1
from src.envs.wrappers import SingleRobotInferenceWrapper


@dataclass
class InferenceArgs:
    checkpoint_path: str = "checkpoints/best_eval_success_at_end.pt"
    env_id: str = "PickCube-SideView-v1"
    num_episodes: int = 5
    max_episode_steps: int = 100
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    control_mode: str = "pd_ee_delta_pos"
    obs_mode: str = "rgb"
    device: str = "cuda"
    render: bool = True
    # Architecture — must match checkpoint
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8


class _FakeEnv:
    """Proxy that presents a stacked observation space to Agent.__init__."""

    def __init__(self, env, obs_horizon: int):
        base = env.single_observation_space
        self.single_action_space = env.single_action_space
        sd = {
            "rgb": spaces.Box(
                0, 255,
                (obs_horizon, *base["rgb"].shape),
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                -np.inf, np.inf,
                (obs_horizon, *base["state"].shape),
                dtype=np.float32,
            ),
        }
        self.single_observation_space = spaces.Dict(sd)


def main():
    args = tyro.cli(InferenceArgs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="human" if args.render else "rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    if isinstance(env.single_action_space, gym.spaces.Dict):
        env = SingleRobotInferenceWrapper(env)
    env = FlattenRGBDObservationWrapper(env)

    fake_env = _FakeEnv(env, args.obs_horizon)
    agent = Agent(fake_env, args).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["ema_agent"])
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    successes = []

    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        obs_buf = deque([obs] * args.obs_horizon, maxlen=args.obs_horizon)

        step = 0
        done = False
        info = {}

        while step < args.max_episode_steps and not done:
            # Stack buffer → (1, obs_horizon, ...) tensors
            # obs values may be CUDA tensors already (ManiSkill GPU sim)
            def to_t(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                return torch.from_numpy(np.asarray(x)).to(device)

            stacked = {
                "rgb": torch.stack([to_t(o["rgb"]) for o in obs_buf], dim=1),
                # (1, obs_horizon, H, W, C) uint8
                "state": torch.stack([to_t(o["state"]) for o in obs_buf], dim=1).float(),
                # (1, obs_horizon, state_dim)
            }

            # Diffusion inference → (1, act_horizon, act_dim)
            action_seq = agent.get_action(stacked)
            action_np = action_seq.cpu().numpy()  # (1, act_horizon, act_dim)

            # Execute action chunk step-by-step
            for i in range(args.act_horizon):
                obs, _rew, terminated, truncated, info = env.step(action_np[:, i])
                obs_buf.append(obs)
                step += 1
                if terminated.any() or truncated.any() or step >= args.max_episode_steps:
                    done = True
                    break
            env.render()

        success = bool(info.get("success", np.array([False]))[0])
        successes.append(success)
        print(f"Episode {ep + 1}/{args.num_episodes}: steps={step}, success={success}")

    env.close()

    rate = sum(successes) / len(successes) if successes else 0.0
    print(f"\nSuccess rate: {sum(successes)}/{len(successes)} = {rate:.1%}")


if __name__ == "__main__":
    main()
