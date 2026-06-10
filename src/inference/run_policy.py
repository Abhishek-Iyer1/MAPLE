"""
Standalone inference script for a trained diffusion policy checkpoint.

Usage:
    (Single robot)
    conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
        --checkpoint_path checkpoints/best_eval_success_at_end.pt \
        --num_episodes 3 --max_episode_steps 100"

    (Dual robot — left + right checkpoints run simultaneously)
    conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
        --dual \
        --checkpoint_path checkpoints/left/best_eval_success_at_end.pt \
        --right_checkpoint checkpoints/right/best_eval_success_at_end.pt \
        --num_episodes 5"
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import random

import gymnasium as gym
import imageio
import numpy as np
import torch
import tyro
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

from src.agent import Agent
from src.args import Args
from src.inference.guidance import obstacle_cost, integrate_ee_deltas, _TCP_XYZ_SLICE

LEFT_TRAIN_POSES = [
    {"cube_p": [-0.0536, -0.2857, 0.02], "cube_q": [-0.1794, 0, 0, 0.9838], "goal_p": [0.002,   -0.2582, 0.2889]},
    {"cube_p": [0.0441,  -0.2335, 0.02], "cube_q": [0.6392,  0, 0, 0.7691],  "goal_p": [0.0206,  -0.225,  0.2463]},
    {"cube_p": [0.0238,  -0.2621, 0.02], "cube_q": [0.9935,  0, 0, -0.1141], "goal_p": [0.0115,  -0.2612, 0.0487]},
    {"cube_p": [0.0789,  -0.3841, 0.02], "cube_q": [-0.6413, 0, 0, 0.7673],  "goal_p": [-0.0544, -0.373,  0.2431]},
    {"cube_p": [-0.0118, -0.2731, 0.02], "cube_q": [0.6909,  0, 0, 0.7229],  "goal_p": [-0.0986, -0.3772, 0.2939]},
    {"cube_p": [0.0748,  -0.2189, 0.02], "cube_q": [0.8606,  0, 0, 0.5093],  "goal_p": [0.0671,  -0.3617, 0.2414]},
    {"cube_p": [-0.0108, -0.2706, 0.02], "cube_q": [0.0482,  0, 0, 0.9988],  "goal_p": [-0.0349, -0.2952, 0.2444]},
    {"cube_p": [0.0602,  -0.278,  0.02], "cube_q": [0.0541,  0, 0, 0.9985],  "goal_p": [0.0586,  -0.3,    0.2089]},
    {"cube_p": [-0.0691, -0.2654, 0.02], "cube_q": [0.7371,  0, 0, -0.6758], "goal_p": [-0.0798, -0.2561, 0.0242]},
    {"cube_p": [0.0396,  -0.2538, 0.02], "cube_q": [0.864,   0, 0, 0.5035],  "goal_p": [0.0479,  -0.1988, 0.306 ]},
]
from src.envs.pick_cube_side import PickCubeSideViewEnv  # noqa — registers all SideView envs
from src.envs.pick_cube_side import PickCubeSideViewDualTaskEnv  # noqa
from src.envs.wrappers import SingleRobotInferenceWrapper
from mani_skill.utils.structs.pose import Pose


@dataclass
class InferenceArgs:
    checkpoint_path: str = "checkpoints/best_eval_success_at_end.pt"
    env_id: str = "PickCube-SideView-Left-v1"
    num_episodes: int = 5
    max_episode_steps: int = 100
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    control_mode: str = "pd_ee_delta_pos"
    obs_mode: str = "rgb"
    device: str = "cuda"
    render: bool = True
    use_train_poses: bool = False
    # Reproducibility
    seed: int = 42
    # Video recording — wraps env with RecordEpisode; disables live render window
    record_video: bool = False
    video_dir: str = "videos/inference"
    # Dual-policy mode: set --dual and provide --right_checkpoint
    dual: bool = False
    right_checkpoint: str = "checkpoints/right/best_eval_success_at_end.pt"
    # Architecture — must match checkpoint
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    # Guidance — sphere obstacle cost applied at every denoising step
    use_guidance: bool = False
    guidance_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.5])
    guidance_radius: float = 0.1
    guidance_scale: float = 1.0
    track_goal: bool = False  # if True, place sphere at the goal site each episode


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


def _set_seed(seed: int):
    """Seed all RNGs for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dual-policy helpers ───────────────────────────────────────────────────────

# State dim matching single-robot training: qpos(9)+qvel(9)+is_grasped(1)+tcp_pose(7)+goal_pos(3)
_DUAL_STATE_DIM = 29
_DUAL_IMG_H, _DUAL_IMG_W = 128, 128


class _DualFakeEnv:
    """Obs/action space proxy matching each single-robot policy's training format."""

    def __init__(self, obs_horizon: int, action_space: spaces.Box):
        self.single_action_space = action_space
        self.single_observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (obs_horizon, _DUAL_IMG_H, _DUAL_IMG_W, 3), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (obs_horizon, _DUAL_STATE_DIM), dtype=np.float32),
        })


def _extract_robot_obs(obs: dict, robot_idx: int) -> dict:
    """Build per-robot obs dict matching single-robot training format (no batch dim)."""
    robot_key = f"panda-{robot_idx}"
    agent = obs["agent"][robot_key]
    extra = obs["extra"]
    qpos     = agent["qpos"][0].cpu().numpy().astype(np.float32)                          # (9,)
    qvel     = agent["qvel"][0].cpu().numpy().astype(np.float32)                          # (9,)
    is_grasped = np.array([float(extra[f"is_grasped_{robot_idx}"][0].cpu())], dtype=np.float32)  # (1,)
    tcp_pose = extra[f"tcp_pose_{robot_idx}"][0].cpu().numpy().astype(np.float32)         # (7,)
    goal_pos = extra[f"goal_pos_{robot_idx}"][0].cpu().numpy().astype(np.float32)         # (3,)
    state = np.concatenate([qpos, qvel, is_grasped, tcp_pose, goal_pos])                  # (29,)
    rgb   = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()                     # (H,W,3)
    return {"rgb": rgb, "state": state}


def _stack_buf(buf: deque, device: torch.device) -> dict:
    rgb   = torch.stack([torch.from_numpy(o["rgb"])   for o in buf], dim=0).unsqueeze(0).to(device)
    state = torch.stack([torch.from_numpy(o["state"]) for o in buf], dim=0).unsqueeze(0).float().to(device)
    return {"rgb": rgb, "state": state}  # (1,obs_h,H,W,3), (1,obs_h,29)


def _load_dual_agent(ckpt_path: str, fake_env: _DualFakeEnv, args: "InferenceArgs", device: torch.device) -> "Agent":
    policy_args = Args()
    policy_args.obs_horizon              = args.obs_horizon
    policy_args.act_horizon              = args.act_horizon
    policy_args.pred_horizon             = args.pred_horizon
    policy_args.control_mode             = args.control_mode
    policy_args.diffusion_step_embed_dim = args.diffusion_step_embed_dim
    policy_args.unet_dims                = args.unet_dims
    policy_args.n_groups                 = args.n_groups
    agent = Agent(fake_env, policy_args).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["ema_agent"])
    agent.eval()
    print(f"Loaded: {ckpt_path}")
    return agent


def dual_main(args: "InferenceArgs"):
    _set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    render_mode = "rgb_array" if args.record_video else ("human" if args.render else "rgb_array")
    env = gym.make(
        "PickCube-SideView-DualTask-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode=args.control_mode,
        render_mode=render_mode,
        max_episode_steps=args.max_episode_steps,
        cost_sphere_radius=args.guidance_radius,
    )
    if args.record_video:
        env = RecordEpisode(env, output_dir=args.video_dir, save_trajectory=False, save_video=True, save_on_reset=True, video_fps=20)

    r0_space = env.single_action_space["panda-0"]
    r1_space = env.single_action_space["panda-1"]

    agent_left  = _load_dual_agent(args.checkpoint_path,  _DualFakeEnv(args.obs_horizon, r0_space), args, device)
    agent_right = _load_dual_agent(args.right_checkpoint, _DualFakeEnv(args.obs_horizon, r1_space), args, device)

    # Build per-robot guidance closures if requested.
    # Each closure:
    #   1. Reads the latest TCP xyz from the robot's obs buffer (updated before get_action).
    #   2. Calls integrate_ee_deltas to convert normalised action deltas → absolute xyz.
    #   3. Evaluates obstacle_cost against the shared sphere centre/radius.
    # tcp_left/tcp_right are mutable single-element lists so the closures always see
    # the value written just before get_action() is called each chunk.
    guidance_fn_left = guidance_fn_right = None
    tcp_left_cell  = [None]   # torch.Tensor (3,), updated each chunk
    tcp_right_cell = [None]

    if args.use_guidance:
        center  = torch.tensor(args.guidance_center, dtype=torch.float32, device=device)
        _radius = args.guidance_radius

        def guidance_fn_left(norm_actions):
            positions = integrate_ee_deltas(norm_actions, tcp_left_cell[0], agent_left)
            return obstacle_cost(positions, center, _radius)

        def guidance_fn_right(norm_actions):
            positions = integrate_ee_deltas(norm_actions, tcp_right_cell[0], agent_right)
            return obstacle_cost(positions, center, _radius)

        print(f"Guidance enabled: center={args.guidance_center}, radius={_radius}, scale={args.guidance_scale}")

    # Unwrapped env for direct actor access (cost sphere repositioning)
    raw_env = env.unwrapped

    def _sphere_pose(xyz):
        """xyz: list[float] or 1-D torch.Tensor"""
        if isinstance(xyz, torch.Tensor):
            p = xyz.detach().float().to(device).unsqueeze(0)
        else:
            p = torch.tensor([xyz], dtype=torch.float32, device=device)
        return Pose.create_from_pq(p=p, q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))

    successes = []

    for ep in range(args.num_episodes):
        ep_seed = args.seed + ep
        _set_seed(ep_seed)
        obs, _ = env.reset(seed=[ep_seed])
        buf_left  = deque([_extract_robot_obs(obs, 0)] * args.obs_horizon, maxlen=args.obs_horizon)
        buf_right = deque([_extract_robot_obs(obs, 1)] * args.obs_horizon, maxlen=args.obs_horizon)

        step, done, info = 0, False, {}
        while step < args.max_episode_steps and not done:
            # Refresh TCP positions from latest obs before running the denoising loop.
            # state layout: qpos(9)|qvel(9)|is_grasped(1)|tcp_pose(7)|goal_pos(3)
            # _TCP_XYZ_SLICE = slice(19, 22) extracts the xyz part of tcp_pose.
            if args.use_guidance:
                tcp_left_cell[0]  = torch.from_numpy(
                    buf_left[-1]["state"][_TCP_XYZ_SLICE]
                ).float().to(device)
                tcp_right_cell[0] = torch.from_numpy(
                    buf_right[-1]["state"][_TCP_XYZ_SLICE]
                ).float().to(device)

            act_left  = agent_left.get_action(
                _stack_buf(buf_left,  device),
                guidance_fn=guidance_fn_left,
                guidance_scale=args.guidance_scale,
            )
            act_right = agent_right.get_action(
                _stack_buf(buf_right, device),
                guidance_fn=guidance_fn_right,
                guidance_scale=args.guidance_scale,
            )

            for i in range(args.act_horizon):
                obs, _rew, terminated, truncated, info = env.step({
                    "panda-0": act_left[:, i].cpu().numpy(),
                    "panda-1": act_right[:, i].cpu().numpy(),
                })
                buf_left.append(_extract_robot_obs(obs, 0))
                buf_right.append(_extract_robot_obs(obs, 1))
                step += 1
                if terminated.any() or truncated.any() or step >= args.max_episode_steps:
                    done = True
                    break

            # cost_sphere is in _hidden_objects so it never appears in base_camera obs;
            # just keep its world position up to date for the render camera.
            if args.use_guidance:
                raw_env.cost_sphere.set_pose(_sphere_pose(args.guidance_center))
            if not args.record_video:
                env.render()

        success      = bool(info.get("success",        np.array([False]))[0])
        left_placed  = bool(info.get("is_obj_placed_0", np.array([False]))[0])
        right_placed = bool(info.get("is_obj_placed_1", np.array([False]))[0])
        successes.append(success)
        print(
            f"Episode {ep+1}/{args.num_episodes}: steps={step}, "
            f"left_placed={left_placed}, right_placed={right_placed}, success={success}"
        )

    if args.record_video:
        env.flush_video()
    env.close()
    rate = sum(successes) / len(successes) if successes else 0.0
    print(f"\nSuccess rate: {sum(successes)}/{len(successes)} = {rate:.1%}")
    if args.record_video:
        print(f"Videos saved to: {Path(args.video_dir).resolve()}")


# ── Single-policy main ────────────────────────────────────────────────────────

def main(args: "InferenceArgs"):
    _set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    render_mode = "rgb_array" if args.record_video else ("human" if args.render else "rgb_array")
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=render_mode,
        max_episode_steps=args.max_episode_steps,
        cost_sphere_radius=args.guidance_radius,
    )
    if isinstance(env.single_action_space, gym.spaces.Dict):
        env = SingleRobotInferenceWrapper(env)
    env = FlattenRGBDObservationWrapper(env)
    if args.record_video:
        env = RecordEpisode(env, output_dir=args.video_dir, save_trajectory=False, save_video=True, save_on_reset=True, video_fps=20)

    fake_env = _FakeEnv(env, args.obs_horizon)
    agent = Agent(fake_env, args).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["ema_agent"])
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    # Build guidance closure for single-robot mode.
    # tcp_cell / center_cell are mutable single-element lists so the closure
    # always sees the values written just before get_action() is called.
    # State layout: qpos(9)|qvel(9)|is_grasped(1)|tcp_pose_xyzquat(7)|goal_pos(3)
    # → tcp xyz lives at obs["state"][0, 19:22]  (same _TCP_XYZ_SLICE as dual mode)
    tcp_cell    = [None]
    center_cell = [torch.tensor(args.guidance_center, dtype=torch.float32, device=device)]
    guidance_fn = None
    if args.use_guidance:
        _radius = args.guidance_radius

        def guidance_fn(norm_actions):
            positions = integrate_ee_deltas(norm_actions, tcp_cell[0], agent)
            return obstacle_cost(positions, center_cell[0], _radius)

        mode = "goal site" if args.track_goal else f"fixed {args.guidance_center}"
        print(f"Guidance enabled: center={mode}, radius={_radius}, scale={args.guidance_scale}")

    raw_env = env.unwrapped

    def _sphere_pose(xyz):
        """xyz: list[float] or 1-D torch.Tensor"""
        if isinstance(xyz, torch.Tensor):
            p = xyz.detach().float().to(device).unsqueeze(0)
        else:
            p = torch.tensor([xyz], dtype=torch.float32, device=device)
        return Pose.create_from_pq(p=p, q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))

    successes = []

    for ep in range(args.num_episodes):
        ep_seed = args.seed + ep
        _set_seed(ep_seed)
        pose_options = LEFT_TRAIN_POSES[ep % len(LEFT_TRAIN_POSES)] if args.use_train_poses else {}
        obs, _ = env.reset(seed=[ep_seed], options=pose_options)
        # Update guidance center and sphere position after reset.
        if args.use_guidance:
            if args.track_goal:
                # Place sphere at the goal site position (randomised per episode).
                goal_p = raw_env.goal_site.pose.p
                if isinstance(goal_p, torch.Tensor):
                    center_cell[0] = goal_p[0].float().to(device)
                else:
                    center_cell[0] = torch.tensor(goal_p[0], dtype=torch.float32, device=device)
            raw_env.cost_sphere.set_pose(_sphere_pose(center_cell[0]))
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

            # Refresh TCP xyz from latest obs before running the denoising loop
            if args.use_guidance:
                raw_state = obs_buf[-1]["state"]   # (1, state_dim) tensor or numpy
                tcp_cell[0] = to_t(raw_state)[0, _TCP_XYZ_SLICE].float()

            stacked = {
                "rgb": torch.stack([to_t(o["rgb"]) for o in obs_buf], dim=1),
                # (1, obs_horizon, H, W, C) uint8
                "state": torch.stack([to_t(o["state"]) for o in obs_buf], dim=1).float(),
                # (1, obs_horizon, state_dim)
            }

            # Diffusion inference → (1, act_horizon, act_dim)
            action_seq = agent.get_action(
                stacked,
                guidance_fn=guidance_fn,
                guidance_scale=args.guidance_scale,
            )
            action_np = action_seq.cpu().numpy()  # (1, act_horizon, act_dim)

            # Execute action chunk step-by-step
            for i in range(args.act_horizon):
                obs, _rew, terminated, truncated, info = env.step(action_np[:, i])
                obs_buf.append(obs)
                step += 1
                if terminated.any() or truncated.any() or step >= args.max_episode_steps:
                    done = True
                    break

            if not args.record_video:
                env.render()

        success = bool(info.get("success", np.array([False]))[0])
        successes.append(success)
        print(f"Episode {ep + 1}/{args.num_episodes}: steps={step}, success={success}")

    if args.record_video:
        env.flush_video()  # flush the final episode (save_on_reset only fires on reset, not close)
    env.close()

    rate = sum(successes) / len(successes) if successes else 0.0
    print(f"\nSuccess rate: {sum(successes)}/{len(successes)} = {rate:.1%}")
    if args.record_video:
        print(f"Videos saved to: {Path(args.video_dir).resolve()}")


if __name__ == "__main__":
    args = tyro.cli(InferenceArgs)
    if args.dual:
        dual_main(args)
    else:
        main(args)
