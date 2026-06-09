# Diffusion Policy — ManiSkill

Visual diffusion policy for robotic manipulation in [ManiSkill](https://maniskill.readthedocs.io). The project trains a single-arm pick-and-place policy on `PickCube-SideView-v1` (RGBD, `pd_ee_delta_pos` control), then evaluates the same checkpoint in a dual-arm scene — optionally applying spherical classifier guidance to constrain end-effector trajectories.

## Setup

```bash
conda create -n dp_maniskill python=3.10 -y
conda activate dp_maniskill
pip install -r requirements.txt
git submodule update --init --recursive
```

All commands require the conda env and a specific `PYTHONPATH`. Use this wrapper for every script:

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python <script> <args>"
```

`PYTHONPATH` needs **both** `.` (for `src.*` imports) and `src` (for `diffusion_policy.*` imports inside the package).

### Smoke test

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/training/train_rgbd.py \
    --env_id PickCube-SideView-v1 \
    --demo_path ~/.maniskill/demos/PickCube-SideView-v1/motionplanning/trajectory.rgbd.pd_joint_pos.h5 \
    --num_demos 2 --max_episode_steps 50 --total_iters 100 \
    --eval_freq 50 --log_freq 10 --batch_size 8 --num_eval_envs 1"
```

---

## Folder Structure

```
diffusion_policy_maniskill/
│
├── src/                        # All source code (see §Source Code below)
│   ├── agent.py                # Agent nn.Module — core shared between training and inference
│   ├── args.py                 # Args dataclass (tyro CLI) — shared config
│   ├── training/               # Training pipeline
│   ├── inference/              # Inference scripts and guidance utilities
│   ├── envs/                   # Custom ManiSkill environment registrations
│   ├── diffusion_policy/       # Architecture modules (UNet, CNN encoder, evaluation, etc.)
│   ├── traj_utils/             # Trajectory manipulation utilities
│   ├── tools/                  # Visualisation helpers
│   └── tests/                  # Unit / diagnostic tests
│
├── demos/                      # Demonstration data (H5 + JSON)
│   └── PickCube-SideView-v1/motionplanning/
│       ├── trajectory.h5                          # Raw motionplanning demos (606 MB)
│       ├── trajectory.rgbd.pd_joint_delta_pos.h5  # RGBD-converted, delta-pos control (6.5 GB)
│       └── *.json                                 # Metadata and state files
│
├── checkpoints/                # Best-eval saved weights
│   ├── best_eval_success_at_end.pt         # Single robot
│   ├── left/best_eval_success_at_end.pt    # Left arm (dual-robot training)
│   └── right/best_eval_success_at_end.pt   # Right arm
│
├── runs/                       # Per-run training outputs (checkpoints + eval videos)
├── plots/                      # Diagnostic plots (guidance cost, etc.)
├── eval_videos/                # Inference videos
├── videos/                     # Inference videos (dual-robot, guided)
├── debug_videos/               # Debug visualisations
├── log/                        # Text logs
├── wandb/                      # W&B experiment tracking data
│
├── external/ManiSkill/         # Git submodule — upstream ManiSkill sim
├── scripts/                    # Intentionally empty (placeholder)
├── configs/                    # Intentionally empty (placeholder)
│
├── .vscode/launch.json         # Pre-configured VS Code debug launchers (6 configs)
├── environment.yml             # Frozen conda env snapshot
├── requirements.txt            # Minimal pip dependencies
└── src/setup.py                # Package install for diffusion_policy subpackage
```

---

## Source Code

### Core (shared between training and inference)

| File | Purpose |
|------|---------|
| `src/agent.py` | `Agent(nn.Module)` — visual encoder (`PlainConv`) + 1-D UNet noise predictor (`ConditionalUnet1D`), DDPM scheduler, `compute_loss()`, `get_action()` with optional classifier guidance |
| `src/args.py` | `Args` dataclass — all training hyperparameters parsed via `tyro`; also used to initialise Agent in inference |

### Training (`src/training/`)

| File | Purpose | Runnable? |
|------|---------|-----------|
| `train_rgbd.py` | Entry point — `tyro.cli(Args)` → `Trainer(args).train()` | ✅ yes |
| `trainer.py` | `Trainer` class — full pipeline: env setup, dataset, agent, optimiser, EMA, training loop, periodic eval, W&B logging, checkpoint saving | No |
| `datasets.py` | `SmallDemoDataset_DiffusionPolicy` (in-memory, fast) and `LazyDemoDataset` (on-demand, memory-efficient) | No |
| `training_utils.py` | `reorder_keys()` — recursively reorder observation dicts to match a reference structure | No |

### Inference (`src/inference/`)

| File | Purpose | Runnable? |
|------|---------|-----------|
| `run_policy.py` | Standalone inference script — single-arm or dual-arm rollout, optional classifier guidance, optional video recording | ✅ yes |
| `guidance.py` | `integrate_ee_deltas()` + `obstacle_cost()` — sphere obstacle cost and gradient for classifier guidance during DDPM denoising | No (module) |
| `play_in_env.py` | Quick interactive env test loop (random actions) | ❌ broken — `gym.make()` is missing env id (all IDs commented out); uncomment one to use |

### Environments (`src/envs/`)

| File | Registered env IDs | Status |
|------|--------------------|--------|
| `pick_cube_side.py` | `PickCube-SideView-v1`, `PickCube-SideView-TwoRobot-v1`, `PickCube-SideView-Left-v1`, `PickCube-SideView-Right-v1`, `PickCube-SideView-DualTask-v1` | ✅ |
| `wrappers.py` | — | ✅ `SingleRobotInferenceWrapper` — maps dict action space to single-robot for inference |
| `collect_demos.py` | — | ❌ broken — ManiSkill API change: `mani_skill.examples.motionplanning.panda.solutions.solvers` no longer exists |

All envs that use `@register_env` are auto-registered when the module is imported. `trainer.py` and `run_policy.py` import `pick_cube_side.py` explicitly for this reason.

### Architecture (`src/diffusion_policy/`)

These are library modules — not runnable directly.

| File | Purpose |
|------|---------|
| `conditional_unet1d.py` | 1-D U-Net with FiLM conditioning for noise prediction; `SinusoidalPosEmb`, `Conv1dBlock`, `ConditionalResidualBlock1D`, `ConditionalUnet1D` |
| `plain_conv.py` | `PlainConv` — 5-layer 2-D CNN visual encoder for 128×128 RGB/depth images → 256-dim feature |
| `utils.py` | `load_demo_dataset()`, `convert_obs()`, `build_obs_space()`, `IterationBasedBatchSampler` |
| `make_env.py` | `make_eval_envs()` — vectorised eval env factory (CPU/GPU backend, frame-stacking, video recording) |
| `evaluate.py` | `evaluate()` — evaluation loop that runs agent for N episodes and returns metrics |
| `wandb_logger.py` | `WandbLogger` — train/val/eval metric logging, video/image/checkpoint upload |

### Trajectory Utilities (`src/traj_utils/`)

Scripts for manipulating ManiSkill H5 demo files. Edit the path constants at the top of each file before running.

| File | Purpose | Status |
|------|---------|--------|
| `regenerate_trajs.py` | Replay existing demos through a new environment (e.g. add side-view camera), save as new H5 | ✅ |
| `transform_and_replay.py` | Replay demos with robot pose transformation (used to generate left/right arm variants) | ✅ |
| `merge_demos.py` | Merge multiple H5 demo files into one; generate combined JSON metadata | ✅ |
| `traj_to_video.py` | Extract `base_camera` RGB frames from H5 and write an MP4 | ✅ |
| `replay_single_trajectory.py` | Load and replay one episode from an H5 file in the environment with rendering | ✅ |
| `replay_trajectory_custom.py` | Replay with custom env/camera options | ✅ |
| `save_demo_states.py` | Save per-step environment states to a JSON for deterministic replay | ✅ |
| `fix_env_id.py` | Patch the `env_id` metadata field inside an H5 file | ✅ |
| `generate_new_trajs.py` | Generate new trajectories via motion planning | ❌ broken — `mani_skill.examples.motion_planning` module path changed |

### Tools & Tests

| File | Purpose | Status |
|------|---------|--------|
| `src/tools/visualize_dataset_obs.py` | Load a demo dataset and render a grid of observations as a PNG | ✅ |
| `src/tests/test_guidance_cost.py` | Diagnostic test for sphere guidance cost — sweeps distances, spot-checks values, saves `plots/guidance_cost_diagnostic.png` | ✅ |

---

## Key Workflows

### 1. Download and convert demos

```bash
# Download ManiSkill demos (adjust env name as needed)
conda run -n dp_maniskill bash -c "python -m mani_skill.utils.download_demo 'PickCube-v1'"

# Convert to RGBD + ee_delta_pos control
conda run -n dp_maniskill bash -c "python -m mani_skill.trajectory.replay_trajectory \
    --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
    --save-traj -o rgbd -c pd_ee_delta_pos"
```

### 2. Train a policy

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/training/train_rgbd.py \
    --env_id PickCube-SideView-Left-v1 \
    --demo_path demos/PickCube-SideView-v1/motionplanning/trajectory.rgbd.pd_joint_delta_pos.h5 \
    --control_mode pd_ee_delta_pos \
    --obs_mode rgb \
    --num_demos 100 \
    --max_episode_steps 100 \
    --total_iters 60000 \
    --eval_freq 5000 \
    --num_eval_envs 5 \
    --no-capture-video"
```

**tyro boolean flags:** use `--no-capture-video` / `--no-cuda` (not `--flag false`).

W&B tracking: add `--track --wandb_project_name <project>`.

### 3. Single-arm inference

```bash
# Basic (renders live window)
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
    --checkpoint_path checkpoints/best_eval_success_at_end.pt \
    --env_id PickCube-SideView-Left-v1 \
    --num_episodes 5 --max_episode_steps 100"

# Save video, no render window
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
    --checkpoint_path checkpoints/best_eval_success_at_end.pt \
    --env_id PickCube-SideView-Left-v1 \
    --num_episodes 5 --record-video --video_dir videos/inference --no-render"

# With fixed sphere guidance
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
    --checkpoint_path checkpoints/best_eval_success_at_end.pt \
    --use_guidance \
    --guidance_center '[0.0, -0.3, 0.4]' \
    --guidance_radius 0.1 \
    --guidance_scale 1.0"

# Guidance sphere tracks the goal site each episode
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
    --checkpoint_path checkpoints/best_eval_success_at_end.pt \
    --use_guidance --track_goal --guidance_radius 0.1"
```

### 4. Dual-arm inference

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/inference/run_policy.py \
    --dual \
    --checkpoint_path checkpoints/left/best_eval_success_at_end.pt \
    --right_checkpoint checkpoints/right/best_eval_success_at_end.pt \
    --num_episodes 5 --max_episode_steps 100"
```

Uses `PickCube-SideView-DualTask-v1`: both arms have independent cubes and goals, each controlled by its own checkpoint simultaneously.

### 5. Regenerate demos with a new camera / environment

Edit `H5_PATH` and `OUTPUT_PATH` at the top of the script, then:

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/traj_utils/regenerate_trajs.py"
```

### 6. Visualise training dataset observations

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/tools/visualize_dataset_obs.py \
    --demo_path demos/PickCube-SideView-v1/motionplanning/trajectory.rgbd.pd_joint_delta_pos.h5 \
    --env_id PickCube-SideView-v1 \
    --control_mode pd_ee_delta_pos \
    --output_path obs_viz.png"
```

### 7. Run guidance cost diagnostic

```bash
conda run -n dp_maniskill bash -c "PYTHONPATH=.:src python src/tests/test_guidance_cost.py"
# Saves plots/guidance_cost_diagnostic.png
```

---

## Broken Files (for cleanup)

| File | Error | Cause |
|------|-------|-------|
| `src/inference/play_in_env.py` | `TypeError: make() missing 1 required positional argument: 'id'` | `gym.make()` called with all env IDs commented out at module level |
| `src/envs/collect_demos.py` | `ImportError: cannot import name 'solvers'` | ManiSkill API change — `mani_skill.examples.motionplanning.panda.solutions.solvers` no longer exists |
| `src/traj_utils/generate_new_trajs.py` | `ModuleNotFoundError: No module named 'mani_skill.examples.motion_planning'` | ManiSkill API change — module path renamed |

---

## Gotchas

**Action spaces**

- `pd_ee_delta_pos`: shape `(4,)` — `(dx, dy, dz, gripper_width)`. No rotation (EE held fixed).
- `pd_joint_pos` / `pd_joint_delta_pos`: shape `(9,)` — 7 arm joints + 2 finger joints.
- Training demos must use the same `control_mode` as inference. Mismatch → garbage actions.

**Observation shapes**

```
Single env:      obs ~ (H, W, C)          action ~ (act_dim,)
Vector env:      obs ~ (N, H, W, C)       action ~ (N, act_dim)   ← gym.make(..., num_envs=1)
FrameStack env:  obs ~ (N, T, H, W, C)    — what Agent expects
```

`Agent.__init__` reads `obs_horizon` from the stacked observation space shape (`shape[1]`). Always pass a stacked obs space (or use `_FakeEnv` in inference).

**Multi-robot observation/action layout**

`PickCube-SideView-DualTask-v1` has a **dict** action space: `{'panda-0': tensor, 'panda-1': tensor}`. Passing a flat tensor will crash `env.step()`. `SingleRobotInferenceWrapper` in `src/envs/wrappers.py` handles the reverse mapping (single tensor → dict) for single-arm evaluation in a two-robot scene.

**TableSceneBuilder pose override**

For `("panda", "panda")` multi-robot envs, ManiSkill's `TableSceneBuilder.initialize()` overrides both robot poses at every `env.reset()`. `PickCubeSideViewTwoRobotEnv._initialize_episode()` explicitly restores the correct training-compatible poses after this. Without that fix, the side-view camera sees the wrong scene.

**State vector layout** (`pd_ee_delta_pos`, single robot)

```
indices   0–8    qpos          (9,)
          9–17   qvel          (9,)
          18     is_grasped    (1,)
          19–25  tcp_pose      (7,) = xyz + quaternion
          26–28  goal_pos      (3,)
```

`guidance.py` uses `_TCP_XYZ_SLICE = slice(19, 22)` to extract TCP xyz for guidance.

**PYTHONPATH**

Both `.` and `src` must be on `PYTHONPATH`. Without `.`, `from src.agent import Agent` fails. Without `src`, `from diffusion_policy.unet import ...` fails.
