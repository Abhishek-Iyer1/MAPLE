# src/traj_utils/transform_and_replay.py
"""
Replay existing PickCube-v1 demos through PickCube-SideView-Left-v1 or
PickCube-SideView-Right-v1 by transforming the cube and goal positions to
match the new robot base pose.

WHY THIS IS NEEDED
------------------
The new side-view envs move the robot from its default pose at [-0.615, 0, 0]
facing +X to a new pose on the left or right of the table.  The original
joint-space actions (pd_joint_pos) are invariant to the robot base pose —
the same qpos produces the same TCP position *relative to the robot base*.
So if we transform the cube and goal to the equivalent positions relative to
the new robot base, the identical actions will succeed.

TRANSFORM ALGEBRA (for a world point p in the default env)
----------------------------------------------------------
  p_rel = p - [-0.615, 0, 0]          # relative to default robot

  LEFT  (R_z +90°):  p_new = R_z(+90°) * p_rel + [0, -0.9, 0]
  RIGHT (R_z -90°):  p_new = R_z(-90°) * p_rel + [0,  0.9, 0]

Verification — default cube centre [0, 0, 0.02]:
  LEFT  → [0, -0.285, 0.02]  (in front of left robot at y=-0.9)  ✓
  RIGHT → [0, +0.285, 0.02]  (in front of right robot at y=+0.9) ✓
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom environments
from src.envs.pick_cube_side import (  # noqa: F401
    PickCubeSideViewEnvLeft,
    PickCubeSideViewEnvRight,
)
import mani_skill.envs
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import gym_utils


# ---------------------------------------------------------------------------
# Transformation config
# ---------------------------------------------------------------------------

DEFAULT_ROBOT_POS = np.array([-0.615, 0.0, 0.0])

TRANSFORMS = {
    "PickCube-SideView-Left-v1":  {"angle_deg":  90.0, "base": np.array([0.0, -0.9, 0.0])},
    "PickCube-SideView-Right-v1": {"angle_deg": -90.0, "base": np.array([0.0,  0.9, 0.0])},
}


def transform_point(p: np.ndarray, angle_deg: float, new_base: np.ndarray) -> np.ndarray:
    """Transform a world point from the default env frame to the target env frame."""
    p_rel = p - DEFAULT_ROBOT_POS
    R = Rotation.from_euler('z', angle_deg, degrees=True)
    return R.apply(p_rel) + new_base


def transform_quat_wxyz(q: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a wxyz quaternion around Z by angle_deg."""
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])           # wxyz → xyzw (scipy)
    r_new = Rotation.from_euler('z', angle_deg, degrees=True) * Rotation.from_quat(q_xyzw)
    q_new = r_new.as_quat()                                 # xyzw
    return np.array([q_new[3], q_new[0], q_new[1], q_new[2]])  # → wxyz


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def save_obs_to_hdf5(grp, obs_list):
    obs_grp = grp.create_group('obs')
    first_obs = obs_list[0]

    def save_recursive(parent_grp, key_path, template):
        for key, val in template.items():
            if isinstance(val, dict):
                sub_grp = parent_grp.create_group(key)
                save_recursive(sub_grp, key_path + [key], val)
            else:
                data = []
                for obs in obs_list:
                    v = obs
                    for k in key_path + [key]:
                        v = v[k]
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    data.append(v)
                data = np.stack(data)
                if len(data.shape) > 1 and data.shape[1] == 1:
                    data = data.squeeze(1)
                parent_grp.create_dataset(key, data=data)

    save_recursive(obs_grp, [], first_obs)


def save_target_env_states_to_hdf5(grp, state_list):
    """Save env states collected from the TARGET env (list of get_state_dict() outputs)."""
    env_states_grp = grp.create_group('env_states')
    actors_grp = env_states_grp.create_group('actors')
    articulations_grp = env_states_grp.create_group('articulations')

    for actor_name in state_list[0]['actors'].keys():
        data = np.stack([
            s['actors'][actor_name].squeeze(0).cpu().numpy() for s in state_list
        ])  # (T, 13)
        actors_grp.create_dataset(actor_name, data=data)

    for art_name in state_list[0]['articulations'].keys():
        data = np.stack([
            s['articulations'][art_name].squeeze(0).cpu().numpy() for s in state_list
        ])  # (T, 31)
        articulations_grp.create_dataset(art_name, data=data)


# ---------------------------------------------------------------------------
# EE delta pos conversion helpers
# ---------------------------------------------------------------------------

def setup_ee_conversion(ori_env, env):
    """
    Build Pinocchio model and extract controller handles needed for
    pd_joint_pos → pd_ee_delta_pos conversion.

    Returns a dict of handles used inside the action loop.
    """
    ori_ctrl  = ori_env.unwrapped.agent.controller
    arm_ctrl  = env.unwrapped.agent.controller.controllers["arm"]
    ori_arm_ctrl = ori_ctrl.controllers["arm"]

    assert arm_ctrl.config.frame in [
        "root_translation",
        "root_translation:root_aligned_body_rotation",
    ], f"Unsupported EE control frame: {arm_ctrl.config.frame}"

    pin_model = ori_ctrl.articulation.create_pinocchio_model()

    return dict(
        ori_ctrl=ori_ctrl,
        ori_arm_ctrl=ori_arm_ctrl,
        arm_ctrl=arm_ctrl,
        ee_link=arm_ctrl.ee_link,
        pin_model=pin_model,
        pos_only=(arm_ctrl.config.frame == "root_translation"),
        use_target=bool(arm_ctrl.config.use_target),
    )


def convert_action_to_ee_delta(ori_action: np.ndarray, handles: dict, device) -> torch.Tensor:
    """
    Convert one pd_joint_pos action to a pd_ee_delta_pos action.

    Must be called AFTER ori_env.step(ori_action) so that
    ori_arm_ctrl._target_qpos already reflects the next target.
    """
    ori_ctrl     = handles["ori_ctrl"]
    ori_arm_ctrl = handles["ori_arm_ctrl"]
    arm_ctrl     = handles["arm_ctrl"]
    ee_link      = handles["ee_link"]
    pin_model    = handles["pin_model"]
    pos_only     = handles["pos_only"]
    use_target   = handles["use_target"]

    # FK on ori_env's controller target to get desired EE pose
    full_qpos = ori_arm_ctrl.articulation.get_qpos().clone()
    full_qpos[:, ori_arm_ctrl.active_joint_indices] = ori_arm_ctrl._target_qpos
    pin_model.compute_forward_kinematics(full_qpos.cpu().numpy()[0])
    target_ee_pose = Pose.create(
        ori_arm_ctrl.articulation.pose.sp
        * pin_model.get_link_pose(ee_link.index)
    )

    # Delta from target env's current EE target (or actual pose)
    if use_target:
        delta_pos = (
            target_ee_pose.p
            - arm_ctrl._target_pose.p
            - arm_ctrl.articulation.pose.p
        ).cpu().numpy()[0]
    else:
        delta_pos = (target_ee_pose.p - ee_link.pose.p).cpu().numpy()[0]

    low  = arm_ctrl.action_space_low.cpu().numpy()
    high = arm_ctrl.action_space_high.cpu().numpy()
    arm_action = np.clip(gym_utils.inv_scale_action(delta_pos, low, high), -1, 1)

    # Rebuild full action (arm replaced, gripper kept from original)
    ori_action_t    = torch.tensor(ori_action, device=device, dtype=torch.float32)
    ori_action_dict = ori_ctrl.to_action_dict(ori_action_t)
    output_dict     = {k: v for k, v in ori_action_dict.items()}
    output_dict["arm"] = torch.tensor(arm_action, device=device, dtype=torch.float32)

    return env_controller_from_dict(handles, output_dict, device)


def env_controller_from_dict(handles, action_dict, device):
    """Call from_action_dict on the *target* env's controller."""
    ctrl = handles["arm_ctrl"].articulation._objs[0]  # not used directly
    # We need the combined controller, not just arm_ctrl
    return None  # placeholder — see usage below


# ---------------------------------------------------------------------------
# Main replay function
# ---------------------------------------------------------------------------

def _sync_controller_targets(env):
    """
    Directly update each sub-controller's internal target to match the current
    simulation state (after set_qpos).

    controller.reset() uses scene._reset_mask which is only True during an
    actual env.reset() call — calling it manually afterwards is a no-op.
    This function bypasses that by writing the targets directly.
    """
    for sub_ctrl in env.unwrapped.agent.controller.controllers.values():
        if hasattr(sub_ctrl, '_target_pose'):
            # PDEEPos/PDEEPose controller: seed target from current EE pose
            sub_ctrl._target_pose = sub_ctrl.ee_pose_at_base
        if hasattr(sub_ctrl, '_target_qpos'):
            # PDJointPos controller: seed target from current joint positions
            sub_ctrl._target_qpos = sub_ctrl.qpos.clone()


def _setup_env_state(env, cube_pos_new, cube_q_new, goal_pos_new, goal_q_new, qpos, device):
    """Reset env, override cube/goal/qpos, then sync controller targets."""
    env.reset()

    env.unwrapped.cube.set_pose(
        Pose.create_from_pq(
            p=torch.tensor([[*cube_pos_new]], device=device),
            q=torch.tensor([[*cube_q_new]],  device=device),
        )
    )
    env.unwrapped.goal_site.set_pose(
        Pose.create_from_pq(
            p=torch.tensor([[*goal_pos_new]], device=device),
            q=torch.tensor([[*goal_q_new]],  device=device),
        )
    )
    env.unwrapped.agent.robot.set_qpos(torch.tensor([qpos], device=device, dtype=torch.float32))
    env.unwrapped.agent.robot.set_qvel(torch.zeros(1, len(qpos), device=device))

    # Sync all controller targets to restored qpos (bypasses _reset_mask guard)
    _sync_controller_targets(env)

def clone_obs_dict(d):
    """Recursively clone PyTorch tensors and numpy arrays to break memory references."""
    if isinstance(d, dict):
        return {k: clone_obs_dict(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.clone().detach() # Copies the tensor memory safely
    elif isinstance(d, np.ndarray):
        return d.copy()
    else:
        return d
    
def transform_and_replay(
    source_path: str,
    target_env: str,
    output_dir: str,
    output_name: str = None,
    num_demos: int = None,
    obs_mode: str = "rgbd",
    control_mode: str = "pd_joint_pos",
    target_control_mode: str = None,
    visualize: bool = False,
):
    """
    Args:
        source_path:          PickCube-v1 source H5 file.
        target_env:           One of the sideview env IDs in TRANSFORMS.
        output_dir:           Where to write the output H5 + JSON.
        output_name:          Stem for output filenames.
        num_demos:            Limit number of trajectories replayed.
        obs_mode:             Observation mode for saved trajectories (e.g. 'rgbd', 'rgb').
        control_mode:         Control mode of the SOURCE trajectories.
        target_control_mode:  Desired control mode for OUTPUT trajectories.
                              If None or same as control_mode, no conversion.
                              Currently supports: pd_joint_pos → pd_ee_delta_pos.
        visualize:            Open human-render GUI while replaying.
    """
    if target_env not in TRANSFORMS:
        raise ValueError(f"Unknown target_env '{target_env}'. Choose from: {list(TRANSFORMS)}")

    cfg       = TRANSFORMS[target_env]
    angle_deg = cfg["angle_deg"]
    new_base  = cfg["base"]

    source_path = os.path.expanduser(source_path)
    output_dir  = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_control_mode = target_control_mode or control_mode
    do_ee_conversion    = (
        target_control_mode is not None
        and target_control_mode != control_mode
        and "ee" in target_control_mode
    )

    print(f"Source:        {source_path}")
    print(f"Target env:    {target_env}  (rotation {angle_deg:+.0f}° around Z)")
    print(f"Obs mode:      {obs_mode}")
    print(f"Control mode:  {control_mode} → {output_control_mode}")
    print(f"Output:        {output_dir}")

    render_mode = "human" if visualize else None

    # Primary env (output control mode + obs mode)
    env = gym.make(
        target_env,
        num_envs=1,
        obs_mode=obs_mode,
        control_mode=output_control_mode,
        render_mode=render_mode,
    )
    device = env.unwrapped.device

    # Secondary env for tracking original joint-pos trajectory (only needed for EE conversion)
    ori_env = None
    if do_ee_conversion:
        ori_env = gym.make(
            target_env,
            num_envs=1,
            obs_mode="state",
            control_mode=control_mode,
        )

    with h5py.File(source_path, 'r') as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith('traj')],
            key=lambda x: int(x.split('_')[1])
        )
    if num_demos:
        traj_keys = traj_keys[:num_demos]

    print(f"Replaying {len(traj_keys)} trajectories\n")

    stem        = output_name or "trajectory"
    output_h5   = os.path.join(output_dir, f"{stem}.h5")
    output_json = os.path.join(output_dir, f"{stem}.json")
    episodes_meta = []

    with h5py.File(output_h5, 'w') as out_f, h5py.File(source_path, 'r') as src_f:
        for traj_idx, traj_key in enumerate(tqdm(traj_keys, desc="Replaying", disable=visualize)):
            src_traj   = src_f[traj_key]
            actions    = src_traj['actions'][:]
            env_states = src_traj['env_states']

            # ---- Extract initial state from source ----------------------------
            cube_state  = env_states['actors']['cube'][0]
            goal_state  = env_states['actors']['goal_site'][0]
            panda_state = env_states['articulations']['panda'][0]

            cube_pos_new = transform_point(cube_state[0:3], angle_deg, new_base)
            cube_q_new   = transform_quat_wxyz(cube_state[3:7], angle_deg)
            goal_pos_new = transform_point(goal_state[0:3], angle_deg, new_base)
            goal_q_new   = goal_state[3:7]
            qpos         = panda_state[13:22]   # joint angles invariant to robot base pose

            # ---- Reset both envs to the same initial state -------------------
            _setup_env_state(env, cube_pos_new, cube_q_new, goal_pos_new, goal_q_new, qpos, device)
            if ori_env is not None:
                _setup_env_state(ori_env, cube_pos_new, cube_q_new, goal_pos_new, goal_q_new, qpos, device)

            # ---- Build EE conversion handles (once per trajectory) -----------
            handles = None
            if do_ee_conversion:
                handles = setup_ee_conversion(ori_env, env)
                # Grab the combined controller for from_action_dict
                handles["combined_ctrl"] = env.unwrapped.agent.controller

            if visualize:
                print(f"\nTrajectory {traj_idx}: {len(actions)} actions")
                print(f"  Cube (orig):        [{cube_state[0]:.3f}, {cube_state[1]:.3f}, {cube_state[2]:.3f}]")
                print(f"  Cube (transformed): [{cube_pos_new[0]:.3f}, {cube_pos_new[1]:.3f}, {cube_pos_new[2]:.3f}]")
                env.render()

            obs = env.unwrapped.get_obs()
            env_state_list = [env.unwrapped.get_state_dict()]
            obs_list, action_list, reward_list = [obs], [], []
            terminated_list, truncated_list = [], []
            success = False

            # ---- Episode loop ------------------------------------------------
            for action in actions:
                if do_ee_conversion:
                    # Step ori_env first so _target_qpos is updated
                    ori_action_t = torch.tensor([action], device=device, dtype=torch.float32)
                    ori_env.step(ori_action_t)

                    # Convert to EE delta action
                    ori_ctrl     = handles["ori_ctrl"]
                    ori_arm_ctrl = handles["ori_arm_ctrl"]
                    arm_ctrl     = handles["arm_ctrl"]
                    ee_link      = handles["ee_link"]
                    pin_model    = handles["pin_model"]
                    pos_only     = handles["pos_only"]
                    use_target   = handles["use_target"]
                    combined_ctrl = handles["combined_ctrl"]

                    full_qpos = ori_arm_ctrl.articulation.get_qpos().clone()
                    full_qpos[:, ori_arm_ctrl.active_joint_indices] = ori_arm_ctrl._target_qpos
                    pin_model.compute_forward_kinematics(full_qpos.cpu().numpy()[0])
                    target_ee_pose = Pose.create(
                        ori_arm_ctrl.articulation.pose.sp
                        * pin_model.get_link_pose(ee_link.index)
                    )

                    # Express target EE position in robot-base LOCAL frame.
                    # ManiSkill's formula (target.p - _target_pose.p - base.p) only
                    # works for identity-rotation robots. Our robot is rotated 90°,
                    # so we must rotate the world-frame target into base-local frame.
                    robot_base_pose = arm_ctrl.articulation.pose
                    target_ee_at_base = (robot_base_pose.inv() * target_ee_pose).p  # base-local

                    if use_target:
                        delta_pos = (target_ee_at_base - arm_ctrl._target_pose.p).cpu().numpy()[0]
                    else:
                        ee_at_base = (robot_base_pose.inv() * ee_link.pose).p
                        delta_pos = (target_ee_at_base - ee_at_base).cpu().numpy()[0]

                    low  = arm_ctrl.action_space_low.cpu().numpy()
                    high = arm_ctrl.action_space_high.cpu().numpy()
                    arm_action = np.clip(gym_utils.inv_scale_action(delta_pos, low, high), -1, 1)

                    # Keep gripper from original action, replace arm
                    ori_action_t_flat = torch.tensor(action, device=device, dtype=torch.float32)
                    ori_action_dict   = ori_ctrl.to_action_dict(ori_action_t_flat)
                    output_dict = {k: v for k, v in ori_action_dict.items()}
                    output_dict["arm"] = torch.tensor(arm_action, device=device, dtype=torch.float32)
                    action_t = combined_ctrl.from_action_dict(output_dict)

                    # Record the converted action (flat numpy)
                    saved_action = action_t.cpu().numpy()
                else:
                    action_t     = torch.tensor([action], device=device, dtype=torch.float32)
                    saved_action = action

                obs, reward, terminated, truncated, info = env.step(action_t)

                env_state_list.append(env.unwrapped.get_state_dict())
                obs_list.append(clone_obs_dict(obs))
                action_list.append(saved_action)
                reward_list.append(float(reward.cpu().item() if isinstance(reward, torch.Tensor) else reward))
                terminated_list.append(bool(terminated.cpu().item() if isinstance(terminated, torch.Tensor) else terminated))
                truncated_list.append(bool(truncated.cpu().item()  if isinstance(truncated, torch.Tensor) else truncated))

                if isinstance(info.get('success'), torch.Tensor):
                    if info['success'].any():
                        success = True
                elif info.get('success', False):
                    success = True

                if visualize:
                    env.render()

                if terminated or truncated:
                    break

            if visualize:
                print(f"  Success: {success}")

            grp = out_f.create_group(f'traj_{traj_idx}')
            grp.create_dataset('actions',    data=np.array(action_list))
            grp.create_dataset('rewards',    data=np.array(reward_list))
            grp.create_dataset('terminated', data=np.array(terminated_list))
            grp.create_dataset('truncated',  data=np.array(truncated_list))
            grp.create_dataset('success',    data=success)
            save_obs_to_hdf5(grp, obs_list)
            save_target_env_states_to_hdf5(grp, env_state_list)

            episodes_meta.append({
                'episode_id':   traj_idx,
                'episode_seed': traj_idx,
                'reset_kwargs': {},
                'control_mode': output_control_mode,
                'success':      success,
                'num_steps':    len(action_list),
                'source_traj':  traj_key,
            })

    meta = {
        'env_info': {
            'env_id':      target_env,
            'env_kwargs':  {'control_mode': output_control_mode, 'obs_mode': obs_mode},
        },
        'episodes': episodes_meta,
    }
    with open(output_json, 'w') as f:
        json.dump(meta, f, indent=2)

    env.close()
    if ori_env is not None:
        ori_env.close()

    success_count = sum(1 for e in episodes_meta if e['success'])
    print(f"\nSaved {len(episodes_meta)} trajectories ({success_count} successful, "
          f"{success_count/len(episodes_meta)*100:.1f}%)")
    print(f"Output: {output_h5}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source",       type=str, required=True,
                        help="Source H5 file (PickCube-v1 motionplanning demos)")
    parser.add_argument("--target_env",   type=str, required=True,
                        choices=list(TRANSFORMS),
                        help="Target environment to transform trajectories for")
    parser.add_argument("--output_dir",   type=str, required=True,
                        help="Directory to save transformed trajectory H5 + JSON")
    parser.add_argument("--output_name",  type=str, default=None,
                        help="Stem for output files (default: 'trajectory')")
    parser.add_argument("--num_demos",    type=int, default=None,
                        help="Limit number of trajectories to replay")
    parser.add_argument("--obs_mode",     type=str, default="rgbd",
                        help="Observation mode for output trajectories (default: rgbd)")
    parser.add_argument("--control_mode", type=str, default="pd_joint_pos",
                        help="Control mode of the SOURCE trajectories (default: pd_joint_pos)")
    parser.add_argument("--target_control_mode", type=str, default=None,
                        help="Output control mode. If omitted, same as --control_mode. "
                             "Supports pd_joint_pos → pd_ee_delta_pos.")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Watch replays in real-time to verify geometry")
    args = parser.parse_args()

    transform_and_replay(
        source_path=args.source,
        target_env=args.target_env,
        output_dir=args.output_dir,
        output_name=args.output_name,
        num_demos=args.num_demos,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        target_control_mode=args.target_control_mode,
        visualize=args.visualize,
    )
