# src/replay_demos_sideview.py
"""
Replay existing PickCube-v1 demos through PickCube-SideView-v1.
Same actions, same positions, just different camera view.

[0:3]   root_pose.p     # Robot base position (fixed, doesn't change)
[3:7]   root_pose.q     # Robot base orientation (fixed)
[7:16]  qpos            # Joint angles (these change during motion)
[16:22] root_vel        # Base velocity (always zero for fixed base)
[22:31] qvel            # Joint velocities
"""

import os
import sys
import h5py
import json
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom environment
from src.envs.pick_cube_side import PickCubeSideViewEnv
import mani_skill.envs
from mani_skill.utils.structs.pose import Pose


def replay_demos(
    source_path: str = "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5",
    output_dir: str = "~/.maniskill/demos/PickCube-SideView-v1/motionplanning",
    num_demos: int = None,
    control_mode: str = "pd_joint_pos",
    visualize: bool = False,
    save_video: bool = False,
    video_every_n: int = 10,
):
    source_path = os.path.expanduser(source_path)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if save_video:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    
    print(f"Loading source demos from: {source_path}")
    
    # Load source JSON for metadata
    source_json = source_path.replace('.h5', '.json')
    with open(source_json, 'r') as f:
        source_meta = json.load(f)
    
    # Create environment
    render_mode = "human" if visualize else ("rgb_array" if save_video else None)
    
    env = gym.make(
        "PickCube-SideView-v1",
        num_envs=1,
        obs_mode="rgbd",
        control_mode=control_mode,
        render_mode=render_mode,
    )
    device = env.unwrapped.device
    
    # Get trajectory keys
    with h5py.File(source_path, 'r') as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith('traj')],
            key=lambda x: int(x.split('_')[1])
        )
    
    if num_demos:
        traj_keys = traj_keys[:num_demos]
    
    print(f"Replaying {len(traj_keys)} trajectories")
    if visualize:
        print("Visualization mode: watching replays...")
    
    # Output files
    output_h5_path = os.path.join(output_dir, "trajectory.h5")
    output_json_path = os.path.join(output_dir, "trajectory.json")
    
    episodes_meta = []
    
    with h5py.File(output_h5_path, 'w') as out_f, h5py.File(source_path, 'r') as src_f:
        
        for traj_idx, traj_key in enumerate(tqdm(traj_keys, desc="Replaying", disable=visualize)):
            src_traj = src_f[traj_key]
            
            actions = src_traj['actions'][:]
            env_states = src_traj['env_states']
            
            # Set 
            cube_state = env_states['actors']['cube'][0]      # (13,)
            goal_state = env_states['actors']['goal_site'][0] # (13,)
            panda_state = env_states['articulations']['panda'][0]  # (31,)
            
            obs, info = env.reset()
            
            # Set cube pose (first 7 values: x, y, z, qw, qx, qy, qz), rest are velocity
            env.unwrapped.cube.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[cube_state[0], cube_state[1], cube_state[2]]], device=device),
                    q=torch.tensor([[cube_state[3], cube_state[4], cube_state[5], cube_state[6]]], device=device)
                )
            )
            
            # Set goal pose (first 7 values: x, y, z, qw, qx, qy, qz), rest are velocity
            env.unwrapped.goal_site.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[goal_state[0], goal_state[1], goal_state[2]]], device=device),
                    q=torch.tensor([[goal_state[3], goal_state[4], goal_state[5], goal_state[6]]], device=device)
                )
            )
            
            # Set robot state
            # robot state found after much pain: [root_p(3), root_q(4), qpos(9), root_vel(6), qvel(9)] = 31 total
            root_p = panda_state[0:3]
            root_q = panda_state[3:7]
            qpos = panda_state[7:16]
            qvel = panda_state[22:31]
            
            env.unwrapped.agent.robot.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[root_p[0], root_p[1], root_p[2]]], device=device),
                    q=torch.tensor([[root_q[0], root_q[1], root_q[2], root_q[3]]], device=device)
                )
            )
            env.unwrapped.agent.robot.set_qpos(torch.tensor([qpos], device=device))
            env.unwrapped.agent.robot.set_qvel(torch.tensor([qvel], device=device))
            
            # Render initial state if visualizing
            if visualize:
                print(f"\nTrajectory {traj_idx}: {len(actions)} actions")
                print(f"  Cube: [{cube_state[0]:.3f}, {cube_state[1]:.3f}, {cube_state[2]:.3f}]")
                print(f"  Goal: [{goal_state[0]:.3f}, {goal_state[1]:.3f}, {goal_state[2]:.3f}]")
                print(f"  Robot root: [{root_p[0]:.3f}, {root_p[1]:.3f}, {root_p[2]:.3f}]")
                env.render()
            
            # Get fresh observation after setting state
            obs = env.unwrapped.get_obs()
            
            # Storage
            obs_list = [obs]
            action_list = []
            reward_list = []
            terminated_list = []
            truncated_list = []
            video_frames = []
            
            should_save_video = save_video and (traj_idx % video_every_n == 0)
            
            if should_save_video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)
            
            # Replay actions
            success = False
            for action in actions:
                action_tensor = torch.tensor([action], device=device, dtype=torch.float32)
                obs, reward, terminated, truncated, info = env.step(action_tensor)
                
                obs_list.append(obs)
                action_list.append(action)
                
                if isinstance(reward, torch.Tensor):
                    reward_list.append(reward.cpu().item())
                else:
                    reward_list.append(float(reward))
                
                if isinstance(terminated, torch.Tensor):
                    terminated_list.append(terminated.cpu().item())
                else:
                    terminated_list.append(bool(terminated))
                    
                if isinstance(truncated, torch.Tensor):
                    truncated_list.append(truncated.cpu().item())
                else:
                    truncated_list.append(bool(truncated))
                
                if isinstance(info.get('success'), torch.Tensor):
                    if info['success'].any():
                        success = True
                elif info.get('success', False):
                    success = True
                
                # Render
                if visualize:
                    env.render()
                elif should_save_video:
                    frame = env.render()
                    if frame is not None:
                        video_frames.append(frame)
                
                if terminated or truncated:
                    break
            
            if visualize:
                print(f"  Success: {success}")
            
            # Save video
            if should_save_video and len(video_frames) > 0:
                video_path = os.path.join(video_dir, f"traj_{traj_idx:04d}.mp4")
                save_video_frames(video_frames, video_path)
            
            # Save trajectory
            grp = out_f.create_group(f'traj_{traj_idx}')
            grp.create_dataset('actions', data=np.array(action_list))
            grp.create_dataset('rewards', data=np.array(reward_list))
            grp.create_dataset('terminated', data=np.array(terminated_list))
            grp.create_dataset('truncated', data=np.array(truncated_list))
            grp.create_dataset('success', data=success)
            
            save_obs_to_hdf5(grp, obs_list)
            save_env_states_to_hdf5(grp, src_traj['env_states'], len(action_list) + 1)
            
            episodes_meta.append({
                'episode_id': traj_idx,
                'success': success,
                'num_steps': len(action_list),
                'source_traj': traj_key,
            })
    
    # Save JSON metadata
    meta = {
        'env_info': {
            'env_id': 'PickCube-SideView-v1',
            'env_kwargs': {
                'control_mode': control_mode,
                'obs_mode': 'rgbd',
            }
        },
        'episodes': episodes_meta,
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    env.close()
    
    success_count = sum(1 for e in episodes_meta if e['success'])
    print(f"\nSaved {len(episodes_meta)} trajectories ({success_count} successful)")
    print(f"Output: {output_h5_path}")
    
    if save_video:
        print(f"Videos saved to: {video_dir}")
    
    return output_h5_path


def save_video_frames(frames, path, fps=30):
    """Save frames as MP4 video."""
    try:
        import imageio
        frames_np = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in frames]
        imageio.mimsave(path, frames_np, fps=fps)
    except Exception as e:
        print(f"Failed to save video: {e}")


def save_obs_to_hdf5(grp, obs_list):
    """Save observations to HDF5 group."""
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


def save_env_states_to_hdf5(grp, src_env_states, num_steps):
    """Copy env_states from source."""
    env_states_grp = grp.create_group('env_states')
    
    actors_grp = env_states_grp.create_group('actors')
    for actor_name in src_env_states['actors'].keys():
        data = src_env_states['actors'][actor_name][:num_steps]
        actors_grp.create_dataset(actor_name, data=data)
    
    articulations_grp = env_states_grp.create_group('articulations')
    for art_name in src_env_states['articulations'].keys():
        data = src_env_states['articulations'][art_name][:num_steps]
        articulations_grp.create_dataset(art_name, data=data)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, 
                        default="~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5")
    parser.add_argument("--output_dir", type=str,
                        default="~/.maniskill/demos/PickCube-SideView-v1/motionplanning")
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--control_mode", type=str, default="pd_joint_pos")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Watch replays in real-time to verify")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_every_n", type=int, default=10)
    args = parser.parse_args()
    
    output_path = replay_demos(
        source_path=args.source,
        output_dir=args.output_dir,
        num_demos=args.num_demos,
        control_mode=args.control_mode,
        visualize=args.visualize,
        save_video=args.save_video,
        video_every_n=args.video_every_n,
    )
    
    if not args.visualize:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Convert to RGBD format for training:")
        print(f"   python -m mani_skill.trajectory.replay_trajectory \\")
        print(f"       --traj-path {output_path} \\")
        print(f"       --save-traj -o rgbd -c pd_joint_pos -b physx_cpu")
        print("\n2. Train the policy:")
        print(f"   python src/train_rgbd.py \\")
        print(f"       --env_id PickCube-SideView-v1 \\")
        print(f"       --demo_path {output_path.replace('.h5', '.rgbd.pd_joint_pos.physx_cpu.h5')} \\")
        print(f"       --control_mode pd_joint_pos \\")
        print(f"       --num_demos 200 --total_iters 50000")