# src/envs/collect_demos.py
"""
Collect demonstrations using motion planning.
"""

import gymnasium as gym
import numpy as np
import h5py
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.one_robot_pick_cube import OneRobotPickCube
from src.envs.two_robot_pick_cube import TwoRobotTwoCubePickCube

from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvers
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def collect_one_robot_demos(num_episodes=100, output_dir="demos/OneRobotPickCube-v1"):
    """Collect demos for single robot environment using motion planning."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    env = gym.make(
        "OneRobotPickCube-v1",
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    
    # Initialize motion planner
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.agent.robot.pose,
    )
    
    successful_episodes = []
    pbar = tqdm(total=num_episodes, desc="Collecting demos")
    attempts = 0
    max_attempts = num_episodes * 3
    
    while len(successful_episodes) < num_episodes and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset()
        
        # Get cube and goal positions
        cube_pos = env.cube_0.pose.p[0].cpu().numpy()
        goal_pos = env.goal_site_0.pose.p[0].cpu().numpy()
        
        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }
        
        try:
            # Phase 1: Move to pre-grasp position above cube
            pre_grasp_pos = cube_pos.copy()
            pre_grasp_pos[2] += 0.1  # 10cm above cube
            
            result = planner.move_to_pose(pre_grasp_pos, env.agent.tcp.pose.q[0].cpu().numpy())
            if result == -1:
                continue
            
            # Execute planned path
            for action in planner.planned_path:
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["observations"].append(obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminated"].append(terminated)
                trajectory["truncated"].append(truncated)
                trajectory["infos"].append(info)
            
            # Phase 2: Move down to grasp
            grasp_pos = cube_pos.copy()
            grasp_pos[2] += 0.02  # At cube height
            
            result = planner.move_to_pose(grasp_pos, env.agent.tcp.pose.q[0].cpu().numpy())
            if result == -1:
                continue
                
            for action in planner.planned_path:
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["observations"].append(obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminated"].append(terminated)
                trajectory["truncated"].append(truncated)
                trajectory["infos"].append(info)
            
            # Phase 3: Close gripper
            for _ in range(10):
                action = np.zeros(env.action_space.shape)
                action[-1] = -1  # Close gripper
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["observations"].append(obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminated"].append(terminated)
                trajectory["truncated"].append(truncated)
                trajectory["infos"].append(info)
            
            # Phase 4: Lift to goal
            result = planner.move_to_pose(goal_pos, env.agent.tcp.pose.q[0].cpu().numpy())
            if result == -1:
                continue
                
            for action in planner.planned_path:
                action[-1] = -1  # Keep gripper closed
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["observations"].append(obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminated"].append(terminated)
                trajectory["truncated"].append(truncated)
                trajectory["infos"].append(info)
            
            # Check success
            if info.get("success", False):
                successful_episodes.append(trajectory)
                pbar.update(1)
                
        except Exception as e:
            print(f"Episode failed: {e}")
            continue
    
    pbar.close()
    env.close()
    
    print(f"Collected {len(successful_episodes)} successful episodes")
    
    # Save trajectories
    save_trajectories(successful_episodes, output_dir)
    
    return successful_episodes


def save_trajectories(trajectories, output_dir):
    """Save trajectories to HDF5 file."""
    
    h5_path = os.path.join(output_dir, "trajectory.h5")
    
    with h5py.File(h5_path, 'w') as f:
        for i, traj in enumerate(trajectories):
            grp = f.create_group(f"traj_{i}")
            grp.create_dataset("actions", data=np.array(traj["actions"]))
            
            # Save observations
            obs_grp = grp.create_group("obs")
            
            # Handle nested observation structure
            first_obs = traj["observations"][0]
            save_obs_recursive(obs_grp, traj["observations"], first_obs)
    
    print(f"Saved trajectories to {h5_path}")


def save_obs_recursive(grp, obs_list, template):
    """Recursively save observations to HDF5."""
    
    if isinstance(template, dict):
        for key in template.keys():
            sub_grp = grp.create_group(key)
            sub_template = template[key]
            sub_obs_list = [obs[key] for obs in obs_list]
            save_obs_recursive(sub_grp, sub_obs_list, sub_template)
    else:
        # It's an array
        data = np.stack([obs.cpu().numpy() if hasattr(obs, 'cpu') else obs for obs in obs_list])
        grp.create_dataset("data", data=data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["one", "two"], default="one")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.env == "one":
        output_dir = args.output_dir or "demos/OneRobotPickCube-v1/motionplanning"
        collect_one_robot_demos(args.num_episodes, output_dir)
    else:
        print("Two robot demo collection not yet implemented")