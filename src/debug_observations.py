# src/debug_observations.py
"""
Visualize what the policy sees during training/inference.
Saves video of RGB, depth, and policy actions.
"""

import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import cv2
import os
from datetime import datetime

import sys
sys.path.insert(0, "src")

from train_rgbd import Agent, Args
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


def to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def visualize_observations(checkpoint_path, num_episodes=3, save_dir="debug_videos"):
    """Run policy and save visualization of observations."""
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Config (must match training)
    args = Args()
    args.env_id = "StackCube-v1"
    args.control_mode = "pd_joint_delta_pos"
    args.obs_horizon = 2
    args.act_horizon = 8
    args.pred_horizon = 16
    args.max_episode_steps = 200
    args.obs_mode = "rgbd"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create env
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    envs = make_eval_envs(
        args.env_id,
        num_envs=1,
        sim_backend="cpu",
        env_kwargs=env_kwargs,
        other_kwargs=other_kwargs,
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    # Load agent
    agent = Agent(envs, args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["ema_agent"])
    agent.eval()
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")
    
    # Check observation shapes
    obs, _ = envs.reset()
    print("\n=== Observation Shapes ===")
    for key, val in obs.items():
        val_np = to_numpy(val)
        print(f"  {key}: {val_np.shape}, dtype={val_np.dtype}, range=[{val_np.min():.2f}, {val_np.max():.2f}]")
    
    # Video writer
    video_path = os.path.join(save_dir, f"obs_debug_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    for ep in range(num_episodes):
        obs, _ = envs.reset()
        done = False
        step = 0
        
        while not done and step < args.max_episode_steps:
            # Get observation tensors for policy
            obs_tensor = {
                "state": obs["state"].float().to(device) if isinstance(obs["state"], torch.Tensor) else torch.from_numpy(obs["state"]).float().to(device),
                "rgb": obs["rgb"].to(device) if isinstance(obs["rgb"], torch.Tensor) else torch.from_numpy(obs["rgb"]).to(device),
                "depth": obs["depth"].to(device) if isinstance(obs["depth"], torch.Tensor) else torch.from_numpy(obs["depth"]).to(device),
            }
            
            # Get action
            with torch.no_grad():
                actions = agent.get_action(obs_tensor)
            action = actions[:, 0, :].cpu().numpy()
            
            # === Visualization ===
            # Convert to numpy and remove batch dim
            rgb_obs = to_numpy(obs["rgb"])[0]      # (obs_horizon, H, W, C*num_cams)
            depth_obs = to_numpy(obs["depth"])[0]  # (obs_horizon, H, W, num_cams)
            state_obs = to_numpy(obs["state"])[0]  # (obs_horizon, state_dim)
            
            # Use the latest observation in the horizon (index -1)
            rgb_obs = rgb_obs[-1]    # (H, W, C*num_cams)
            depth_obs = depth_obs[-1]  # (H, W, num_cams)
            state_obs = state_obs[-1]  # (state_dim,)
            
            # Get third-person render
            render_frame = envs.call("render")[0]
            
            # Process RGB - split if multiple cameras (6 channels = 2 cameras * 3 RGB)
            num_rgb_channels = rgb_obs.shape[-1]
            rgb_panels = []
            if num_rgb_channels > 3:
                num_cams = num_rgb_channels // 3
                for i in range(num_cams):
                    cam_rgb = rgb_obs[..., i*3:(i+1)*3].astype(np.uint8)
                    cam_rgb = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
                    rgb_panels.append(cam_rgb)
            else:
                rgb_panels = [cv2.cvtColor(rgb_obs.astype(np.uint8), cv2.COLOR_RGB2BGR)]
            
            # Process depth - split if multiple cameras
            num_depth_channels = depth_obs.shape[-1]
            depth_panels = []
            for i in range(num_depth_channels):
                d = depth_obs[..., i].astype(np.float32)
                # Normalize to 0-255
                d_min, d_max = d.min(), d.max()
                if d_max > d_min:
                    d_normalized = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    d_normalized = np.zeros_like(d, dtype=np.uint8)
                d_colored = cv2.applyColorMap(d_normalized, cv2.COLORMAP_JET)
                depth_panels.append(d_colored)
            
            # Render frame
            render_bgr = cv2.cvtColor(render_frame.cpu().numpy(), cv2.COLOR_RGB2BGR)
            
            # Resize all panels
            target_height = 200
            
            def resize_panel(img, height):
                scale = height / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), height))
            
            rgb_panels = [resize_panel(p, target_height) for p in rgb_panels]
            depth_panels = [resize_panel(p, target_height) for p in depth_panels]
            render_bgr = resize_panel(render_bgr, target_height)
            
            # Add labels
            def add_label(img, text):
                img = img.copy()
                cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return img
            
            cam_names = ["base_cam", "hand_cam"]
            for i, p in enumerate(rgb_panels):
                label = cam_names[i] if i < len(cam_names) else f"cam{i}"
                rgb_panels[i] = add_label(p, f"RGB {label}")
            for i, p in enumerate(depth_panels):
                label = cam_names[i] if i < len(cam_names) else f"cam{i}"
                depth_panels[i] = add_label(p, f"Depth {label}")
            render_bgr = add_label(render_bgr, "Third-person")
            
            # Combine panels
            top_row = np.hstack(rgb_panels + [render_bgr])
            bottom_row = np.hstack(depth_panels)
            
            # Pad rows to match width
            if bottom_row.shape[1] < top_row.shape[1]:
                pad = np.zeros((bottom_row.shape[0], top_row.shape[1] - bottom_row.shape[1], 3), dtype=np.uint8)
                bottom_row = np.hstack([bottom_row, pad])
            elif bottom_row.shape[1] > top_row.shape[1]:
                bottom_row = bottom_row[:, :top_row.shape[1]]
            
            combined = np.vstack([top_row, bottom_row])
            
            # Add info
            info_text = f"Ep {ep+1} | Step {step} | State dim: {state_obs.shape[0]}"
            cv2.putText(combined, info_text, (10, combined.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            action_text = f"Action: [{action[0, 0]:.2f}, {action[0, 1]:.2f}, {action[0, 2]:.2f}, ... {action[0, -1]:.2f}]"
            cv2.putText(combined, action_text, (10, combined.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if video_writer is None:
                h, w = combined.shape[:2]
                video_writer = cv2.VideoWriter(video_path, fourcc, 20, (w, h))
            
            video_writer.write(combined)
            
            # Step environment
            obs, reward, terminated, truncated, info = envs.step(action)
            step += 1
            done = terminated[0] or truncated[0]
        
        success = info.get('success', [False])[0]
        print(f"Episode {ep+1}: steps={step}, success={success}")
    
    video_writer.release()
    envs.close()
    
    print(f"\n=== Video saved to: {video_path} ===")
    print(f"Open with: vlc {video_path}")
    
    return video_path


def check_observation_stats(demo_path):
    """Check statistics of observations in the demo dataset."""
    import h5py
    
    print(f"\n=== Checking demo file: {demo_path} ===\n")
    
    def print_hdf5_structure(name, obj):
        """Recursively print HDF5 structure."""
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/")
    
    with h5py.File(demo_path, 'r') as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj')])
        print(f"Number of trajectories: {len(traj_keys)}")
        
        if traj_keys:
            print(f"\n=== Structure of {traj_keys[0]} ===")
            f[traj_keys[0]].visititems(print_hdf5_structure)
            
            traj = f[traj_keys[0]]
            
            if 'obs/sensor_data' in traj:
                sensor_data = traj['obs/sensor_data']
                print(f"\n=== Sensor Data Keys ===")
                for cam_name in sensor_data.keys():
                    print(f"\nCamera: {cam_name}")
                    cam = sensor_data[cam_name]
                    for data_type in cam.keys():
                        data = cam[data_type]
                        sample = data[0]
                        print(f"  {data_type}: shape={data.shape}, range=[{np.min(sample):.2f}, {np.max(sample):.2f}]")
            
            if 'actions' in traj:
                actions = traj['actions'][:]
                print(f"\n=== Actions ===")
                print(f"  shape: {actions.shape}")
                print(f"  range: [{actions.min():.4f}, {actions.max():.4f}]")
                print(f"  mean: {actions.mean():.4f}, std: {actions.std():.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to checkpoint")
    parser.add_argument("--demo_path", "-d", type=str, default=None,
                       help="Path to demo h5 file to check stats")
    parser.add_argument("--num_episodes", "-n", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="debug_videos")
    args = parser.parse_args()
    
    if args.demo_path:
        check_observation_stats(args.demo_path)
    
    visualize_observations(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
    )