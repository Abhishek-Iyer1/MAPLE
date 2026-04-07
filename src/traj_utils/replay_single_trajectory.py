import h5py
import json
import gymnasium as gym
import mani_skill.envs
import time

def replay_trajectory(h5_path, episode_index=0):
    json_path = h5_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        meta = json.load(f)
    
    env_id = meta["env_info"]["env_id"]
    env_kwargs = meta["env_info"]["env_kwargs"]
    env_kwargs["render_mode"] = "human"
    env = gym.make(env_id, **env_kwargs)
    
    episode_meta = meta["episodes"][episode_index]
    traj_id = f"traj_{episode_meta['episode_id']}"
    
    with h5py.File(h5_path, "r") as f:
        actions = f[traj_id]["actions"][:]
    
    print(f"Replaying {traj_id} from {h5_path}...")
    
    env.reset(seed=episode_meta["episode_seed"], options=episode_meta.get("reset_kwargs", {}))
    
    for action in actions:
        env.step(action)
        env.render() # Updates the GUI window
        # time.sleep(0.01) # Slow it down a bit to see the motion

    env.close()

if __name__ == "__main__":
    H5_FILE = "/home/stryder/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_joint_delta_pos.physx_cpu.h5"
    replay_trajectory(H5_FILE, episode_index=0)