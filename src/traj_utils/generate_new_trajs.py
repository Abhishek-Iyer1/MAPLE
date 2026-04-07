import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.motion_planning.pick_cube import PickCubeSolver
from tqdm import tqdm

def generate_new_data(num_episodes, output_dir, env_id="PickCube-SideView-v1"):
    # 1. Setup Environment
    # If you want to change robot positions, pass them via env_kwargs
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_pos", # Planners usually work in absolute pos
        render_mode="rgb_array"
    )
    
    env = RecordEpisode(env, output_dir=output_dir, save_trajectory=True)
    
    # 2. Initialize the Task-Specific Solver
    solver = PickCubeSolver(env)
    
    success_count = 0
    pbar = tqdm(total=num_episodes, desc="Generating New Demos")
    
    while success_count < num_episodes:
        obs, _ = env.reset()
        
        # Solve the task (returns a list of actions)
        result = solver.solve()
        
        if result["success"]:
            # Execute the planned actions
            for action in result["actions"]:
                env.step(action)
            
            env.flush_trajectory()
            success_count += 1
            pbar.update(1)
        else:
            # If the planner fails, we just try a new seed
            continue

    env.close()

if __name__ == "__main__":
    generate_new_data(
        num_episodes=100, 
        output_dir="demos/PickCube-SideView-v1/planned"
    )