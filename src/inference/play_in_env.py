import gymnasium as gym
import mani_skill.envs

# Import your custom env to register it
from src.envs.two_robot_pick_cube import TwoRobotTwoCubePickCube
from src.envs.one_robot_pick_cube import OneRobotPickCube
from src.envs.pick_cube_side import PickCubeSideViewEnv


env = gym.make(
    # "TwoRobotTwoCubePickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    # "OneRobotPickCube-v1",
    # "PickCube-v1",
    "PickCube-SideView-v1",
    num_envs=1,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated
    env.render()  # a display is required to render
env.close()