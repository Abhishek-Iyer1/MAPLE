import gymnasium as gym
import mani_skill.envs

from src.envs.pick_cube_side import PickCubeSideViewEnv  # noqa — registers all SideView envs


env = gym.make(
    # "PickCube-SideView-v1",
    # "PickCube-SideView-TwoRobot-v1",
    # "PickCube-SideView-Left-v1",
    # "PickCube-SideView-Right-v1",
    "PickCube-SideView-DualTask-v1",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    render_mode="rgb_array"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated
    env.render()
env.close()