import gymnasium as gym
import numpy as np
import torch


class SingleRobotInferenceWrapper(gym.Wrapper):
    """
    Adapts a two-robot Dict action space env to look like a single-robot env.
    - action_space / single_action_space → robot 0's Box
    - step(flat_action) → {'panda-0': flat_action, 'panda-1': zeros}
    - obs["agent"] → robot-0's sub-dict only

    Also patches raw_env._init_raw_obs so that downstream wrappers
    (FlattenRGBDObservationWrapper, FrameStack) see single-robot observation
    dimensions from the start.
    """

    def __init__(self, env):
        super().__init__(env)
        act_spaces = env.single_action_space.spaces  # {'panda-0': Box(4,), 'panda-1': Box(4,)}
        self._robot_keys = list(act_spaces.keys())   # ['panda-0', 'panda-1']
        self._r0_key = self._robot_keys[0]
        self._r1_key = self._robot_keys[1]
        r0_space = act_spaces[self._r0_key]
        self.single_action_space = r0_space
        self.action_space = gym.spaces.Box(
            r0_space.low, r0_space.high, shape=r0_space.shape, dtype=r0_space.dtype
        )
        # Patch raw_env._init_raw_obs so downstream wrappers (FlattenRGBD, FrameStack)
        # compute obs dims based on robot-0 only, not both robots.
        raw_env = env.unwrapped
        self._filter_agent_obs(raw_env._init_raw_obs)
        raw_env.update_obs_space(raw_env._init_raw_obs)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            zeros = torch.zeros_like(action)
        else:
            zeros = np.zeros_like(action)
        obs, rew, term, trunc, info = self.env.step(
            {self._r0_key: action, self._r1_key: zeros}
        )
        self._filter_agent_obs(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._filter_agent_obs(obs)
        return obs, info

    def _filter_agent_obs(self, obs):
        """Replace obs["agent"] with robot-0's sub-dict in-place."""
        obs["agent"] = obs["agent"][self._r0_key]
        return obs
