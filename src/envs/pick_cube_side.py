# src/envs/pick_cube_sideview.py
"""
PickCube-v1 with side-view camera.
Robot stays at original position, only camera changes.
"""

from typing import Any, Tuple
import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

ROBOT_POSE_LEFT = sapien.Pose(p=[0, -0.9, 0], q=[-0.7071068, 0, 0, -0.7071068])
ROBOT_POSE_RIGHT = sapien.Pose(p=[0, 0.9, 0], q=[0.7071068, 0, 0, -0.7071068])
BASE_CAMERA_POSE = sapien_utils.look_at(eye=[1.0, 0, 0.75], target=[0.0, 0.0, 0.25])
RENDER_CAMERA_POSE = sapien_utils.look_at(eye=[1.4, 0.8, 0.75], target=[0.0, 0.1, 0.1])
BASE_CAMERA_CONFIG = CameraConfig("base_camera", BASE_CAMERA_POSE, 128, 128, np.pi / 2, 0.01, 100)
RENDER_CAMERA_CONFIG = CameraConfig("render_camera", RENDER_CAMERA_POSE, 512, 512, 1, 0.01, 100)

@register_env("PickCube-SideView-v1", max_episode_steps=100)
class PickCubeSideViewEnv(PickCubeEnv):
    """
    PickCube with side-view camera for multi-robot compatibility.
    Inherits everything from PickCubeEnv, only changes camera.
    """

    @property
    def _default_sensor_configs(self):
        # Side view camera - pulled back to see where second robot would be
        pose = BASE_CAMERA_POSE
        return [BASE_CAMERA_CONFIG]

    @property
    def _default_human_render_camera_configs(self):
        pose = RENDER_CAMERA_POSE
        return RENDER_CAMERA_CONFIG


@register_env("PickCube-SideView-TwoRobot-v1", max_episode_steps=100)
class PickCubeSideViewTwoRobotEnv(PickCubeSideViewEnv):
    """
    PickCube-SideView with a second idle Panda added to the scene.
    Robot 0 (at [-0.615, 0, 0]) has the task; Robot 1 (at [0, 0.8, 0]) is passive.
    """

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    def __init__(self, *args, robot_uids=("panda", "panda"), **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(
            self, options,
            [ROBOT_POSE_LEFT,   # robot 0: same as PickCubeEnv
             ROBOT_POSE_RIGHT]  # robot 1: behind table, facing -Y
        )

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        # TableSceneBuilder overrides ("panda","panda") robot 0 to [0,-0.75,0]+90°.
        # Restore to training-compatible pose.
        self.agent.agents[0].robot.set_pose(ROBOT_POSE_LEFT)
        self.agent.agents[1].robot.set_pose(ROBOT_POSE_RIGHT)

    def evaluate(self):
        agent0 = self.agent.agents[0]
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = agent0.is_grasping(self.cube)
        is_robot_static = agent0.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def _get_obs_extra(self, info: dict):
        agent0 = self.agent.agents[0]
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=agent0.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - agent0.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs, action, info):
        agent0 = self.agent.agents[0]
        tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - agent0.tcp_pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        is_grasped = info["is_grasped"]
        reward += is_grasped
        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        reward += (1 - torch.tanh(5 * obj_to_goal_dist)) * is_grasped
        # qvel static reward for robot 0 only (panda: slice off last 2 finger joints)
        qvel = agent0.robot.get_qvel()[..., :-2]
        reward += (1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))) * info["is_obj_placed"]
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
    

# Test environment with robot on the side
@register_env("PickCube-SideView-Right-v1", max_episode_steps=100)
class PickCubeSideViewEnvRight(PickCubeSideViewEnv):
    """
    PickCube with side-view camera for multi-robot compatibility.
    Inherits everything from PickCubeEnv, only changes camera.
    """

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(self, options, ROBOT_POSE_RIGHT)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        # TableSceneBuilder resets the robot pose — restore it
        self.agent.robot.set_pose(ROBOT_POSE_RIGHT)

@register_env("PickCube-SideView-Left-v1", max_episode_steps=100)
class PickCubeSideViewEnvLeft(PickCubeSideViewEnv):
    """
    PickCube with side-view camera for multi-robot compatibility.
    Inherits everything from PickCubeEnv, only changes camera.
    """

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(self, options, ROBOT_POSE_LEFT)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        # TableSceneBuilder resets the robot pose — restore it
        self.agent.robot.set_pose(ROBOT_POSE_LEFT)