# src/envs/two_robot_pick_cube.py
"""
Two-robot PickCube: Each arm has its own cube and goal.
Based on ManiSkill's TwoRobotPickCube-v1.
"""

from typing import Any, Tuple

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("TwoRobotTwoCubePickCube-v1", max_episode_steps=100)
class TwoRobotTwoCubePickCube(BaseEnv):
    """
    Two Panda robots, each with its own cube and goal.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0, 0.75], [0.0, 0.0, 0.25])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Cube for left robot
        self.cube_0 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube_0",
            initial_pose=sapien.Pose(p=[0, -0.15, 0.02]),
        )

        # Cube for right robot
        self.cube_1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],
            name="cube_1",
            initial_pose=sapien.Pose(p=[0, 0.15, 0.02]),
        )

        # Goal for left robot
        self.goal_site_0 = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site_0",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        # Goal for right robot
        self.goal_site_1 = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 1, 1],
            name="goal_site_1",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        self._hidden_objects.append(self.goal_site_0)
        self._hidden_objects.append(self.goal_site_1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Cube 0 position (left side, for left robot)
            xyz_0 = torch.zeros((b, 3))
            xyz_0[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            xyz_0[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            xyz_0[:, 2] = self.cube_half_size
            qs_0 = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube_0.set_pose(Pose.create_from_pq(xyz_0, qs_0))

            # Cube 1 position (right side, for right robot)
            xyz_1 = torch.zeros((b, 3))
            xyz_1[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            xyz_1[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            xyz_1[:, 2] = self.cube_half_size
            qs_1 = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube_1.set_pose(Pose.create_from_pq(xyz_1, qs_1))

            # Goal 0 position (left side)
            goal_xyz_0 = torch.zeros((b, 3))
            goal_xyz_0[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz_0[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            goal_xyz_0[:, 2] = torch.rand((b,)) * 0.3 + xyz_0[:, 2]
            self.goal_site_0.set_pose(Pose.create_from_pq(goal_xyz_0))

            # Goal 1 position (right side)
            goal_xyz_1 = torch.zeros((b, 3))
            goal_xyz_1[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz_1[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            goal_xyz_1[:, 2] = torch.rand((b,)) * 0.3 + xyz_1[:, 2]
            self.goal_site_1.set_pose(Pose.create_from_pq(goal_xyz_1))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        is_obj_placed_0 = (
            torch.linalg.norm(self.goal_site_0.pose.p - self.cube_0.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_obj_placed_1 = (
            torch.linalg.norm(self.goal_site_1.pose.p - self.cube_1.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_left_arm_static = self.left_agent.is_static(0.2)
        is_right_arm_static = self.right_agent.is_static(0.2)

        return {
            "success": is_obj_placed_0 & is_obj_placed_1 & is_left_arm_static & is_right_arm_static,
            "is_obj_placed_0": is_obj_placed_0,
            "is_obj_placed_1": is_obj_placed_1,
            "is_left_arm_static": is_left_arm_static,
            "is_right_arm_static": is_right_arm_static,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
            goal_pos_0=self.goal_site_0.pose.p,
            goal_pos_1=self.goal_site_1.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                cube_pose_0=self.cube_0.pose.raw_pose,
                cube_pose_1=self.cube_1.pose.raw_pose,
                left_arm_tcp_to_cube_pos=self.cube_0.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cube_pos=self.cube_1.pose.p - self.right_agent.tcp.pose.p,
                cube_to_goal_pos_0=self.goal_site_0.pose.p - self.cube_0.pose.p,
                cube_to_goal_pos_1=self.goal_site_1.pose.p - self.cube_1.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = torch.zeros(self.num_envs, device=self.device)

        # Left robot: reach and place
        tcp_to_obj_dist_0 = torch.linalg.norm(
            self.cube_0.pose.p - self.left_agent.tcp.pose.p, axis=1
        )
        reaching_reward_0 = 1 - torch.tanh(5 * tcp_to_obj_dist_0)
        reward += reaching_reward_0

        obj_to_goal_dist_0 = torch.linalg.norm(
            self.goal_site_0.pose.p - self.cube_0.pose.p, axis=1
        )
        place_reward_0 = 1 - torch.tanh(5 * obj_to_goal_dist_0)
        reward += place_reward_0

        # Right robot: reach and place
        tcp_to_obj_dist_1 = torch.linalg.norm(
            self.cube_1.pose.p - self.right_agent.tcp.pose.p, axis=1
        )
        reaching_reward_1 = 1 - torch.tanh(5 * tcp_to_obj_dist_1)
        reward += reaching_reward_1

        obj_to_goal_dist_1 = torch.linalg.norm(
            self.goal_site_1.pose.p - self.cube_1.pose.p, axis=1
        )
        place_reward_1 = 1 - torch.tanh(5 * obj_to_goal_dist_1)
        reward += place_reward_1

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10