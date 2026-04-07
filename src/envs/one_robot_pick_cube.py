# src/envs/one_robot_pick_cube.py
"""
One-robot PickCube: EXACT copy of TwoRobotTwoCubePickCube but with only one robot.
Same scene, camera, everything - just no second robot, cube, or goal.
"""

from typing import Any

import numpy as np
import sapien
import torch

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


@register_env("OneRobotPickCube-v1", max_episode_steps=100)
class OneRobotPickCube(BaseEnv):
    """
    One Panda robot with one cube and goal. Same scene as TwoRobotTwoCubePickCube.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Panda
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
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
            options, sapien.Pose(p=[0, -1, 0])
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Cube for robot
        self.cube_0 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube_0",
            initial_pose=sapien.Pose(p=[0, -0.15, 0.02]),
        )

        # Goal for robot
        self.goal_site_0 = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site_0",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        self._hidden_objects.append(self.goal_site_0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Override robot position to match left robot in two-robot env
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[0.0, -0.8, 0.0]], device=self.device).expand(b, -1),
                    q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(b, -1)
                )
            )
            
            # Reset qpos to rest position
            qpos = self.agent.keyframes["rest"].qpos
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(torch.zeros_like(torch.tensor(qpos)))

            # Cube 0 position (left side, for robot)
            xyz_0 = torch.zeros((b, 3))
            xyz_0[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            xyz_0[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            xyz_0[:, 2] = self.cube_half_size
            qs_0 = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube_0.set_pose(Pose.create_from_pq(xyz_0, qs_0))

            # Goal 0 position (left side)
            goal_xyz_0 = torch.zeros((b, 3))
            goal_xyz_0[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz_0[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            goal_xyz_0[:, 2] = torch.rand((b,)) * 0.3 + xyz_0[:, 2]
            self.goal_site_0.set_pose(Pose.create_from_pq(goal_xyz_0))

    def evaluate(self):
        is_obj_placed_0 = (
            torch.linalg.norm(self.goal_site_0.pose.p - self.cube_0.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_arm_static = self.agent.is_static(0.2)

        return {
            "success": is_obj_placed_0 & is_arm_static,
            "is_obj_placed_0": is_obj_placed_0,
            "is_arm_static": is_arm_static,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.agent.tcp.pose.raw_pose,
            goal_pos_0=self.goal_site_0.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                cube_pose_0=self.cube_0.pose.raw_pose,
                left_arm_tcp_to_cube_pos=self.cube_0.pose.p - self.agent.tcp.pose.p,
                cube_to_goal_pos_0=self.goal_site_0.pose.p - self.cube_0.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = torch.zeros(self.num_envs, device=self.device)

        # Robot: reach and place
        tcp_to_obj_dist_0 = torch.linalg.norm(
            self.cube_0.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward_0 = 1 - torch.tanh(5 * tcp_to_obj_dist_0)
        reward += reaching_reward_0

        obj_to_goal_dist_0 = torch.linalg.norm(
            self.goal_site_0.pose.p - self.cube_0.pose.p, axis=1
        )
        place_reward_0 = 1 - torch.tanh(5 * obj_to_goal_dist_0)
        reward += place_reward_0

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10