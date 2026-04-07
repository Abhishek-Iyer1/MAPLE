# src/envs/two_robot_pick_cube_custom_cam.py
"""
Two-robot PickCube environment for multi-agent coordination research.
"""

import numpy as np
import sapien
import torch
from typing import Any
from scipy.spatial.transform import Rotation

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import mani_skill.envs.utils.randomization as randomization


def euler2quat(roll, pitch, yaw):
    """Convert euler angles (XYZ) to quaternion (wxyz)."""
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    quat = rot.as_quat()  # Returns xyzw
    return [quat[3], quat[0], quat[1], quat[2]]  # Convert to wxyz


@register_env("TwoRobotPickCubeCustomCam-v1", max_episode_steps=100)
class TwoRobotPickCubeCustomCamEnv(BaseEnv):
    """Two Panda arms side by side, each with their own cube and goal."""
    
    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: Any
    
    robot_spacing = 0.35
    cube_half_size = 0.02
    goal_thresh = 0.025
    
    def __init__(self, *args, robot_uids=("panda", "panda"), **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.0, -0.9, 0.7],
            target=[0.0, 0.0, 0.1]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi/2, 0.01, 100)]
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.0, -1.2, 0.8],
            target=[0.0, 0.0, 0.1]
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    def _load_scene(self, options: dict):
        """Load table and objects."""
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        
        self.cube_0 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube_0",
            initial_pose=sapien.Pose(p=[-0.15, 0.0, self.cube_half_size]),
        )
        
        self.cube_1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],
            name="cube_1",
            initial_pose=sapien.Pose(p=[0.15, 0.0, self.cube_half_size]),
        )
        
        self.goal_0 = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1, 0.5, 0.5, 0.5],
            name="goal_0",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.15, 0.1, 0.1]),
        )
        
        self.goal_1 = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0.5, 0.5, 1, 0.5],
            name="goal_1",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.15, 0.1, 0.1]),
        )
        
        self._hidden_objects.extend([self.goal_0, self.goal_1])
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with robot positions and randomized objects."""
        
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # === Set robot base positions ===
            # Robot 0: Left side, facing forward (+Y)
            q0 = euler2quat(0, 0, np.pi/2)
            robot0_pose = Pose.create_from_pq(
                p=torch.tensor([[-self.robot_spacing, -0.5, 0.0]], device=self.device).expand(b, -1),
                q=torch.tensor([q0], device=self.device).expand(b, -1)
            )
            
            # Robot 1: Right side, facing forward (+Y)
            robot1_pose = Pose.create_from_pq(
                p=torch.tensor([[self.robot_spacing, -0.5, 0.0]], device=self.device).expand(b, -1),
                q=torch.tensor([q0], device=self.device).expand(b, -1)
            )
            
            # Apply poses to robots
            self.agent.agents[0].robot.set_pose(robot0_pose)
            self.agent.agents[1].robot.set_pose(robot1_pose)
            
            # Reset robot joint positions to home
            for agent in self.agent.agents:
                qpos = agent.keyframes["rest"].qpos
                agent.robot.set_qpos(qpos)
                agent.robot.set_qvel(torch.zeros_like(qpos))
            
            # === Randomize cube positions ===
            cube0_xyz = torch.zeros((b, 3))
            cube0_xyz[:, 0] = -0.15 + (torch.rand((b,)) * 0.06 - 0.03)
            cube0_xyz[:, 1] = 0.0 + (torch.rand((b,)) * 0.06 - 0.03)
            cube0_xyz[:, 2] = self.cube_half_size
            cube0_q = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube_0.set_pose(Pose.create_from_pq(cube0_xyz, cube0_q))
            
            cube1_xyz = torch.zeros((b, 3))
            cube1_xyz[:, 0] = 0.15 + (torch.rand((b,)) * 0.06 - 0.03)
            cube1_xyz[:, 1] = 0.0 + (torch.rand((b,)) * 0.06 - 0.03)
            cube1_xyz[:, 2] = self.cube_half_size
            cube1_q = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube_1.set_pose(Pose.create_from_pq(cube1_xyz, cube1_q))
            
            # === Randomize goal positions ===
            goal0_xyz = torch.zeros((b, 3))
            goal0_xyz[:, 0] = -0.15 + (torch.rand((b,)) * 0.06 - 0.03)
            goal0_xyz[:, 1] = 0.1 + (torch.rand((b,)) * 0.06 - 0.03)
            goal0_xyz[:, 2] = 0.08 + torch.rand((b,)) * 0.08
            self.goal_0.set_pose(Pose.create_from_pq(goal0_xyz))
            
            goal1_xyz = torch.zeros((b, 3))
            goal1_xyz[:, 0] = 0.15 + (torch.rand((b,)) * 0.06 - 0.03)
            goal1_xyz[:, 1] = 0.1 + (torch.rand((b,)) * 0.06 - 0.03)
            goal1_xyz[:, 2] = 0.08 + torch.rand((b,)) * 0.08
            self.goal_1.set_pose(Pose.create_from_pq(goal1_xyz))
    
    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose_0=self.agent.agents[0].tcp_pose.raw_pose,
            goal_pos_0=self.goal_0.pose.p,
            tcp_pose_1=self.agent.agents[1].tcp_pose.raw_pose,
            goal_pos_1=self.goal_1.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                cube_pose_0=self.cube_0.pose.raw_pose,
                cube_pose_1=self.cube_1.pose.raw_pose,
            )
        return obs
    
    def evaluate(self):
        dist_0 = torch.linalg.norm(self.goal_0.pose.p - self.cube_0.pose.p, axis=1)
        success_0 = dist_0 <= self.goal_thresh
        
        dist_1 = torch.linalg.norm(self.goal_1.pose.p - self.cube_1.pose.p, axis=1)
        success_1 = dist_1 <= self.goal_thresh
        
        is_robot0_static = self.agent.agents[0].is_static(0.2)
        is_robot1_static = self.agent.agents[1].is_static(0.2)
        
        return {
            "success": success_0 & success_1 & is_robot0_static & is_robot1_static,
            "success_0": success_0,
            "success_1": success_1,
        }
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = torch.zeros(self.num_envs, device=self.device)
        
        tcp0_to_cube0 = torch.linalg.norm(
            self.cube_0.pose.p - self.agent.agents[0].tcp_pose.p, axis=1
        )
        reward += 1 - torch.tanh(5 * tcp0_to_cube0)
        
        tcp1_to_cube1 = torch.linalg.norm(
            self.cube_1.pose.p - self.agent.agents[1].tcp_pose.p, axis=1
        )
        reward += 1 - torch.tanh(5 * tcp1_to_cube1)
        
        reward[info["success"]] = 10
        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10