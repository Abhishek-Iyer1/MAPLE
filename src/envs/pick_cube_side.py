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
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

ROBOT_POSE_LEFT = sapien.Pose(p=[0, -0.7, 0], q=[-0.7071068, 0, 0, -0.7071068])
ROBOT_POSE_RIGHT = sapien.Pose(p=[0, 0.7, 0], q=[0.7071068, 0, 0, -0.7071068])
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
        
        # --- NEW: Transform Cube and Goal ---
        orig_base = Pose.create_from_pq(
            p=torch.tensor([-0.615, 0.0, 0.0], device=self.device, dtype=torch.float32),
            q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        )
        new_base = Pose.create_from_pq(
            p=torch.tensor(ROBOT_POSE_RIGHT.p, device=self.device, dtype=torch.float32),
            q=torch.tensor(ROBOT_POSE_RIGHT.q, device=self.device, dtype=torch.float32)
        )

        # 1. Map world poses to original robot's local frame
        # 2. Map local poses back out to the world using the new robot's frame
        cube_local = orig_base.inv() * self.cube.pose
        self.cube.set_pose(new_base * cube_local)

        goal_local = orig_base.inv() * self.goal_site.pose
        self.goal_site.set_pose(new_base * goal_local)


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

        if "cube_p" in options:
            # Explicit poses provided (e.g. in-domain eval) — use them directly
            self.cube.set_pose(Pose.create_from_pq(
                p=torch.tensor(options["cube_p"], device=self.device, dtype=torch.float32),
                q=torch.tensor(options["cube_q"], device=self.device, dtype=torch.float32)
            ))
            self.goal_site.set_pose(Pose.create_from_pq(
                p=torch.tensor(options["goal_p"], device=self.device, dtype=torch.float32),
                q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            ))
        else:
            # Transform randomised poses from default robot frame → left robot frame
            orig_base = Pose.create_from_pq(
                p=torch.tensor([-0.615, 0.0, 0.0], device=self.device, dtype=torch.float32),
                q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            )
            new_base = Pose.create_from_pq(
                p=torch.tensor(ROBOT_POSE_LEFT.p, device=self.device, dtype=torch.float32),
                q=torch.tensor(ROBOT_POSE_LEFT.q, device=self.device, dtype=torch.float32)
            )
            cube_local = orig_base.inv() * self.cube.pose
            self.cube.set_pose(new_base * cube_local)
            goal_local = orig_base.inv() * self.goal_site.pose
            self.goal_site.set_pose(new_base * goal_local)


@register_env("PickCube-SideView-DualTask-v1", max_episode_steps=100)
class PickCubeSideViewDualTaskEnv(PickCubeSideViewTwoRobotEnv):
    """
    Two-robot side-view env with independent tasks per robot.
    robot 0 (left, y=-0.9): manages self.cube + self.goal_site
    robot 1 (right, y=+0.9): manages self.cube_1 + self.goal_site_1
    Cube/goal positions are symmetric about y=0.
    """

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cube_1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube_1",
            initial_pose=sapien.Pose(p=[0, 0.3, 0.02]),
        )
        self.goal_site_1 = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 1, 1],
            name="goal_site_1",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site_1)

    def _initialize_episode(self, env_idx, options):
        # super() handles: PickCubeEnv randomises self.cube + self.goal_site,
        # then PickCubeSideViewTwoRobotEnv restores both robot poses.
        super()._initialize_episode(env_idx, options)

        orig_base = Pose.create_from_pq(
            p=torch.tensor([-0.615, 0.0, 0.0], device=self.device, dtype=torch.float32),
            q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32),
        )
        left_base = Pose.create_from_pq(
            p=torch.tensor(ROBOT_POSE_LEFT.p, device=self.device, dtype=torch.float32),
            q=torch.tensor(ROBOT_POSE_LEFT.q, device=self.device, dtype=torch.float32),
        )
        right_base = Pose.create_from_pq(
            p=torch.tensor(ROBOT_POSE_RIGHT.p, device=self.device, dtype=torch.float32),
            q=torch.tensor(ROBOT_POSE_RIGHT.q, device=self.device, dtype=torch.float32),
        )

        cube_local = orig_base.inv() * self.cube.pose
        goal_local = orig_base.inv() * self.goal_site.pose

        # Left robot gets cube/goal transformed into its frame
        self.cube.set_pose(left_base * cube_local)
        self.goal_site.set_pose(left_base * goal_local)
        # Right robot gets the symmetric version
        self.cube_1.set_pose(right_base * cube_local)
        self.goal_site_1.set_pose(right_base * goal_local)

    def evaluate(self):
        agent0 = self.agent.agents[0]
        agent1 = self.agent.agents[1]
        is_obj_placed_0 = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1) <= self.goal_thresh
        )
        is_obj_placed_1 = (
            torch.linalg.norm(self.goal_site_1.pose.p - self.cube_1.pose.p, axis=1) <= self.goal_thresh
        )
        is_grasped_0 = agent0.is_grasping(self.cube)
        is_grasped_1 = agent1.is_grasping(self.cube_1)
        is_robot_static_0 = agent0.is_static(0.2)
        is_robot_static_1 = agent1.is_static(0.2)
        return {
            "success": is_obj_placed_0 & is_obj_placed_1 & is_robot_static_0 & is_robot_static_1,
            "is_obj_placed_0": is_obj_placed_0,
            "is_obj_placed_1": is_obj_placed_1,
            "is_grasped_0": is_grasped_0,
            "is_grasped_1": is_grasped_1,
            "is_robot_static_0": is_robot_static_0,
            "is_robot_static_1": is_robot_static_1,
        }

    def _get_obs_extra(self, info: dict):
        agent0 = self.agent.agents[0]
        agent1 = self.agent.agents[1]
        return dict(
            is_grasped_0=info["is_grasped_0"],
            tcp_pose_0=agent0.tcp_pose.raw_pose,
            goal_pos_0=self.goal_site.pose.p,
            is_grasped_1=info["is_grasped_1"],
            tcp_pose_1=agent1.tcp_pose.raw_pose,
            goal_pos_1=self.goal_site_1.pose.p,
        )

    def compute_dense_reward(self, obs, action, info):
        agent0 = self.agent.agents[0]
        agent1 = self.agent.agents[1]

        def _reach_place_reward(agent, cube, goal_site, is_grasped, is_obj_placed):
            tcp_to_obj = torch.linalg.norm(cube.pose.p - agent.tcp_pose.p, axis=1)
            reward = 1 - torch.tanh(5 * tcp_to_obj)
            reward += is_grasped
            obj_to_goal = torch.linalg.norm(goal_site.pose.p - cube.pose.p, axis=1)
            reward += (1 - torch.tanh(5 * obj_to_goal)) * is_grasped
            qvel = agent.robot.get_qvel()[..., :-2]
            reward += (1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))) * is_obj_placed
            return reward

        r0 = _reach_place_reward(agent0, self.cube,   self.goal_site,   info["is_grasped_0"], info["is_obj_placed_0"])
        r1 = _reach_place_reward(agent1, self.cube_1, self.goal_site_1, info["is_grasped_1"], info["is_obj_placed_1"])
        reward = r0 + r1
        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
