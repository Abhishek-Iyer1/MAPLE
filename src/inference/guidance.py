"""
Guidance cost functions and EE trajectory utilities for diffusion policy inference.

Typical usage in dual_main():

    center = torch.tensor([0.1, -0.3, 0.4], device=device)
    tcp_left = [None]   # mutable cell updated before each get_action() call

    def guidance_fn_left(norm_actions):
        positions = integrate_ee_deltas(norm_actions, tcp_left[0], agent_left)
        return obstacle_cost(positions, center, radius=0.1)

    # before each chunk:
    tcp_left[0] = torch.from_numpy(buf_left[-1]["state"][_TCP_XYZ_SLICE]).float().to(device)
    act_left = agent_left.get_action(obs, guidance_fn=guidance_fn_left, guidance_scale=1.0)
"""

import torch
import torch.nn.functional as F

# State vector layout inside each obs dict built by _extract_robot_obs():
#   qpos(9) | qvel(9) | is_grasped(1) | tcp_pose(7) | goal_pos(3)  = 29 dims
# tcp_pose = xyz(3) + quaternion_xyzw(4), so tcp_xyz lives at indices 19:22.
_TCP_XYZ_SLICE = slice(19, 22)


def integrate_ee_deltas(
    norm_actions: torch.Tensor,
    tcp_xyz: torch.Tensor,
    agent,
) -> torch.Tensor:
    """
    Convert a normalised EE-delta action chunk to absolute world-frame xyz positions.

    The control mode is ``pd_ee_delta_pos``: the first 3 action dimensions are
    (dx, dy, dz) deltas expressed in the world/root frame.  Integration is a
    simple cumulative sum starting from the current TCP position.

    Args:
        norm_actions: (B, T, act_dim) normalised actions in [-1, 1].
        tcp_xyz:      (3,) or (B, 3) current TCP xyz in world frame.
        agent:        Agent instance — supplies ``action_low`` / ``action_high``
                      buffers needed for denormalisation.

    Returns:
        (B, T, 3) absolute xyz positions for each timestep in the chunk.
    """
    # Denormalise to actual delta space (e.g. [-0.1, 0.1] m per axis)
    deltas = agent.denormalize_action(norm_actions)   # (B, T, act_dim)

    # Only the first 3 dims are xyz position deltas; gripper dim is ignored
    delta_xyz = deltas[..., :3]                       # (B, T, 3)

    # Cumulative sum of deltas → relative displacement at each timestep
    cumsum = torch.cumsum(delta_xyz, dim=1)            # (B, T, 3)

    # Broadcast tcp_xyz so addition works for any input shape
    if tcp_xyz.dim() == 1:
        tcp_xyz = tcp_xyz.unsqueeze(0).unsqueeze(0)   # (1, 1, 3)
    elif tcp_xyz.dim() == 2:
        tcp_xyz = tcp_xyz.unsqueeze(1)                # (B, 1, 3)

    return tcp_xyz + cumsum                            # (B, T, 3)


def obstacle_cost(
    positions: torch.Tensor,
    center: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """
    Sphere obstacle cost: penalises positions that fall inside the sphere.

    C = sum( 0.5 * relu(r - ||t - c||_2)² )

    The 0.5 factor is chosen so the gradient simplifies cleanly:
        dC/dt = relu(r - dist) * -(t - c) / dist

    Gradient magnitude = relu(r - dist), which is:
      - 0 at and outside the boundary (no jerk when entering/leaving)
      - grows linearly toward the centre (stronger push the deeper inside)

    Args:
        positions: (..., 3) tensor of xyz coordinates in the robot world frame.
        center:    (3,) tensor — sphere centre in the same frame.
        radius:    sphere radius in metres.

    Returns:
        Scalar cost tensor (differentiable w.r.t. positions).
    """
    dist = torch.norm(positions - center, dim=-1).clamp(min=1e-6)   # (...,)
    return (0.5 * F.relu(radius - dist) ** 2).sum()
