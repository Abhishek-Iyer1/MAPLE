# src/replay_trajectory_custom.py
"""Replay trajectories with custom environments registered."""

import sys
import os

# Add src to path so we can import our custom envs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom environments BEFORE importing ManiSkill replay
from src.envs.pick_cube_side import PickCubeSideViewEnv, PickCubeSideViewEnvLeft, PickCubeSideViewEnvRight


# Now import and run ManiSkill's replay_trajectory
from mani_skill.trajectory.replay_trajectory import main, parse_args

if __name__ == "__main__":
    main(parse_args())