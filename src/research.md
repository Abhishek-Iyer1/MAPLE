## Research Direction Summary
- Setting up a diffusion policy baseline for the task ```PickCube-v1``` for a single agent.
- Evaluating the performance of the single policy in a the same setting (in and out of training distribution)
- Evaluating the performance of the single policy in a multi-agent coordination setting.

## Current Status
- Diffusion policy baseline on PickCube-v1 is not training

## Next steps
- Firstly, run a training loop to train the diffusion policy on PickCube-v1 with no changes. Good litmus test
- Run the same training loop but with the observation space shifted to the wider FOV.

## Immediate Todo List (23rd-27th March)
- ~~Verify training data supervision is using the ```pd_joint_delta_pos``` controller.~~
- ~~If using RGBD observation space, verify if wrist cameras are part of the observation space~~ (No wrist cameras used)
- ~~Modify the data loader to mix the teleop + motionplanning trajectories (teleop trajectories should help)~~
- ~~Write W&B logging for important metrics so you can monitor remotely (including videos)~~
- ~~Train the single agent policy on PickCube-v1 (trained ee_delta, joint_delta, joint). ee_delta_pos far outperforms everything else ~~
- ~~Investigate (rgbd vs rgb seems to be weird) **Running experiments, inital thoughts seem like rgb traj file has something that is better than rgbd. Should play it back**~~
- ~~Generate ee_delta trajectories with side_view observations **Working on it**~~
- ~~Use it to train a robust baseline policy with side_view **Put on training**~~
- Load that policy in the environment and perform inference for a sample inside its training data to see how good the output is
    - Load the environment to be PickCube-SideView-v1 using gym.make
    - Get the observations from the camera and the 29 vector state that you need to give to the policy
    - Maintain an observation buffer and load that 
    - Generate a noise vector of the pred_horizon * R^n where n depends on the control strategy
    - Iterate over execution horizon and load the actions and take steps through the environment (remember to update observation buffer)
    - Loop until max_steps reached (200 or something?)
- Once this is done, load the same view and episode but with another (static) agent in the scene. Observe any changes in behaviour and record videos
- Now have the policy predict outputs for both agents in the scene and observe

## Long Term Goals
- Make a trajectory class so you don't need so many util scripts
- Think about detecting collisions and adding constraints where collisions occur

## Important points

### Data
- there are 1000 trajectories in /motionplanning and 10 in /teleop without any action space is ```pd_joint_pose```. There are some .h5 and .json files in /rl but no trajectory.h5 which is the original format.


## Gotchas
- ``pd_ee_delta_pos`` is of shape (4,) for (x, y, z, gripper width). NOTE this does not include rotation as the end effector is held in the same position. [Refer here ](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html) for more state space of different controllers

- Single env — one simulation instance:
  obs shape: (H, W, C)   # e.g. (128, 128, 3)
  action shape: (4,)

- Vector env — N parallel simulation instances batched together (ManiSkill's default, even with num_envs=1):
  obs shape: (N, H, W, C)   # e.g. (1, 128, 128, 3)
  action shape: (N, 4)
  This is what gym.make(..., num_envs=1) gives you. The single_observation_space attribute describes the space for one env in the batch (without the N dimension).

- "Stacked" env (FrameStack-style) — wraps a vector env so observations are stacked over the last T timesteps:
  obs shape: (N, T, H, W, C)   # e.g. (1, 2, 128, 128, 3)
  This is what Agent assumes it's working with — it wants the obs space to already include the time dimension so it knows state_dim = shape[1].

 Two hard blockers when passing --env_id PickCube-SideView-TwoRobot-v1:
 1. Dict action space: env.step expects {'panda-0': tensor, 'panda-1': tensor}, not flat tensor
 2. obs["agent"] is nested: obs["agent"] = {'panda-0': {qpos,qvel}, 'panda-1': {...}} instead of
 flat {qpos, qvel}, which breaks FlattenRGBDObservationWrapper and FrameStack init

Root cause: TableSceneBuilder.initialize() in ManiSkill checks env.robot_uids at episode
 reset time and unconditionally overrides robot poses. For any ("panda", "panda") env it sets:

 agent.agents[0].robot.set_pose(sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi/2)))
 agent.agents[1].robot.set_pose(sapien.Pose([0,  0.75, 0], q=euler2quat(0, 0, -np.pi/2)))

 But training used the single-robot branch which sets:
 self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))   # no rotation

 So despite _load_agent placing robot 0 correctly at [-0.615, 0, 0], every call to
 env.reset() moves it to [0, -0.75, 0] with 90° rotation — completely changing what
 the side-view camera sees.


## Running Experiments
1) [rgbd in path, ee_delta_pos controller, obs mode]
2) [rgbd in path, ee_delta_pos controller, obs mode]
3) [rgb in path, ee_delta_pos controller, obs mode rgb]
4) [rgb in path, ee_delta_pos controller, obs mode rgbd]