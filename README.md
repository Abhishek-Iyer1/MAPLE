# Diffusion Policy for ManiSkill

Train diffusion policies on ManiSkill robotic manipulation tasks.

## Setup
```bash
conda create -n dp_maniskill python=3.10 -y
conda activate dp_maniskill
pip install -r requirements.txt
git submodule update --init --recursive
```

## Download Demos
```bash
python -m mani_skill.utils.download_demo "StackCube-v1"
```

## Convert Demos
```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
    --save-traj -o rgbd -c pd_joint_delta_pos
```

## Train
```bash
python src/train_rgbd.py \
    --env_id StackCube-v1 \
    --demo_path <path_to_converted_demos> \
    --control_mode pd_joint_delta_pos \
    --max_episode_steps 200 \
    --total_iters 3000
```

## Evaluate
```bash
python src/eval_live.py --checkpoint runs/<run_name>/checkpoints/best_eval_success_once.pt
```
