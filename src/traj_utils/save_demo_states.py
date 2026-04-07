 # src/save_demo_states.py
"""Save initial states from demo trajectories for inference testing."""

import os
import sys
import json
import h5py
import numpy as np

def save_demo_states(demo_path, output_path=None, num_traj=None):
    demo_path = os.path.expanduser(demo_path)
    
    if output_path is None:
        output_path = demo_path.replace('.h5', '_states.json')
    
    states = {}
    
    with h5py.File(demo_path, 'r') as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith('traj')],
            key=lambda x: int(x.split('_')[1])
        )
        
        if num_traj:
            traj_keys = traj_keys[:num_traj]
        
        for traj_key in traj_keys:
            traj = f[traj_key]
            
            # Get initial state (timestep 0)
            states[traj_key] = {
                'cube_pose': traj['env_states/actors/cube'][0].tolist(),
                'goal_pose': traj['env_states/actors/goal_site'][0].tolist(),
                'robot_state': traj['env_states/articulations/panda'][0].tolist(),
                'goal_pos': traj['obs/extra/goal_pos'][0].tolist(),
                'num_steps': traj['actions'].shape[0],
                'success': bool(traj['success'][()]) if traj['success'].shape == () else bool(traj['success'][-1]),
            }
    
    with open(output_path, 'w') as f:
        json.dump(states, f, indent=2)
    
    print(f"Saved {len(states)} trajectory states to: {output_path}")
    
    # Print summary
    print(f"\nExample (traj_0):")
    print(f"  Cube pos: [{states['traj_0']['cube_pose'][0]:.3f}, {states['traj_0']['cube_pose'][1]:.3f}, {states['traj_0']['cube_pose'][2]:.3f}]")
    print(f"  Goal pos: [{states['traj_0']['goal_pos'][0]:.3f}, {states['traj_0']['goal_pos'][1]:.3f}, {states['traj_0']['goal_pos'][2]:.3f}]")
    print(f"  Num steps: {states['traj_0']['num_steps']}")
    print(f"  Success: {states['traj_0']['success']}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_traj", type=int, default=None)
    args = parser.parse_args()
    
    save_demo_states(args.demo_path, args.output, args.num_traj)