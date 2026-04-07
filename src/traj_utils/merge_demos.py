# src/merge_demos.py
"""
Merge multiple demo H5 files into one for training.
Assumes all input files have been converted with the same obs_mode and control_mode.
"""

import h5py
import os
import json
import argparse


def merge_demo_files(input_paths, output_path):
    """Merge multiple demo H5 files into one."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    traj_count = 0
    source_info = []  # Track where trajectories came from
    
    with h5py.File(output_path, 'w') as out_f:
        for input_path in input_paths:
            print(f"\nProcessing: {input_path}")
            
            if not os.path.exists(input_path):
                print(f"  WARNING: File not found, skipping!")
                continue
            
            with h5py.File(input_path, 'r') as in_f:
                traj_keys = sorted(
                    [k for k in in_f.keys() if k.startswith('traj')],
                    key=lambda x: int(x.split('_')[1])  # Sort numerically
                )
                print(f"  Found {len(traj_keys)} trajectories")
                
                start_idx = traj_count
                for old_key in traj_keys:
                    new_key = f"traj_{traj_count}"
                    
                    # Copy entire trajectory group
                    in_f.copy(old_key, out_f, name=new_key)
                    traj_count += 1
                
                source_info.append({
                    "source": input_path,
                    "num_trajectories": len(traj_keys),
                    "traj_range": [start_idx, traj_count - 1]
                })
    
    print(f"\n{'='*60}")
    print(f"Total trajectories: {traj_count}")
    print(f"Saved to: {output_path}")
    
    return traj_count, source_info


def create_merged_json(output_json_path, source_info, env_id, control_mode):
    """Create a JSON metadata file for the merged demos."""
    
    total_trajs = sum(s["num_trajectories"] for s in source_info)
    
    metadata = {
        "env_info": {
            "env_id": env_id,
            "env_kwargs": {
                "control_mode": control_mode,
                "obs_mode": "rgbd"
            }
        },
        "source_files": source_info,
        "total_trajectories": total_trajs,
        "episodes": [
            {"episode_id": i, "control_mode": control_mode}
            for i in range(total_trajs)
        ]
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created JSON: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple demo H5 files")
    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="Input H5 files to merge"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output H5 file path"
    )
    parser.add_argument(
        "--env_id",
        default="PickCube-v1",
        help="Environment ID"
    )
    parser.add_argument(
        "--control_mode",
        default="pd_joint_delta_pos",
        help="Control mode used in demos"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Merging Demo Files")
    print("="*60)
    print(f"Inputs: {args.inputs}")
    print(f"Output: {args.output}")
    print(f"Env ID: {args.env_id}")
    print(f"Control Mode: {args.control_mode}")
    
    # Merge H5 files
    traj_count, source_info = merge_demo_files(args.inputs, args.output)
    
    if traj_count == 0:
        print("ERROR: No trajectories merged!")
        return
    
    # Create JSON metadata
    json_path = args.output.replace('.h5', '.json')
    create_merged_json(json_path, source_info, args.env_id, args.control_mode)
    
    print(f"\n{'='*60}")
    print("DONE! Use for training:")
    print(f"  --demo_path {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()