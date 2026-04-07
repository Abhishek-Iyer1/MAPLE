# src/fix_env_id.py
import json
import os

json_path = os.path.expanduser("~/.maniskill/demos/PickCube-SideView-v1/motionplanning/trajectory_source.json")

with open(json_path, 'r') as f:
    meta = json.load(f)

# Change env_id to your custom environment
meta['env_info']['env_id'] = 'PickCube-SideView-v1'

with open(json_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"Changed env_id to PickCube-SideView-v1")