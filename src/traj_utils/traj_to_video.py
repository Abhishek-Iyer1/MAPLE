import h5py
import cv2
import numpy as np
from tqdm import tqdm

def h5_to_video(h5_path, output_video="output.mp4", fps=30):
    with h5py.File(h5_path, "r") as f:
        # Just pick the first trajectory as a test
        traj = f["traj_0"]
        # Navigate to your custom SideView camera observations
        # Usually: traj['obs']['sensor_data']['base_camera']['rgb']
        images = traj['obs']['sensor_data']['base_camera']['rgb'][:]
        
        height, width, _ = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for img in tqdm(images, desc="Writing Video"):
            # Convert RGB to BGR for OpenCV
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        video.release()
    print(f"Video saved to {output_video}")

if __name__ == "__main__":
    h5_to_video("/home/stryder/.maniskill/demos/PickCube-SideView-v1/motionplanning/trajectory.h5")