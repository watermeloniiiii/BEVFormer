from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)
scene = nusc.scene[0]
first_sample_token = scene["first_sample_token"]
my_sample = nusc.get("sample", first_sample_token)
sensor = "LIDAR_TOP"
cam_front_data = nusc.get("sample_data", my_sample["data"][sensor])
import numpy as np
import os
import sys


def read_bin_file(file_path):
    # Load binary data as a flat array of floats
    data = np.fromfile(file_path, dtype=np.float32)

    # Reshape into N x 4 (x, y, z, intensity)
    points = data.reshape(-1, 4)
    return points


# Read the binary file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "nuscenes", cam_front_data["filename"]
)
point_cloud = read_bin_file(file_path)

# Inspect the data
print(f"Point cloud shape: {point_cloud.shape}")
print(point_cloud[:5])  # Print first 5 points
