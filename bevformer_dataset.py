import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image
from pyquaternion import Quaternion

from PIL import Image
import numpy as np

# Initialize nuScenes dataset


import os
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image
from pyquaternion import Quaternion
from PIL import Image
import numpy as np


class BEVFormerDataset(Dataset):
    def __init__(
        self,
        nusc,
        data_root,
        bev_height=50,
        bev_width=50,
        bev_resolution=0.5,
        bev_offset=(-25, -25),
        transform=None,
    ):
        """
        Args:
            nusc: nuScenes dataset object.
            data_root: Root directory of the nuScenes dataset.
            bev_height: Height of the BEV grid.
            bev_width: Width of the BEV grid.
            bev_resolution: Size of each BEV grid cell in meters.
            bev_offset: Offset of the BEV grid (x_min, y_min) in meters.
            transform: Optional image transformations (e.g., resizing, normalization).
        """
        self.nusc = nusc
        self.data_root = data_root
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.bev_resolution = bev_resolution
        self.bev_offset = bev_offset
        self.transform = transform
        self.samples = nusc.sample  # List of all samples in the dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}

        # Load camera data
        images = []
        intrinsics = []
        extrinsics = []
        for cam_name in [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]:
            cam_token = sample["data"][cam_name]
            cam_data = self.nusc.get("sample_data", cam_token)
            cam_cs = self.nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )
            ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

            # Load image
            img_path = os.path.join(self.data_root, cam_data["filename"])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

            # Camera intrinsics and extrinsics
            intrinsic = np.array(cam_cs["camera_intrinsic"])
            intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))

            cam_to_ego = np.eye(4)
            cam_to_ego[:3, :3] = Quaternion(cam_cs["rotation"]).rotation_matrix
            cam_to_ego[:3, 3] = cam_cs["translation"]

            ego_to_global = np.eye(4)
            ego_to_global[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
            ego_to_global[:3, 3] = ego_pose["translation"]

            cam_to_global = np.dot(ego_to_global, cam_to_ego)
            extrinsics.append(torch.tensor(cam_to_global, dtype=torch.float32))

        data["images"] = torch.stack(images)  # (num_cameras, C, H, W)
        data["intrinsics"] = torch.stack(intrinsics)  # (num_cameras, 3, 3)
        data["extrinsics"] = torch.stack(extrinsics)  # (num_cameras, 4, 4)

        # Initialize classification and regression targets
        cls_target = torch.zeros(
            (self.bev_height, self.bev_width), dtype=torch.long
        )  # Default to background (0)
        reg_target = torch.zeros(
            (self.bev_height, self.bev_width, 7), dtype=torch.float32
        )  # Default regression values
        reg_target[:, :, :] = float("nan")  # Mark empty cells as NaN

        # Populate targets
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            x, y, z = ann["translation"]  # Object position in global coordinates
            w, l, h = ann["size"]  # Object size
            yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[
                0
            ]  # Object yaw angle (rotation around Z-axis)
            category = ann["category_name"]  # Object category

            # Map global coordinates to BEV grid
            grid_x = int((x - self.bev_offset[0]) / self.bev_resolution)
            grid_y = int((y - self.bev_offset[1]) / self.bev_resolution)

            # Assign targets if within BEV grid bounds
            if 0 <= grid_x < self.bev_width and 0 <= grid_y < self.bev_height:
                cls_target[grid_y, grid_x] = self.get_class_index(
                    category
                )  # Classification target
                reg_target[grid_y, grid_x] = torch.tensor(
                    [x, y, z, w, l, h, yaw], dtype=torch.float32
                )  # Regression target

        data["cls_target"] = cls_target  # Shape: (bev_height, bev_width)
        data["reg_target"] = reg_target  # Shape: (bev_height, bev_width, 7)

        return data

    def get_class_index(self, category_name):
        """
        Map category name to a class index (implement as needed).
        """
        category_map = {
            "vehicle.car": 1,
            "vehicle.truck": 2,
            "vehicle.bus": 3,
            "vehicle.motorcycle": 4,
            "vehicle.bicycle": 5,
            "pedestrian.adult": 6,
            "pedestrian.child": 7,
        }
        return category_map.get(category_name, 0)  # Default to background (0)


# from torchvision.transforms import Compose, Resize, ToTensor

# # Initialize nuScenes dataset
# nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)

# # Define image transformations
# transform = Compose(
#     [
#         Resize((224, 224)),  # Resize images to a fixed size
#         ToTensor(),  # Convert to PyTorch tensors
#     ]
# )

# # Initialize dataset
# dataset = BEVFormerDataset(nusc, data_root="data/nuscenes", transform=transform)

# # DataLoader
# from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Iterate through the data
# for batch in dataloader:
#     images = batch["images"]  # Shape: (batch_size, num_cameras, C, H, W)
#     intrinsics = batch["intrinsics"]  # Shape: (batch_size, num_cameras, 3, 3)
#     extrinsics = batch["extrinsics"]  # Shape: (batch_size, num_cameras, 4, 4)
#     boxes = batch["boxes"]  # List of bounding boxes per sample
#     print("Batch processed")
#     break
