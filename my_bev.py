from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from bevformer_dataset import BEVFormerDataset
from nuscenes.nuscenes import NuScenes
from torchvision.transforms import Compose, Resize, ToTensor
from bevformer import BEVFormer
import torch.nn.functional as F
import torch


def bevformer_collate_fn(batch):
    """
    Custom collate function for the BEVFormer dataset.

    Args:
        batch: List of samples, where each sample is a dictionary with keys:
            - "images": Tensor of shape (num_cameras, C, H, W)
            - "intrinsics": Tensor of shape (num_cameras, 3, 3)
            - "extrinsics": Tensor of shape (num_cameras, 4, 4)
            - "boxes": List of bounding box dictionaries (variable length)

    Returns:
        dict: Collated batch with keys:
            - "images": Tensor of shape (B, num_cameras, C, H, W)
            - "intrinsics": Tensor of shape (B, num_cameras, 3, 3)
            - "extrinsics": Tensor of shape (B, num_cameras, 4, 4)
            - "boxes": List of lists of bounding box dictionaries
    """
    # Combine fixed-size elements into tensors
    images = torch.stack(
        [sample["images"] for sample in batch]
    )  # (B, num_cameras, C, H, W)
    intrinsics = torch.stack(
        [sample["intrinsics"] for sample in batch]
    )  # (B, num_cameras, 3, 3)
    extrinsics = torch.stack(
        [sample["extrinsics"] for sample in batch]
    )  # (B, num_cameras, 4, 4)

    cls_target = torch.stack([sample["cls_target"] for sample in batch])
    reg_target = torch.stack([sample["reg_target"] for sample in batch])

    return {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "cls_target": cls_target,
        "reg_target": reg_target,
    }


# Initialize nuScenes dataset
nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)

# Define image transformations
transform = Compose(
    [
        Resize((224, 224)),  # Resize images to a fixed size
        ToTensor(),  # Convert to PyTorch tensors
    ]
)

# Initialize dataset
dataset = BEVFormerDataset(nusc, data_root="data/nuscenes", transform=transform)

# DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset, collate_fn=bevformer_collate_fn, batch_size=2, shuffle=True
)

model = BEVFormer(bev_height=50, bev_width=50, bev_dim=256, num_heads=8, num_layers=6)
model = model.cuda()  # Move to GPU if available

optimizer = Adam(model.parameters(), lr=1e-4)


for epoch in range(10):
    for batch in dataloader:
        images = batch["images"].cuda()  # Shape: (B, num_cameras, C, H, W)
        intrinsics = batch["intrinsics"].cuda()  # Shape: (B, num_cameras, 3, 3)
        extrinsics = batch["extrinsics"].cuda()  # Shape: (B, num_cameras, 4, 4)

        # Forward pass
        bev_features, cls_preds, reg_preds = model(images, intrinsics, extrinsics)

        # Compute loss (implement your own loss function)
        cls_loss = F.cross_entropy(
            cls_preds.view(-1, 50, 50, 8).permute(0, 3, 1, 2),
            batch["cls_target"].cuda(),
        )
        reg_loss = F.mse_loss(reg_preds.view(-1, 50, 50, 7), batch["reg_target"].cuda())
        loss = cls_loss + reg_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
