import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from einops import rearrange


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )  # Remove FC and AvgPool layers

    def forward(self, x):
        """
        Args:
            x: Images of shape (B, num_cameras, C, H, W)
        Returns:
            Features of shape (B, num_cameras, C_feat, H_feat, W_feat)
        """
        B, num_cameras, C, H, W = x.shape
        x = rearrange(
            x, "B num_cams C H W -> (B num_cams) C H W"
        )  # Merge batch and camera dims
        features = self.backbone(x)  # Extract features
        C_feat, H_feat, W_feat = features.shape[1:]
        features = rearrange(
            features,
            "(B num_cams) C H W -> B num_cams C H W",
            B=B,
            num_cams=num_cameras,
        )
        return features


class BEVFormer(nn.Module):
    def __init__(
        self, bev_height=30, bev_width=30, bev_dim=256, num_heads=8, num_layers=6
    ):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.bev_dim = bev_dim

        # Image backbone
        self.image_backbone = ImageBackbone()

        # BEV query embeddings
        self.bev_queries = nn.Parameter(torch.randn(bev_height * bev_width, bev_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bev_dim, nhead=num_heads, dim_feedforward=1024, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head (e.g., for object detection)
        self.cls_head = nn.Linear(bev_dim, 8)  # Example: 10 classes
        self.reg_head = nn.Linear(
            bev_dim, 7
        )  # Example: 7 box parameters (x, y, z, w, l, h, yaw)

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images: Tensor of shape (B, num_cameras, C, H, W)
            intrinsics: Tensor of shape (B, num_cameras, 3, 3)
            extrinsics: Tensor of shape (B, num_cameras, 4, 4)
        Returns:
            BEV features and predictions
        """
        B = images.size(0)

        # Step 1: Extract image features
        img_features = self.image_backbone(
            images
        )  # Shape: (B, num_cameras, C_feat, H_feat, W_feat)

        # Step 2: Flatten and project image features (Cross-attention can be added here)
        img_features = rearrange(img_features, "B num_cams C H W -> B (num_cams H W) C")

        # Step 3: Repeat BEV queries for batch
        bev_queries = self.bev_queries.unsqueeze(0).repeat(
            B, 1, 1
        )  # Shape: (B, bev_height * bev_width, bev_dim)

        # Step 4: Transformer Encoder
        bev_features = self.transformer_encoder(
            bev_queries
        )  # Shape: (B, bev_height * bev_width, bev_dim)

        # Step 5: Decode outputs
        cls_preds = self.cls_head(bev_features)  # Classification predictions
        reg_preds = self.reg_head(bev_features)  # Regression predictions

        # Reshape BEV features for downstream tasks
        bev_features = bev_features.view(
            B, self.bev_height, self.bev_width, -1
        )  # Shape: (B, H_bev, W_bev, bev_dim)

        return bev_features, cls_preds, reg_preds
