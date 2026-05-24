import torch
import torch.nn as nn
import timm
import sys
import os

# Add diffusion_policy to path
sys.path.append("diffusion_policy")

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

class VisionEncoder(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        dummy_input = torch.randn(1, 4, 3, 224, 224) # 4 is dummy batch size
        with torch.no_grad():
            # timm models expect B,C,H,W
            # We want to infer the feature dim
            feat = self.backbone(torch.randn(1, 3, 224, 224))
            feat_dim = feat.shape[-1]
            
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        feat = self.norm(feat)
        return feat

class DiffusionPolicyGCBC(nn.Module):
    def __init__(self, 
                 vision_encoder, 
                 action_dim=7, 
                 obs_dim=512, # Embedding dim from vision encoder
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024]):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.action_dim = action_dim
        
        # Observation (Start) + Goal = 2 * obs_dim
        global_cond_dim = obs_dim * 2
        
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8
        )

    def get_latents(self, obs_img, goal_img):
        start_feat = self.vision_encoder(obs_img)
        goal_feat = self.vision_encoder(goal_img)
        # Concatenate start and goal features for global conditioning
        return torch.cat([start_feat, goal_feat], dim=-1)

    def forward(self, sample, timestep, global_cond=None):
        """
        sample: (B, Horizon, Action_Dim)
        timestep: (B,)
        global_cond: (B, Global_Cond_Dim) created by get_latents
        """
        # ConditionalUnet1D forward expects:
        # sample: (B, T, input_dim) -> (B, Horizon, Action_Dim)
        # global_cond: (B, global_cond_dim)
        return self.unet(sample, timestep, global_cond=global_cond)
