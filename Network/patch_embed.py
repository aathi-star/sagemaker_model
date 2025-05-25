import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding layer"""
    def __init__(self, in_channels=1, embed_dim=512, patch_size=8, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """x: [batch, channels, depth, height, width]"""
        B, C, D, H, W = x.shape
        assert D % self.patch_size[0] == 0 and H % self.patch_size[1] == 0 and W % self.patch_size[2] == 0, \
            f"Input dimensions must be divisible by patch size. Got {x.shape} with patch size {self.patch_size}"
            
        x = self.proj(x)  # [B, C, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, C]
        x = self.norm(x)
        return x
