"""3D Transformer for 3D-aware generation with TCGA"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat

from .tcga import TCGATransformerEncoder
from .positional_encoding import PositionalEncoding3D
from .patch_embed import PatchEmbed3D

class MaskGIT3D(nn.Module):
    """3D Masked Generative Transformer with TCGA and ULIP-2 text conditioning"""
    def __init__(
        self,
        in_channels=1,
        patch_size=(4, 4, 4),
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.,
        num_classes=1000, # Output dim for reconstruction, not classification classes
        dropout=0.1,
        tcga_ratio=0.5,
        text_encoder=None,
        text_dim=768,
        masking_strategy='random', # New argument: 'random' or 'halton'
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=nn.LayerNorm
        )
        
        # Class token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding3D(embed_dim)
        
        # For point clouds, dynamically created positional embeddings
        self.point_pos_embed_dict = nn.ParameterDict()
        
        # Text conditioning
        self.text_encoder = text_encoder
        self.text_proj = nn.Linear(text_dim, embed_dim) if text_encoder is not None else None
        self.text_embed = nn.Parameter(torch.zeros(1, 1, embed_dim)) if text_encoder is not None else None
        
        # Masking strategy and sampler
        self.masking_strategy = masking_strategy
        if self.masking_strategy == 'halton':
            self.halton_sampler = Halton3DSampler(dim=1) # For 1D token sequence masking

        # Initialize text projection if needed
        if self.text_encoder is not None:
            nn.init.normal_(self.text_embed, std=0.02)
            nn.init.xavier_uniform_(self.text_proj.weight)
            nn.init.constant_(self.text_proj.bias, 0)
        
        # Transformer encoder with TCGA
        self.encoder = TCGATransformerEncoder(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            tcga_ratio=tcga_ratio,
            num_classes=num_classes
        )
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mask_ratio=0.0, return_features=False, text=None, return_text_emb=False):
        """
        Args:
            x: Input tensor of shape (B, N, 3) for point clouds or (B, C, D, H, W) for voxels
            mask_ratio: Ratio of tokens to mask
            return_features: If True, return intermediate features
            text: Optional list of text strings for conditioning
            return_text_emb: If True, return text embeddings along with output
        """
        # Handle both point cloud data (B, N, 3) and voxel data (B, C, D, H, W)
        if len(x.shape) == 3:  # Point cloud data: (B, N, 3)
            B, N, C = x.shape
            # Use the PointNet-like encoder to transform points to features
            x = x.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
            # Create simple PointNet encoder on first use
            if not hasattr(self, 'point_encoder'):
                self.point_encoder = nn.Sequential(
                    nn.Conv1d(3, 64, 1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, self.embed_dim, 1),
                ).to(x.device)
            
            # Encode points to feature space: (B, 3, N) -> (B, embed_dim, N)
            x = self.point_encoder(x)
            # Reshape for transformer: (B, embed_dim, N) -> (B, N, embed_dim)
            x = x.transpose(1, 2)
        else:  # Voxel data: (B, C, D, H, W)
            B, C, D, H, W = x.shape
        
        # Encode text if provided
        text_emb = None
        if self.text_encoder is not None and text is not None:
            print(f"DEBUG_TRANSFORMER: Inside MaskGIT3D.forward, about to call self.text_encoder(text)") # DEBUG
            print(f"DEBUG_TRANSFORMER: type(self.text_encoder): {type(self.text_encoder)}") # DEBUG
            print(f"DEBUG_TRANSFORMER: text is not None: {text is not None}") # DEBUG
            if text is not None:
                print(f"DEBUG_TRANSFORMER: type(text): {type(text)}, text.shape: {text.shape if hasattr(text, 'shape') else 'N/A'}") # DEBUG
            
            text_emb = self.text_encoder(text) # This is line 120 in the traceback
            
            print(f"DEBUG_TRANSFORMER: self.text_encoder(text) call completed.") # DEBUG

            text_emb = self.text_proj(text_emb)  # (B, text_dim) -> (B, embed_dim)
            text_emb = text_emb.unsqueeze(1)  # (B, 1, embed_dim)
            text_emb = text_emb + self.text_embed  # Add learned text embedding
        
        # Process differently based on input type
        if len(x.shape) == 3 and x.shape[2] == self.embed_dim:  # Already point cloud features
            # Skip patch embedding for point cloud data - already processed above
            # Add class token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Add text token if using text conditioning
            if text_emb is not None:
                # x was [CLS, P1...PN], after this it's [CLS, TEXT_EMB, P1...PN]
                # Note: x[:, 1:] correctly takes P1...PN from the original x cat with CLS.
                x = torch.cat((x[:, 0:1], text_emb, x[:, 1:]), dim=1)
            
            # For point clouds, use 1D positional embeddings
            has_text = text_emb is not None
            # N is the number of points from x.shape before cls/text tokens were added.
            # This N was captured when x was (B,N,C) for point clouds.
            # If x was voxel data, N would be different (num_patches).
            # Assuming N here refers to the original number of point cloud tokens.
            # The sequence length of x is now: 1 (cls) + (1 if has_text else 0) + N (points)
            num_tokens_in_x = x.size(1)

            # The key for pos_embed should reflect the actual structure it's built for.
            # Original N (number of points) is a good part of the key.
            # Whether text is present also changes the required length.
            pos_embed_key = (N, has_text) # N is num_points
            str_pos_embed_key = str(pos_embed_key) # Keys for ParameterDict must be strings
            
            if str_pos_embed_key not in self.point_pos_embed_dict:
                # Total tokens for pos embedding: CLS + (TEXT if present) + N_points
                num_tokens_for_pos_embed = 1 + (1 if has_text else 0) + N
                print(f"DEBUG_TRANSFORMER: Creating point_pos_embed for key {str_pos_embed_key}, num_tokens: {num_tokens_for_pos_embed}")
                # Create nn.Parameter with tensor directly on x.device
                new_param = nn.Parameter(
                    torch.zeros(1, num_tokens_for_pos_embed, self.embed_dim, device=x.device)
                )
                nn.init.trunc_normal_(new_param, std=0.02)
                self.point_pos_embed_dict[str_pos_embed_key] = new_param
                
            final_pos_embed = self.point_pos_embed_dict[str_pos_embed_key]
            
            # Add positional encoding for point cloud data
            if x.size(1) != final_pos_embed.size(1):
                raise ValueError(
                    f"Runtime mismatch: x seq len {x.size(1)} vs pos_embed seq len {final_pos_embed.size(1)}. "
                    f"N_points={N}, has_text={has_text}. This indicates a logic error in token construction or pos_embed keying."
                )
            x = x + final_pos_embed
        else:  # Voxel data
            # Patch embedding
            x = self.patch_embed(x)  # (B, L, D)
            
            # Add class token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Add text token if using text conditioning
            if text_emb is not None:
                x = torch.cat((cls_tokens, text_emb, x[:, 1:]), dim=1)
            
            # Add positional encoding for voxel data
            x = self.pos_embed(x, (D, H, W))
        
        # Apply transformer - note TCGATransformerEncoder takes (x, cond, mask) not mask_ratio
        # Create condition tensor from class index (default to 0 if not available)
        cond = torch.zeros(B, dtype=torch.long, device=x.device)
        
        # Create mask for the encoder based on mask_ratio and strategy
        encoder_mask = None # This will be (B, S) boolean mask, True means masked (src_key_padding_mask)
        if mask_ratio > 0 and x.size(1) > 1: # Ensure there are tokens to mask
            # B, S, E = x.shape # B and S already available from input x processing
            
            # Determine number of prefix tokens (CLS, optional TEXT)
            # This calculation needs to be consistent with how x is constructed earlier
            num_prefix_tokens = 1 # CLS token
            if self.text_encoder is not None and text_emb is not None: # Check if text_emb was actually created and used
                num_prefix_tokens += 1
            
            # Content tokens are those after the prefix
            num_content_tokens = x.size(1) - num_prefix_tokens
            
            if num_content_tokens > 0:
                num_to_mask = int(mask_ratio * num_content_tokens)
                if num_to_mask > 0:
                    
                    if self.masking_strategy == 'halton':
                        if not hasattr(self, 'halton_sampler'): # Should have been init in __init__
                            # Fallback, though ideally __init__ handles this based on strategy
                            self.halton_sampler = Halton3DSampler(dim=1)
                        # Generate Halton sequence for content tokens
                        halton_points = self.halton_sampler.sample(num_content_tokens, device=x.device).squeeze(-1)
                        # Get indices that would sort the Halton points (deterministic order)
                        sorted_indices = torch.argsort(halton_points)
                        # These are relative indices within the content tokens (0 to num_content_tokens-1)
                        relative_indices_to_mask = sorted_indices[:num_to_mask]
                    else: # Default to 'random'
                        perm = torch.randperm(num_content_tokens, device=x.device)
                        relative_indices_to_mask = perm[:num_to_mask]
                    
                    # Convert relative indices to absolute indices in the full sequence
                    absolute_indices_to_mask = relative_indices_to_mask + num_prefix_tokens
                    
                    # Create boolean mask (B, S) - True means masked for src_key_padding_mask
                    encoder_mask = torch.zeros(B, x.size(1), dtype=torch.bool, device=x.device)
                    # Scatter True to the mask positions for each batch item. Indices are same for all batch items.
                    encoder_mask[:, absolute_indices_to_mask] = True
            
        # Pass to encoder with correct parameters
        x = self.encoder(x, cond, mask=encoder_mask) # 'mask' is src_key_padding_mask
        
        # Store input type for decoding later
        self.input_type = 'point_cloud' if len(x.shape) == 3 and x.shape[2] == self.embed_dim else 'voxel'
        
        # Get features from transformer
        features = x
        
        # For point cloud reconstruction, we need a decoder
        if self.input_type == 'point_cloud':
            # Create point cloud decoder on first use
            if not hasattr(self, 'point_decoder'):
                self.point_decoder = nn.Sequential(
                    nn.Linear(self.embed_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, 3)  # Output (x,y,z) coordinates
                ).to(x.device)
            
            # Decode features (excluding class token) back to point coordinates
            if return_features:
                decoded = features  # Return transformer features directly
            else:
                point_features = features[:, 1:]  # Skip the class token
                decoded = self.point_decoder(point_features)  # (B, N, 3)
        else:  # Voxel data - standard processing
            decoded = features if return_features else features[:, 0]  # Just use class token for voxels
        
        # Return with or without text embeddings
        if return_text_emb and text_emb is not None:
            return (decoded, text_emb)
        return decoded
    
    def _init_3d_layers(self):
        """Initialize 3D-specific layers"""
        # Add any 3D-specific layers here
        pass
    
    def get_num_patches(self, input_shape):
        """Get number of patches for a given input shape"""
        D, H, W = input_shape
        return (D // self.patch_size[0]) * (H // self.patch_size[1]) * (W // self.patch_size[2])
    
    @torch.no_grad()
    def generate(self, num_samples=1, temperature=1.0, top_k=100, top_p=0.9, device='cuda', 
                text=None, use_halton=True):
        """Generate samples using nucleus sampling with optional Halton sequence ordering
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            device: Device to run generation on
            text: Optional list of text prompts for conditional generation
            use_halton: Whether to use Halton sequence for sampling order
            
        Returns:
            Generated token sequences of shape (B, seq_len)
        """
        self.eval()
        
        # Get spatial dimensions (assuming cubic input)
        grid_size = int(round((self.get_num_patches((32, 32, 32))) ** (1/3)))
        num_patches = grid_size ** 3
        
        # Initialize with masked tokens
        tokens = torch.full((num_samples, num_patches), -1, device=device, dtype=torch.long)
        
        # Generate sampling order using Halton sequence if enabled
        if use_halton:
            halton = Halton3DSampler(dim=3)
            points = halton.sample(num_patches, device=device)  # (seq_len, 3) in [0,1]^3
            
            # Scale to grid coordinates
            grid_points = (points * grid_size).long()
            
            # Convert 3D grid coordinates to flat indices
            sampling_order = (grid_points[:, 0] * grid_size * grid_size + 
                           grid_points[:, 1] * grid_size + 
                           grid_points[:, 2])
        else:
            # Default to raster scan order if Halton is disabled
            sampling_order = torch.arange(num_patches, device=device)
        
        # Text conditioning if provided
        text_emb = None
        if text is not None and self.text_encoder is not None:
            with torch.no_grad():
                text_emb = self.text_encoder(text)  # (B, text_dim)
                text_emb = self.text_proj(text_emb)  # (B, embed_dim)
                text_emb = text_emb.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Generation loop
        for i in sampling_order:
            # Get model predictions for all samples at current position
            with torch.no_grad():
                # Create input with current tokens and mask
                input_tokens = tokens.clone()
                input_tokens[input_tokens == -1] = 0  # Replace masked tokens with 0
                
                # Forward pass with current tokens
                logits = self.forward(input_tokens.unsqueeze(1), text=text_emb)  # [B, seq_len, vocab_size]
                
                # Get logits for current position
                logits = logits[:, i, :]  # [B, vocab_size]
                
                # Apply temperature and filtering
                logits = logits / temperature
                logits = self._top_k_top_p_filtering(logits.unsqueeze(1), top_k=top_k, top_p=top_p).squeeze(1)
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Update tokens at current position
                tokens[:, i] = next_token
        
        return tokens
    
    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted indices back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
            
        return logits
    
class Halton3DSampler:
    """3D Halton sequence for efficient 3D point sampling"""
    def __init__(self, dim=3):
        self.dim = dim
        self.primes = [2, 3, 5][:dim]  # First n primes for n dimensions
    
    def sample(self, n_points, device='cuda'):
        """Generate Halton sequence in 3D"""
        points = torch.zeros(n_points, self.dim, device=device)
        for i in range(self.dim):
            points[:, i] = self._van_der_corput(n_points, self.primes[i])
        return points
    
    def _van_der_corput(self, n, base):
        """Van der Corput sequence"""
        sequence = []
        for i in range(n):
            n, d = i + 1, 1.0
            x = 0.0
            while n > 0:
                d /= base
                x += (n % base) * d
                n = n // base
            sequence.append(x)
        return torch.tensor(sequence, device='cuda' if torch.cuda.is_available() else 'cpu')
