import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from einops import rearrange
import torch.nn.functional as F

class AttentionVisualizer:
    """
    Tool for visualizing and comparing attention patterns between standard attention
    and Token-Centric Graph Attention (TCGA).
    """
    def __init__(self, save_dir="./attention_maps"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def register_hooks(self, model):
        """
        Register hooks to capture attention maps from both standard attention and TCGA.
        """
        self.standard_attention_maps = []
        self.tcga_attention_maps = []
        
        # Register hook for standard attention
        def hook_standard_attn(module, input, output):
            # Get attention weights from standard attention
            # Shape typically (batch_size, num_heads, seq_len, seq_len)
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                self.standard_attention_maps.append(module.attn_weights.detach().cpu())
        
        # Register hook for TCGA
        def hook_tcga_attn(module, input, output):
            # Get attention weights from TCGA
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                self.tcga_attention_maps.append(module.attn_weights.detach().cpu())
        
        # Attach hooks to all attention layers
        for name, module in model.named_modules():
            if 'attn' in name and 'tcga' not in name:
                module.register_forward_hook(hook_standard_attn)
            elif 'tcga' in name:
                module.register_forward_hook(hook_tcga_attn)
                
    def reset_hooks(self):
        """
        Clear stored attention maps between forward passes.
        """
        self.standard_attention_maps = []
        self.tcga_attention_maps = []
        
    def visualize_attention_maps(self, img_tokens=None, token_positions=None, layer_idx=0, head_idx=0, 
                                iteration=0, show=False):
        """
        Generate visualization comparing standard attention and TCGA patterns.
        
        Args:
            img_tokens: Optional token representation to show alongside attention
            token_positions: Optional specific token positions to visualize
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize
            iteration: Current training iteration (for filename)
            show: Whether to display the plot (vs just saving)
        """
        if not self.standard_attention_maps or not self.tcga_attention_maps:
            print("No attention maps captured. Run forward pass with hooks first.")
            return
            
        # Get attention maps for the specified layer
        if layer_idx < len(self.standard_attention_maps) and layer_idx < len(self.tcga_attention_maps):
            std_attn = self.standard_attention_maps[layer_idx][0, head_idx].numpy()  # First batch item
            tcga_attn = self.tcga_attention_maps[layer_idx][0, head_idx].numpy()
        else:
            print(f"Layer index {layer_idx} out of bounds")
            return
            
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot standard attention
        im0 = axes[0].imshow(std_attn, cmap='viridis')
        axes[0].set_title(f'Standard Attention (Layer {layer_idx}, Head {head_idx})')
        axes[0].set_xlabel('Token Position (Target)')
        axes[0].set_ylabel('Token Position (Source)')
        
        # Plot TCGA attention
        im1 = axes[1].imshow(tcga_attn, cmap='viridis')
        axes[1].set_title(f'TCGA (Layer {layer_idx}, Head {head_idx})')
        axes[1].set_xlabel('Token Position (Target)')
        axes[1].set_ylabel('Token Position (Source)')
        
        # Add colorbars
        plt.colorbar(im0, ax=axes[0])
        plt.colorbar(im1, ax=axes[1])
        
        # Highlight specific tokens if provided
        if token_positions is not None:
            for pos in token_positions:
                for ax in axes:
                    # Highlight both rows and columns for the token positions
                    ax.axhline(y=pos, color='red', linestyle='--', alpha=0.5)
                    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
        
        # Add analysis metrics
        std_entropy = self.calculate_entropy(std_attn)
        tcga_entropy = self.calculate_entropy(tcga_attn)
        sparsity_std = self.calculate_sparsity(std_attn)
        sparsity_tcga = self.calculate_sparsity(tcga_attn)
        
        plt.figtext(0.5, 0.01, 
                   f"Entropy: Standard={std_entropy:.2f}, TCGA={tcga_entropy:.2f} | "
                   f"Sparsity: Standard={sparsity_std:.2f}, TCGA={sparsity_tcga:.2f}", 
                   ha="center", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        filename = os.path.join(self.save_dir, f"attention_map_iter{iteration}_layer{layer_idx}_head{head_idx}.png")
        plt.savefig(filename, dpi=150)
        print(f"Saved attention map to {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def calculate_entropy(attention_map):
        """
        Calculate entropy of attention distribution as a measure of focus.
        Lower entropy means more focused attention.
        """
        # Normalize if not already normalized
        if not np.isclose(attention_map.sum(axis=-1), 1.0).all():
            attention_map = attention_map / (attention_map.sum(axis=-1, keepdims=True) + 1e-8)
            
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -np.sum(attention_map * np.log(attention_map + eps), axis=-1).mean()
        return entropy
    
    @staticmethod
    def calculate_sparsity(attention_map, threshold=0.01):
        """
        Calculate sparsity of attention map (percentage of values below threshold).
        Higher sparsity means more focused attention on specific tokens.
        """
        total_elements = attention_map.size
        sparse_elements = np.sum(attention_map < threshold)
        return sparse_elements / total_elements
    
    def visualize_attention_over_image(self, image, tokens, attention_map, token_size=16, 
                                      save_path=None, title="Attention Heatmap"):
        """
        Visualize attention overlaid on the original image.
        
        Args:
            image: Original image tensor [3, H, W]
            tokens: Token grid [H//patch_size, W//patch_size]
            attention_map: Attention weights for a specific token
            token_size: Size of each token in the original image
            save_path: Path to save the visualization
            title: Title for the plot
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
            # Normalize to [0, 1] for display
            image = (image - image.min()) / (image.max() - image.min())
            
        if torch.is_tensor(attention_map):
            attention_map = attention_map.cpu().numpy()
        
        # Resize attention map to match token grid
        h, w = tokens.shape
        attention_map = attention_map.reshape(h, w)
        
        # Upsample attention map to image size
        attention_heatmap = F.interpolate(
            torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
            size=(image.shape[0], image.shape[1]),
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot attention heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(attention_heatmap, cmap='hot')
        plt.title("Attention Heatmap")
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(attention_heatmap, cmap='hot', alpha=0.5)
        plt.title("Attention Overlay")
        plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
