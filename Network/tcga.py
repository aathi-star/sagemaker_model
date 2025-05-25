# Token-Centric Graph Attention (TCGA) module for Halton-MaskGIT
# This module addresses the long-range dependency limitation in the original Halton-MaskGIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class HaltonEdgeSampler:
    """
    Samples edges for the graph attention using Halton sequences for efficiency.
    This ensures a quasi-random distribution of edges that covers the space well.
    """
    @staticmethod
    def halton_sequence(dim, n_sample):
        """Generate a Halton sequence for edge sampling."""
        if n_sample == 0:
            return torch.empty((dim, 0), dtype=torch.float32) # Return tensor of shape (dim, 0)

        def halton(b, n):
            h, d = 0, 1
            seq = []
            for i in range(n):
                x = d - h
                if x == 1:
                    h = 1
                    d *= b
                else:
                    y = d // b
                    while x <= y:
                        y //= b
                    h = (b + 1) * y - x
                seq.append(h / d)
            return seq
        
        # Generate sequences for each dimension
        sequences = []
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(dim):
            sequences.append(torch.tensor(halton(primes[i % len(primes)], n_sample)))
        
        return torch.stack(sequences, dim=1)
    
    @staticmethod
    def sample_edges(num_tokens, num_edges, device):
        """Sample edges using Halton sequence for better coverage."""
        if num_edges == 0 or num_tokens < 2: # Need at least 2 tokens to form an edge
            return torch.empty((0,), dtype=torch.long, device=device), \
                   torch.empty((0,), dtype=torch.long, device=device)

        # Generate Halton sequence
        # Ensure n_sample for halton_sequence is not negative or zero if we proceed
        halton_points = HaltonEdgeSampler.halton_sequence(2, num_edges) 
        
        # Scale to token indices
        source_indices = (halton_points[:, 0] * num_tokens).long()
        target_indices = (halton_points[:, 1] * num_tokens).long()
        
        # Ensure no self-loops
        mask = source_indices != target_indices
        source_indices = source_indices[mask]
        target_indices = target_indices[mask]
        
        # If we lost some edges due to self-loop filtering, add more
        if len(source_indices) < num_edges:
            additional = num_edges - len(source_indices)
            extra_sources = torch.randint(0, num_tokens, (additional,))
            extra_targets = torch.randint(0, num_tokens, (additional,))
            mask = extra_sources != extra_targets
            source_indices = torch.cat([source_indices, extra_sources[mask]])
            target_indices = torch.cat([target_indices, extra_targets[mask]])
            
            # If we still don't have enough, just take what we have
            if len(source_indices) < num_edges:
                num_edges = len(source_indices)
        
        # Take only what we need
        source_indices = source_indices[:num_edges]
        target_indices = target_indices[:num_edges]
        
        return source_indices.to(device), target_indices.to(device)


class TokenCentricGraphAttention(nn.Module):
    """
    Token-Centric Graph Attention (TCGA) module that enables long-range dependencies
    in the transformer by connecting distant tokens via learned adjacency matrices.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, edge_dropout=0.5, num_edges_ratio=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.edge_dropout = edge_dropout
        self.num_edges_ratio = num_edges_ratio  # Ratio of edges to total possible edges
        
        # Projections for graph attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Learnable adjacency parameters
        self.edge_weight = nn.Parameter(torch.randn(1, num_heads, 1))
        
        # Edge-wise attention
        self.edge_attn = nn.Sequential(
            nn.Linear(2 * self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Memory-efficient implementation
        # For long sequences, cap the number of edges to prevent memory issues
        # Start with a reasonable number based on sequence length
        num_edges_to_sample = min(1000, int(seq_len * 2))  # Very conservative estimate
        
        # Further limit for memory constraints (critical for CPU training)
        if torch.cuda.is_available():
            # GPU has more memory, can use more edges
            max_edges = min(2000, int(0.05 * seq_len * seq_len))
        else:
            # CPU mode - extremely conservative with memory
            max_edges = min(500, int(0.01 * seq_len * seq_len))
            
        num_edges_to_sample = min(num_edges_to_sample, max_edges)
        
        src_indices, dst_indices = HaltonEdgeSampler.sample_edges(
            seq_len, 
            max_edges,
            x.device
        )
        num_edges_filtered = src_indices.shape[0]

        # print(f"TCGA DEBUG (L129): seq_len={seq_len}, num_edges_to_sample={num_edges_to_sample}", flush=True)
        # print(f"TCGA DEBUG (L130): src_indices.shape: {src_indices.shape}", flush=True)
        # print(f"TCGA DEBUG (L131): dst_indices.shape: {dst_indices.shape}", flush=True)
        # print(f"TCGA DEBUG (L132): num_edges_filtered={num_edges_filtered}", flush=True)
        # print(f"TCGA DEBUG (L133): src_indices[:5]: {src_indices[:5] if src_indices.numel() > 0 else 'N/A'}", flush=True)
        # print(f"TCGA DEBUG (L134): dst_indices[:5]: {dst_indices[:5] if dst_indices.numel() > 0 else 'N/A'}", flush=True)

        if num_edges_filtered == 0:
            # print("TCGA DEBUG (L137): Exiting early due to num_edges_filtered == 0", flush=True)
            return self.dropout(self.o_proj(torch.zeros_like(x)))
        
        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # print(f"TCGA DEBUG (L151) before q_src = q[:,:,src_indices]:", flush=True)
        # print(f"  q.shape: {q.shape}", flush=True)
        # print(f"  src_indices.shape: {src_indices.shape}", flush=True)
        # print(f"  max(src_indices) if valid: {torch.max(src_indices) if src_indices.numel() > 0 else 'N/A'}", flush=True)
        # print(f"  min(src_indices) if valid: {torch.min(src_indices) if src_indices.numel() > 0 else 'N/A'}", flush=True)

        q_src = q[:, :, src_indices]  # Potential error source
        k_dst = k[:, :, dst_indices]  # Potential error source
        
        edge_features = torch.cat([q_src, k_dst], dim=-1)
        edge_features = edge_features.permute(0, 2, 1, 3)
        edge_scores = self.edge_attn(edge_features).permute(0, 2, 1, 3)

        # print(f"TCGA DEBUG (L165) before F.softmax(edge_scores...):", flush=True)
        # print(f"  edge_scores.shape: {edge_scores.shape}", flush=True)
        
        edge_attn_pre_softmax = edge_scores * self.scale
        edge_attn = F.softmax(edge_attn_pre_softmax, dim=2)

        # print(f"TCGA DEBUG (L171) before 'edge_attn = edge_attn * self.edge_weight' (reported error line):", flush=True)
        # print(f"  edge_attn.shape: {edge_attn.shape}", flush=True)
        # print(f"  self.edge_weight.shape: {self.edge_weight.shape}", flush=True)
        # print(f"  edge_attn has NaNs: {torch.isnan(edge_attn).any() if edge_attn.numel() > 0 else 'N/A (empty)'}", flush=True)

        edge_attn = edge_attn * self.edge_weight.unsqueeze(3) # Make broadcasting explicit
        
        # Apply edge dropout during training
        if self.training and self.edge_dropout > 0:
            edge_mask = torch.rand(edge_attn.shape, device=edge_attn.device) > self.edge_dropout
            edge_attn = edge_attn * edge_mask.float()
        
        # Compute weighted values for each edge
        v_dst = v[:, :, dst_indices]  # [batch, heads, num_edges_filtered, head_dim]
        weighted_values = edge_attn * v_dst # [batch, heads, num_edges_filtered, head_dim]
        
        # Memory-efficient approach - process in chunks to avoid large allocation
        # Create an empty tensor for the result
        attn_output = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=x.device)
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000  # Smaller chunk size to avoid memory issues
        num_chunks = (num_edges_filtered + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            # Get chunk indices
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_edges_filtered)
            
            # Get chunk of source indices and weighted values
            src_chunk = src_indices[start_idx:end_idx]
            weighted_chunk = weighted_values[:, :, start_idx:end_idx, :]
            
            # Process this chunk
            for b in range(batch_size):
                for h in range(self.num_heads):
                    for j in range(end_idx - start_idx):
                        src_idx = src_chunk[j].item()
                        attn_output[b, h, src_idx] += weighted_chunk[b, h, j]
        
        # Use the processed result
        scatter_target = attn_output
        
        # Reshape and project output
        final_output = scatter_target.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        final_output = self.o_proj(final_output)
        final_output = self.dropout(final_output)
        
        return final_output


class TCGAEnhancedBlock(nn.Module):
    """
    Enhanced transformer block that combines standard self-attention with 
    Token-Centric Graph Attention for improved long-range dependencies.
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0., tcga_ratio=0.5, num_classes=1000):
        super().__init__()
        
        self.tcga_ratio = tcga_ratio  # Balance between standard attention and TCGA
        
        # Class embedding for conditioning
        self.class_embed = nn.Embedding(num_classes + 1, dim)  # +1 for null class
        
        # Projection for modulation parameters
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 6)
        )
        
        # Projection for feature dimension matching
        self.proj = nn.Linear(dim, dim) if dim != dim else nn.Identity()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        
        # TCGA module
        self.tcga = TokenCentricGraphAttention(dim, num_heads=heads, dropout=dropout)
        
        # Second layer norm and feed-forward
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, cond, mask=None):
        # Ensure x is float
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
            
        # Prepare condition
        if cond is not None:
            # Handle different input dimensions
            if cond.dim() > 2:
                cond = cond.view(cond.size(0), -1)
            
            # Get class embeddings if input is class indices
            if cond.dtype == torch.long:
                cond_emb = self.class_embed(cond)  # [batch_size, dim]
            else:
                cond_emb = cond  # Assume already embedded
                
            # Project to match hidden dim if needed
            cond_emb = self.proj(cond_emb)
            
            # Get modulation parameters
            cond_params = self.mlp(cond_emb)  # [batch_size, dim*6]
            
            # Split and reshape parameters
            params = cond_params.chunk(6, dim=1)  # 6 x [batch_size, dim]
            params = [p.unsqueeze(1) for p in params]  # 6 x [batch_size, 1, dim]
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params
        else:
            # If no condition, use zeros
            zeros = torch.zeros_like(x[:, :1, :])  # [batch_size, 1, dim]
            gamma1 = beta1 = alpha1 = gamma2 = beta2 = alpha2 = zeros
        
        # First normalization and modulation
        normed_x = self.ln1(x)  # [batch_size, seq_len, dim]
        modulated_x = normed_x * (1 + gamma1) + beta1  # Broadcasts [batch_size, 1, dim] to match [batch_size, seq_len, dim]
        
        # Standard self-attention
        attn_output, _ = self.attn(
            modulated_x, 
            modulated_x, 
            modulated_x,
            key_padding_mask=mask if mask is not None else None
        )
        
        # TCGA attention
        tcga_output = self.tcga(modulated_x, mask)
        
        # Combine standard attention and TCGA with the ratio
        combined_attn = (1 - self.tcga_ratio) * attn_output + self.tcga_ratio * tcga_output
        
        # First residual connection with modulation
        x = x + alpha1 * combined_attn
        
        # Second normalization and feed-forward
        normed_x = self.ln2(x)  # [batch_size, seq_len, dim]
        
        # Second normalization and feed-forward with modulation
        modulated_x = normed_x * (1 + gamma2) + beta2
        ff_output = self.ff(modulated_x)
        
        # Second residual connection with modulation
        x = x + alpha2 * ff_output
        
        return x


class TCGATransformerEncoder(nn.Module):
    """
    Enhanced transformer encoder that uses TCGAEnhancedBlock instead of standard Block.
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., tcga_ratio=0.5, num_classes=1000):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TCGAEnhancedBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                tcga_ratio=tcga_ratio,
                num_classes=num_classes
            ))

    def forward(self, x, cond, mask=None):
        # Ensure x is float
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
            
        # Process through each block
        for block in self.layers:
            x = block(x, cond, mask=mask)
        return x
