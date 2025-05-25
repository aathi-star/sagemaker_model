# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# Import TCGA module for enhanced long-range dependencies
from Network.tcga import TCGATransformerEncoder


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # swinGLU
        h_dim = int(2 * h_dim / 3)
        # make sure it is a multiple of multiple_of
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        # Ensure input is float
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
            
        # SwiGLU activation
        x = self.act(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)

        return self.w2(x)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash # use flash attention?
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None):
        b, h_w, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # normalize queries and keys
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
        xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
        xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)

        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
        if self.flash:
            if mask is not None:
                mask = mask.view(b, 1, 1, h_w)
            output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
        # output projection
        proj = self.wo(output)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class AdaNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.norm_final = RMSNorm(x_dim, linear=True, bias=True, eps=1e-5)
        self.act = nn.SiLU()
        self.linear = nn.Linear(y_dim, x_dim * 2)
        
    def forward(self, x, y):
        # Ensure inputs are float
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
        if y.dtype != torch.float32 and y.dtype != torch.float16:
            y = y.float()
            
        # Process through MLP
        y = self.linear(self.act(y))
        shift, scale = y.chunk(2, dim=1)
        
        # Apply normalization and modulation
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = Attention(dim, heads, dropout=dropout)

        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond, mask=None):
        # Ensure x is float
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
            
        # Ensure cond is float for MLP
        if cond is not None and cond.dtype != torch.float32 and cond.dtype != torch.float16:
            cond = cond.float()

        # extract modulation parameters
        cond_params = self.mlp(cond)  # [batch_size, dim*6]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = cond_params.chunk(6, dim=1)
        
        # first norm
        normed_x = self.ln1(x)
        # modulate
        normed_x = modulate(normed_x, beta1, gamma1)
        # attention
        x = x + alpha1.unsqueeze(1) * self.attn(normed_x, mask)
        # second norm
        normed_x = self.ln2(x)
        # modulate
        normed_x = modulate(normed_x, beta2, gamma2)
        # feedforward
        x = x + alpha2.unsqueeze(1) * self.ff(normed_x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None):
        for block in self.layers:
            x = block(x, cond, mask=mask)
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1, use_tcga=False, tcga_ratio=0.5, **kwargs):
        super().__init__()

        self.nclass = nclass                                             # Number of classes
        self.input_size = input_size                                     # Number of tokens as input
        self.hidden_dim = hidden_dim                                     # Hidden dimension of the transformer
        self.codebook_size = codebook_size                               # Amount of code in the codebook
        self.proj = proj                                                 # Projection
        self.use_tcga = use_tcga                                         # Whether to use TCGA for long-range dependencies

        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)              # Embedding layer for the class token
        self.class_embed = self.cls_emb                                  # Alias for use in cls_trainer
        self.null_class_feature = nn.Parameter(torch.zeros(1, hidden_dim)) # Learnable null embedding for CFG

        self.tok_emb = nn.Embedding(codebook_size + 1, hidden_dim)       # Embedding layer for the 'visual' token
        self.pos_emb = nn.Embedding(input_size ** 2, hidden_dim)         # Learnable Positional Embedding

        if self.proj > 1:
            self.in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim*4, kernel_size=1, stride=1, padding=0, bias=False
            ).to(memory_format=torch.channels_last)

        # The Transformer Encoder - use TCGA enhanced version if specified
        if use_tcga:
            self.transformer = TCGATransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, 
                                                     mlp_dim=mlp_dim, dropout=dropout, tcga_ratio=tcga_ratio)
        else:
            self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, 
                                                 mlp_dim=mlp_dim, dropout=dropout)

        # Add projection for condition if needed
        self.proj_cond = None
        if hasattr(self, 'cls_emb'):
            self.proj_cond = nn.Linear(self.cls_emb.weight.size(1), hidden_dim)
        
        self.last_norm = AdaNorm(x_dim=hidden_dim, y_dim=hidden_dim)   # Last Norm

        self.head = nn.Linear(hidden_dim, codebook_size + 1)
        self.head.weight = self.tok_emb.weight  # weight tied with the tok_emb layer

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()  # Init weight

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.cls_emb.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Zero-out adaNorm modulation layers in blocks:
        for block in self.transformer.layers:
            # Handle both standard blocks and TCGAEnhancedBlocks
            if hasattr(block, 'mlp'):
                # For standard blocks
                if len(block.mlp) > 1 and hasattr(block.mlp[1], 'weight'):
                    nn.init.constant_(block.mlp[1].weight, 0)
                    if hasattr(block.mlp[1], 'bias'):
                        nn.init.constant_(block.mlp[1].bias, 0)
            # For TCGAEnhancedBlocks, we don't need to zero out weights as they're handled in the block

        # Init proj layer
        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

        # Init embedding
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def forward(self, x, y_context, mask=None): 
        # Ensure inputs are in the correct format
        if x.dtype != torch.long:
            x = x.long()
            
        b, h, w = x.size()
        x = x.reshape(b, h*w)

        # Prepare y_context (class conditioning)
        if y_context is None:
            # If no y_context is provided, use a zero vector
            y_context = torch.zeros(b, self.hidden_dim, device=x.device)
        elif y_context.dim() == 1:
            # If y_context is class indices, convert to embeddings
            y_context = self.cls_emb(y_context.to(x.device))
        
        # Ensure y_context is float and has the right shape [batch_size, hidden_dim]
        if y_context.dim() > 2:
            y_context = y_context.view(b, -1)  # Flatten if needed
        if y_context.size(-1) != self.hidden_dim:
            # If y_context has wrong dimension, project it
            y_context = y_context.to(x.device).float()
            if hasattr(self, 'proj_cond') and self.proj_cond is not None:
                y_context = self.proj_cond(y_context)
            else:
                y_context = y_context[:, :self.hidden_dim]  # Truncate or pad
        else:
            y_context = y_context.to(x.device).float()

        pos = torch.arange(0, w*h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)

        # Get token embeddings and add positional encoding
        x = self.tok_emb(x) + pos

        # reshape, proj to smaller space, reshape (patchify!)
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape
            x = rearrange(x, 'b c h proj_w -> b (h proj_w) c', proj_h=h, proj_w=w, b=b, c=self.hidden_dim).contiguous()

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            reg_tokens = self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)
            x = torch.cat([x, reg_tokens], dim=1)

        # Ensure x is float before transformer
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
            
        x = self.transformer(x, y_context, mask=mask) 

        # drop the register
        x = x[:, :h*w].contiguous()

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.out_proj(x)
            x = rearrange(x, 'b (c s1 s2) h w -> b (h s1 w s2) c', s1=self.proj, s2=self.proj, b=b, h=h, w=w, c=self.hidden_dim).contiguous()

        x = self.last_norm(x, y_context) 
        logit = self.head(x)

        return logit


if __name__ == "__main__":
    from thop import profile

    # for size in ["tiny", "small", "base", "large", "xlarge"]:
    size = "base"
    print(size)
    if size == "tiny":
        hidden_dim, depth, heads = 384, 6, 6
    elif size == "small":
        hidden_dim, depth, heads = 512, 8, 6
    elif size == "base":
        hidden_dim, depth, heads = 768, 12, 12
    elif size == "large":
        hidden_dim, depth, heads = 1024, 24, 16
    elif size == "xlarge":
        hidden_dim, depth, heads = 1152, 28, 16
    else:
        hidden_dim, depth, heads = 768, 12, 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 32
    model = Transformer(input_size=input_size, nclass=1000, hidden_dim=hidden_dim, codebook_size=16834,
                        depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=0.1).to(device)
    # model = torch.compile(model)
    code = torch.randint(0, 16384, size=(1, input_size, input_size)).to(device)
    cls = torch.randint(0, 1000, size=(1,)).to(device)
    d_label = (torch.rand(1) < 0.1).to(device)
    attn_mask = torch.cat([
        torch.rand(1, (input_size//2)**2).to(device) > 1,
        torch.tensor([[True]], dtype=torch.bool, device=device)
    ], dim=1)
    flops, params = profile(model, inputs=(code, cls, d_label))
    print(f"FLOPs: {flops//1e9:.2f}G, Params: {params/1e6:.2f}M")
