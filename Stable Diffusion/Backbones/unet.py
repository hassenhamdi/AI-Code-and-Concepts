import math
import torch
from torch import nn
from einops import rearrange

# --- Helper Functions and Layers ---

def exists(val):
    """Check if a value is not None."""
    return val is not None

def default(val, d):
    """Return the value if it exists, otherwise return a default."""
    return val if exists(val) else d() if callable(d) else d

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps (torch.Tensor): A 1-D tensor of N timesteps.
        dim (int): The dimension of the embedding.
    Returns:
        torch.Tensor: An (N, dim) tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def scaled_dot_product_attention(q, k, v, heads, mask=None):
    """
    Computes scaled dot-product attention, a core component of transformers.
    """
    dim_head = q.shape[-1] // heads
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))
    sim = torch.einsum('b i d, b j d -> b i j', q, k) * (dim_head ** -0.5)

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        sim = sim.masked_fill(~mask, -1e9)

    attn = sim.softmax(dim=-1)
    out = torch.einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
    return out

class GEGLU(nn.Module):
    """A variant of the gated linear unit activation function."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    """A standard feed-forward network for transformer blocks."""
    def __init__(self, dim, dim_out=None, mult=4, glu=True, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = GEGLU(dim, inner_dim) if glu else nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning."""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        out = scaled_dot_product_attention(q, k, v, self.heads, mask)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    """A transformer block with self-attention and cross-attention."""
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x # Self-Attention
        x = self.attn2(self.norm2(x), context=context) + x # Cross-Attention
        x = self.ff(self.norm3(x)) + x # Feed-Forward
        return x

class SpatialTransformer(nn.Module):
    """Transformer block for 2D feature maps."""
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

class Downsample(nn.Module):
    """Downsampling layer with a strided convolution."""
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channels or channels, kernel_size=3, stride=2, padding=1) if use_conv else nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    """Upsampling layer with interpolation and convolution."""
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class ResBlock(nn.Module):
    """A residual block with timestep conditioning."""
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=True, use_scale_shift_norm=False, down=False, up=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )
        self.updown = up or down
        self.h_upd, self.x_upd = (Upsample(channels, False), Upsample(channels, False)) if up else ((Downsample(channels, False), Downsample(channels, False)) if down else (nn.Identity(), nn.Identity()))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = nn.Conv2d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class TimestepEmbedSequential(nn.Sequential):
    """A sequential container that passes timestep embeddings and context to its children."""
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

# --- Main U-Net Model ---

class UNetModel(nn.Module):
    """The core U-Net architecture for noise prediction."""
    def __init__(
        self,
        in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=2, attention_resolutions=(4, 2, 1),
        dropout=0.0, channel_mult=(1, 2, 4, 4), num_heads=8,
        context_dim=768, use_scale_shift_norm=True
    ):
        super().__init__()
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))])
        input_block_chans = [model_channels]
        ch, ds = model_channels, 1

        # --- Encoder (Down-sampling path) ---
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, num_heads, ch // num_heads, context_dim=context_dim))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, use_conv=True)))
                input_block_chans.append(ch)
                ds *= 2

        # --- Middle Block ---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, num_heads, ch // num_heads, context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
        )

        # --- Decoder (Up-sampling path) ---
        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, use_scale_shift_norm=use_scale_shift_norm)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, num_heads, ch // num_heads, context_dim=context_dim))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # --- Output Layer ---
        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))

    def forward(self, x, timesteps, context=None):
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        saved_hidden_states = []
        h = x
        for module in self.input_blocks:
            h = module(h, t_emb, context)
            saved_hidden_states.append(h)
        h = self.middle_block(h, t_emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, saved_hidden_states.pop()], dim=1)
            h = module(h, t_emb, context)
        return self.out(h)