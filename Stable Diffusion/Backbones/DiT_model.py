import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, labels):
        return self.dropout(self.embedding_table(labels))

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm (adaLN) conditioning."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_output
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x

class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class DiTModel(nn.Module):
    """Diffusion model with a Transformer backbone."""
    def __init__(self, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = in_channels * 2

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, 0.1)

        num_patches = (256 // patch_size) ** 2 # Assuming 256x256 latent
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def forward(self, x, t, y):
        x = self.x_embedder(x).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, T, D)
        x = x + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        
        # Unpatchify: (N, T, patch_size**2 * C_out) -> (N, C_out, H, W)
        h, w = x.shape[1] ** 0.5, x.shape[1] ** 0.5
        h, w = int(h), int(w)
        x = x.reshape(x.shape[0], h, w, self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.shape[0], self.out_channels, h * self.patch_size, w * self.patch_size)
        return x