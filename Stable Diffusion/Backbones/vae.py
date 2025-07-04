import torch
import torch.nn as nn
import torch.nn.functional as F

# --- VAE Building Blocks ---

class ResnetBlock(nn.Module):
    """A standard residual block with two convolutional layers."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, self.out_channels, 1) if in_channels != self.out_channels else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class AttnBlock(nn.Module):
    """A self-attention block to capture long-range dependencies."""
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.qkv(h_).chunk(3, dim=1)
        b, c, h, w = q.shape
        q, k, v = map(lambda t: t.view(b, c, h * w).transpose(1, 2), (q, k, v))
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = h_.transpose(1, 2).view(b, c, h, w)
        return x + self.proj_out(h_)

# --- VAE Encoder & Decoder ---

class Encoder(nn.Module):
    """The Encoder part of the VAE. Compresses an image into a latent representation."""
    def __init__(self, in_channels=3, z_channels=4, channels=128, ch_mults=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        in_ch = channels
        for i, mult in enumerate(ch_mults):
            out_ch = channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            if i != len(ch_mults) - 1:
                self.down_blocks.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
        
        self.middle_block = nn.Sequential(ResnetBlock(in_ch, in_ch), AttnBlock(in_ch), ResnetBlock(in_ch, in_ch))
        self.out_block = nn.Sequential(nn.GroupNorm(32, in_ch), nn.SiLU(), nn.Conv2d(in_ch, 2 * z_channels, 3, padding=1))

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.middle_block(x)
        return self.out_block(x)

class Decoder(nn.Module):
    """The Decoder part of the VAE. Reconstructs an image from a latent representation."""
    def __init__(self, out_channels=3, z_channels=4, channels=128, ch_mults=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        in_ch = channels * ch_mults[-1]
        self.conv_in = nn.Conv2d(z_channels, in_ch, 3, padding=1)
        self.middle_block = nn.Sequential(ResnetBlock(in_ch, in_ch), AttnBlock(in_ch), ResnetBlock(in_ch, in_ch))
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            if i != 0:
                self.up_blocks.append(nn.Upsample(scale_factor=2.0, mode='nearest'))
        self.out_block = nn.Sequential(nn.GroupNorm(32, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_channels, 3, padding=1))

    def forward(self, z):
        z = self.conv_in(z)
        z = self.middle_block(z)
        for block in self.up_blocks:
            z = block(z)
        return self.out_block(z)

# --- Main VAE Model ---

class DiagonalGaussianDistribution:
    """Helper class for the reparameterization trick."""
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

class AutoencoderKL(nn.Module):
    """The full VAE model, combining the Encoder and Decoder."""
    def __init__(self, in_channels=3, z_channels=4, channels=128, ch_mults=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        self.encoder = Encoder(in_channels, z_channels, channels, ch_mults, num_res_blocks)
        self.decoder = Decoder(in_channels, z_channels, channels, ch_mults, num_res_blocks)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Encodes an image and returns the distribution parameters."""
        latent_params = self.encoder(x)
        latent_params = self.quant_conv(latent_params)
        return DiagonalGaussianDistribution(latent_params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a latent vector z into an image."""
        z = self.post_quant_conv(z)
        return self.decoder(z)