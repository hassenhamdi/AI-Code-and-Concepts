import torch
import torch.nn as nn
from copy import deepcopy

class ZeroConv2d(nn.Module):
    """A 2D convolutional layer initialized with all zeros."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)

class ControlNet(nn.Module):
    """
    The ControlNet model, which adds spatial conditioning to a pre-trained U-Net.
    """
    def __init__(self, unet_model: nn.Module, hint_channels: int = 3):
        super().__init__()
        # The original U-Net is frozen and not trained.
        self.unet = unet_model
        self.unet.requires_grad_(False)
        print("U-Net has been frozen.")

        # Create a trainable copy of the U-Net's encoder blocks.
        print("Creating trainable copy of U-Net encoder blocks...")
        self.trainable_input_blocks = deepcopy(self.unet.input_blocks)
        self.trainable_middle_block = deepcopy(self.unet.middle_block)

        # This block processes the control condition map (e.g., Canny edges, pose).
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(256, self.unet.input_blocks[0][0].out_channels, 3, padding=1)
        )

        # These "zero-convolution" layers connect the trainable copy to the frozen U-Net.
        # They are initialized to zero so that the ControlNet initially has no effect.
        self.zero_convs = nn.ModuleList([ZeroConv2d(block[-1].out_channels, block[-1].out_channels) for block in self.unet.input_blocks])
        self.middle_block_zero_conv = ZeroConv2d(self.unet.middle_block[-1].out_channels, self.unet.middle_block[-1].out_channels)

    def forward(self, x, t_emb, context, control_hint):
        """
        The forward pass for ControlNet does not return a final output.
        Instead, it returns a list of "residuals" that are added to the
        frozen U-Net's internal feature maps during its forward pass.
        """
        # 1. Process the control hint
        hint = self.input_hint_block(control_hint)
        residuals = []
        h = x

        # 2. Pass inputs through the trainable copy of the U-Net encoder
        for i, module in enumerate(self.trainable_input_blocks):
            h = module(h, t_emb, context)
            if i == 0:  # Add the hint after the first block
                h = h + hint
            residuals.append(self.zero_convs[i](h))

        # 3. Process the middle block
        h = self.trainable_middle_block(h, t_emb, context)
        residuals.append(self.middle_block_zero_conv(h))
        
        return residuals