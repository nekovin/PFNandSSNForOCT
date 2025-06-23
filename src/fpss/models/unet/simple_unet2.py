import torch
import torch.nn as nn
import torch.nn.functional as F

from fpss.models.components.components import DoubleConv, Down, Up, OutConv

class SimpleUNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(SimpleUNet2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder path - only 3 levels as in SSN2V
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        self.dropout = nn.Dropout2d(0.1)
        
        # Bottleneck - simple, no attention
        self.bottleneck = DoubleConv(256, 256)
        
        
        self.up1 = Up(256 + 128, 128, bilinear)  # 256 + 256 = 512 input channels
        self.up2 = Up(128 + 64, 64, bilinear)   # 128 + 128 = 256 input channels
        
        self.outc = OutConv(64, out_channels)
        
        # No dropout, no attention - keep it simple like SSN2V
    
    def forward(self, x):
        # Encoder path - no dropout
        # Encoder path
        x1 = self.inc(x)           # 64 channels (skip for up2)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)        # 128 channels (skip for up1)  
        x2 = self.dropout(x2)
        x3 = self.down2(x2)        # 256 channels
        x3 = self.dropout(x3)

        # Bottleneck
        x4 = self.bottleneck(x3)   # 256 channels

        # Decoder path
        x = self.up1(x4, x2)       # Takes 256 + 128 = 384 channels → outputs 128
        x = self.dropout(x)
        x = self.up2(x, x1)        # Takes 128 + 64 = 192 channels → outputs 64
        x = self.dropout(x)

        # Final output
        x = self.outc(x)           # 64 → out_channels
        
        return x
    
    def __str__(self):
        return "SimpleUNet"