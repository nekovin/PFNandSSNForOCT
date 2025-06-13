import torch
import torch.nn as nn
import torch.nn.functional as F

from fpss.models.components.components import DoubleConv, Down, Up, OutConv

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(SimpleUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder path - only 3 levels as in SSN2V
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.dropout = nn.Dropout2d(0.1)
        
        # Bottleneck - simple, no attention
        self.bottleneck = DoubleConv(256, 256)
        
        
        self.up1 = Up(256 + 128, 128, bilinear)  # 256 + 256 = 512 input channels
        self.up2 = Up(128 + 64, 64, bilinear)   # 128 + 128 = 256 input channels
        self.up3 = Up(64 + 32, 32, bilinear)
        
        self.outc = OutConv(32, out_channels)
        
        # No dropout, no attention - keep it simple like SSN2V
    
    def forward(self, x):
        # Encoder path - no dropout
        x1 = self.inc(x)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)

        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        
        # Bottleneck
        x4 = self.bottleneck(x4)
        
        # Decoder path - no attention, no dropout
        x = self.up1(x4, x3)
        x = self.dropout(x)
        x = self.up2(x, x2)
        x = self.dropout(x)
        x = self.up3(x, x1)
        x = self.dropout(x)

        
        x = self.outc(x)
        
        return x
    
    def __str__(self):
        return "SimpleUNet"