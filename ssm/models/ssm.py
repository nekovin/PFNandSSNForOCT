import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")

class SpeckleSeparationModule(nn.Module):
    """
    A simplified module to separate OCT speckle into:
    - Informative speckle (related to blood flow)
    - Noise speckle (to be removed)
    """
    
    def __init__(self, input_channels=1, feature_dim=32):
        """
        Initialize the Speckle Separation Module
        
        Args:
            input_channels: Number of input image channels (default: 1 for grayscale OCT)
            feature_dim: Dimension of feature maps
        """
        super(SpeckleSeparationModule, self).__init__()
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Flow component branch
        self.flow_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1),
            nn.Tanh()
        )
        
        # Noise component branch
        self.noise_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass of the Speckle Separation Module
        
        Args:
            x: Input OCT image tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - 'flow_component': Flow-related speckle component
                - 'noise_component': Noise-related speckle component
        """
        # Extract features
        features = self.feature_extraction(x)
        
        # Separate into flow and noise components
        flow_component = self.flow_branch(features)
        noise_component = self.noise_branch(features)
        
        return {
            'flow_component': flow_component,
            'noise_component': noise_component
        }
    
######################
    
class SpeckleSeparationUNet(nn.Module):
    """
    Enhanced deeper U-Net architecture for OCT speckle separation
    """
    
    def __init__(self, input_channels=1, feature_dim=32, depth=5, block_depth=3):
        """
        Initialize the Deeper Speckle Separation U-Net Module
        
        Args:
            input_channels: Number of input image channels (default: 1 for grayscale OCT)
            feature_dim: Initial dimension of feature maps
            depth: Depth of the U-Net (number of downsampling/upsampling operations)
            block_depth: Number of convolution layers in each encoder/decoder block
        """
        super(SpeckleSeparationUNet, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.depth = depth
        
        # Encoder path with deeper blocks
        in_channels = input_channels
        for i in range(depth):
            out_channels = feature_dim * (2**min(i, 3))  # Cap feature growth to avoid excessive memory usage
            encoder_block = []
            
            # First conv in the block
            encoder_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            encoder_block.append(nn.BatchNorm2d(out_channels))
            encoder_block.append(nn.ReLU(inplace=True))
            
            # Additional conv layers in each block
            for _ in range(block_depth - 1):
                encoder_block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                encoder_block.append(nn.BatchNorm2d(out_channels))
                encoder_block.append(nn.ReLU(inplace=True))
            
            self.encoder_blocks.append(nn.Sequential(*encoder_block))
            in_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = feature_dim * (2**min(depth, 3))
        bottleneck = []
        for _ in range(block_depth + 1):  # Slightly deeper bottleneck
            bottleneck.append(nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1))
            bottleneck.append(nn.BatchNorm2d(bottleneck_channels))
            bottleneck.append(nn.ReLU(inplace=True))
            in_channels = bottleneck_channels
        
        self.bottleneck = nn.Sequential(*bottleneck)
        
        # Decoder path with deeper blocks
        in_channels = bottleneck_channels
        for i in range(depth):
            out_channels = feature_dim * (2**min(depth-i-1, 3))
            decoder_block = []
            
            # First conv after the skip connection
            decoder_block.append(nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1))
            decoder_block.append(nn.BatchNorm2d(out_channels))
            decoder_block.append(nn.ReLU(inplace=True))
            
            # Additional conv layers in each block
            for _ in range(block_depth - 1):
                decoder_block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                decoder_block.append(nn.BatchNorm2d(out_channels))
                decoder_block.append(nn.ReLU(inplace=True))
            
            self.decoder_blocks.append(nn.Sequential(*decoder_block))
            in_channels = out_channels
        
        # Output layers with residual connections
        self.flow_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1)
        )
        
        self.noise_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1)
        )
        
        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Add dilated convolutions for wider receptive field
        self.dilation_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass of the Speckle Separation U-Net
        
        Args:
            x: Input OCT image tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - 'flow_component': Flow-related speckle component
                - 'noise_component': Noise-related speckle component
        """
        # Store encoder outputs for skip connections
        encoder_features = []
        
        # Encoder path
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            encoder_features.append(x)
            if i < self.depth - 1:
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i in range(self.depth):
            x = self.up(x)
            # Ensure matching sizes for concatenation
            encoder_feature = encoder_features[self.depth - i - 1]
            if x.size() != encoder_feature.size():
                x = nn.functional.interpolate(x, size=encoder_feature.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
        
        # Apply dilated convolutions for larger receptive field
        x = self.dilation_block(x)
        
        # Generate flow and noise components
        flow_component = self.flow_branch(x)
        noise_component = self.noise_branch(x)
        
        return {
            'flow_component': flow_component,
            'noise_component': noise_component
        }