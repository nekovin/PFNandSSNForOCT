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
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel attention module for OCT image processing
    Adapted from CBAM (Convolutional Block Attention Module)
    """
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module for OCT image processing
    Focuses on important spatial regions containing vessel structures
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_features = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_features))
        
        return x * attention_map  # Apply attention


# Integration example: add these to your encoder/decoder blocks
def add_attention_to_model(model):
    """
    Function to integrate attention modules into existing SpeckleSeparationUNet model
    Call this after model initialization
    """
    # Example of how to integrate attention into encoder blocks
    for i, encoder_block in enumerate(model.encoder_blocks):
        channels = 32 * (2**min(i, 3))  # Match your channel calculation
        
        # Add attention after each encoder block
        attention_modules = nn.Sequential(
            ChannelAttention(channels),
            SpatialAttention()
        )
        
        # Create a new sequential combining original block with attention
        new_block = nn.Sequential(
            encoder_block,
            attention_modules
        )
        
        # Replace the original block
        model.encoder_blocks[i] = new_block
    
    # Similarly for decoder blocks - adapt based on your model's structure
    for i, decoder_block in enumerate(model.decoder_blocks):
        channels = 32 * (2**min(model.depth-i-1, 3))  # Match your channel calculation
        
        attention_modules = nn.Sequential(
            ChannelAttention(channels),
            SpatialAttention()
        )
        
        new_block = nn.Sequential(
            decoder_block,
            attention_modules
        )
        
        model.decoder_blocks[i] = new_block
    
    return model

class AttentionConvBlock(nn.Module):
    """
    Convolutional block with integrated attention mechanisms
    """
    def __init__(self, in_channels, out_channels):
        super(AttentionConvBlock, self).__init__()
        
        # Standard convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SpeckleSeparationUNetAttention(nn.Module):
    """
    Enhanced deeper U-Net architecture for OCT speckle separation with attention mechanisms
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
        super(SpeckleSeparationUNetAttention, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()  # New attention modules for encoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()  # New attention modules for decoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.depth = depth
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.block_depth = block_depth
        
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
            
            # Add attention modules after each encoder block
            self.encoder_attentions.append(
                nn.Sequential(
                    ChannelAttention(out_channels),
                    SpatialAttention()
                )
            )
            
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
        
        # Bottleneck attention
        self.bottleneck_attention = nn.Sequential(
            ChannelAttention(bottleneck_channels),
            SpatialAttention()
        )
        
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
            
            # Add attention modules after each decoder block
            self.decoder_attentions.append(
                nn.Sequential(
                    ChannelAttention(out_channels),
                    SpatialAttention()
                )
            )
            
            in_channels = out_channels
        
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
        
        # Final attention after dilation
        self.final_attention = nn.Sequential(
            ChannelAttention(feature_dim),
            nn.Dropout(0.2),  # Add dropout here
            SpatialAttention(),
            nn.Dropout(0.2)   # And here
        )
        
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

    def __repr__(self):
        return f"SpeckleSeparationUNetAttention(input_channels={self.input_channels}, feature_dim={self.feature_dim}, depth={self.depth}, block_depth={self.block_depth})"

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
        
        # Encoder path with attention
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            x = self.encoder_attentions[i](x)  # Apply attention
            encoder_features.append(x)
            if i < self.depth - 1:
                x = self.pool(x)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.bottleneck_attention(x)
        
        # Decoder path with skip connections and attention
        for i in range(self.depth):
            x = self.up(x)
            # Ensure matching sizes for concatenation
            encoder_feature = encoder_features[self.depth - i - 1]
            if x.size() != encoder_feature.size():
                x = nn.functional.interpolate(x, size=encoder_feature.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
            x = self.decoder_attentions[i](x)  # Apply attention
        
        # Apply dilated convolutions for larger receptive field
        x = self.dilation_block(x)
        x = self.final_attention(x)  # Final attention
        
        # Generate flow and noise components
        flow_component = self.flow_branch(x)
        noise_component = self.noise_branch(x)
        
        return {
            'flow_component': flow_component,
            'noise_component': noise_component
        }
    
def get_ssm_model(input_channels=1, feature_dim=32, depth=5, block_depth=3, checkpoint=None):
    """
    Function to create and return the Speckle Separation Module model
    
    Args:
        input_channels: Number of input image channels (default: 1 for grayscale OCT)
        feature_dim: Initial dimension of feature maps
        depth: Depth of the U-Net (number of downsampling/upsampling operations)
        block_depth: Number of convolution layers in each encoder/decoder block
        
    Returns:
        An instance of the SpeckleSeparationUNetAttention model
    """
    model = SpeckleSeparationUNetAttention(input_channels, feature_dim, depth, block_depth)
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    return model