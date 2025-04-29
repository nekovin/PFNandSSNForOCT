import torch
import torch.nn as nn
import torch.nn.functional as F

class BlindSpotConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BlindSpotConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        
        # Create parameter mask that zeros center weight and won't be updated
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        mask[:, :, kernel_size//2, kernel_size//2] = 0
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        # Apply mask to weights (without modifying the weights permanently)
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, self.conv.bias, 
                        self.conv.stride, self.conv.padding)

class BlindSpotDoubleConv(nn.Module):
    """Double convolution block with blind spots"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(BlindSpotDoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            BlindSpotConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            BlindSpotConv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class N2VUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256]):
        super(N2VUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Intermediate layers for blind-spot at skip connections
        self.blind_skips = nn.ModuleList()

        # Downsampling
        in_features = in_channels
        for feature in features:
            self.downs.append(BlindSpotDoubleConv(in_features, feature))
            in_features = feature

        # Bottleneck
        self.bottleneck = BlindSpotDoubleConv(features[-1], features[-1] * 2)

        # Upsampling and blind-spot skip connections
        for i, feature in enumerate(reversed(features)):
            # Transpose convolution for upsampling
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            
            # Add blind-spot layer for skip connections
            concat_channels = feature * 2  # Skip features + upsampled features
            self.blind_skips.append(BlindSpotConv2d(concat_channels, concat_channels, kernel_size=1, padding=0))
            
            # Standard double conv after concatenation
            self.ups.append(BlindSpotDoubleConv(concat_channels, feature))

        # Final convolution - also using blind spot
        self.final_conv = BlindSpotConv2d(features[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # Store original shape
        original_shape = x.shape
        
        skip_connections = []

        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling path
        for idx in range(0, len(self.ups), 2):
            # Apply transpose convolution for upsampling
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Handle size mismatches with interpolation if needed
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], 
                    mode="bilinear", align_corners=False
                )

            # Concatenate skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Apply blind-spot convolution to skip connection
            x = self.blind_skips[idx//2](concat_skip)
            
            # Apply double convolution
            x = self.ups[idx+1](x)

        # Final convolution
        output = self.final_conv(x)
        
        # Ensure output has same dimensions as input
        if output.shape[2:] != original_shape[2:]:
            output = F.interpolate(
                output, size=original_shape[2:], 
                mode="bilinear", align_corners=False
            )
            
        return output
    
class Noise2VoidLoss(nn.Module):
    def __init__(self):
        super(Noise2VoidLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)

def get_N2VUNet(in_channels=1, out_channels=1, device='cpu'):
    model = N2VUNet(in_channels=in_channels, 
                   out_channels=out_channels,
                   features=[64, 128, 256])
    return model.to(device)

def get_blind_n2v_unet_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_N2VUNet(in_channels=1, out_channels=1, device=device)
    
    criterion = Noise2VoidLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return device, model, criterion, optimizer