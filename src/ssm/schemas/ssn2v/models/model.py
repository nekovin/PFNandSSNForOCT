import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # For kernel_size=5 and dilation=2, we need padding=4 to maintain dimensions
        # The formula is: padding = (kernel_size - 1) * dilation / 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            # For the second conv, we need padding=2 with kernel_size=5 and dilation=1
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        nn.init.kaiming_normal_(self.double_conv[0].weight)
        nn.init.kaiming_normal_(self.double_conv[4].weight)

    def forward(self, x):
        return self.double_conv(x)

class NoiseToVoidUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling
        in_features = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_features, feature))
            in_features = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.ReLU()
        )

        # Feature extraction layers - keep these for internal use
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(f, f//2, kernel_size=1),
                nn.BatchNorm2d(f//2),
                nn.LeakyReLU(inplace=True)
            ) for f in features
        ])

    def forward(self, x):
        # Store original dimensions
        original_shape = x.shape
        
        skip_connections = []
        # We'll still calculate extracted_features but won't return them
        extracted_features = []

        # Downsampling and feature extraction
        for down, feature_layer in zip(self.downs, self.feature_layers):
            x = down(x)
            skip_connections.append(x)
            extracted_features.append(feature_layer(x))
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], 
                    mode="bilinear", align_corners=False
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Get the output
        output = self.final_conv(x)
        
        # Ensure output matches the original input dimensions
        if output.shape[2:] != original_shape[2:]:
            output = torch.nn.functional.interpolate(
                output, size=original_shape[2:], 
                mode="bilinear", align_corners=False
            )
            
        # Return only the output image
        return output
    
    def __str__(self):
        return "N2NUNet"

def get_n2n_unet_model(in_channels=1, out_channels=1, device='cpu'):
    model = NoiseToVoidUNet(in_channels=in_channels, 
                           out_channels=out_channels,
                           features=[64, 128, 256, 512])
    return model.to(device)