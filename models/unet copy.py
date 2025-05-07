import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.enc1_features = 64
        self.enc2_features = 128
        self.enc3_features = 256
        self.enc4_features = 512
        self.bottleneck_features = 512
        
        # Encoder (downsampling) with increasing feature dimensions
        self.enc1 = self._block(in_channels, self.enc1_features, name="enc1")
        self.enc2 = self._block(self.enc1_features, self.enc2_features, name="enc2")
        self.enc3 = self._block(self.enc2_features, self.enc3_features, name="enc3")
        self.enc4 = self._block(self.enc3_features, self.enc4_features, name="enc4")
        
        # Bottleneck
        #self.bottleneck = self._block(self.bottleneck_features, self.bottleneck_features, name="bottleneck")
        self.bottleneck = self._block_dilated(self.bottleneck_features, self.bottleneck_features, name="bottleneck")
        
        self.dec1 = self._block(self.bottleneck_features + self.enc4_features, self.enc4_features, name="dec1")
        self.dec2 = self._block(self.enc4_features + self.enc3_features, self.enc3_features, name="dec2")
        self.dec3 = self._block(self.enc3_features + self.enc2_features, self.enc2_features, name="dec3")
        self.dec4 = self._block(self.enc2_features + self.enc1_features, self.enc1_features, name="dec4")
        
        # Final layer
        #self.final = nn.Conv2d(self.enc1_features, out_channels, kernel_size=1)
        self.final = nn.Conv2d(self.enc1_features, out_channels, kernel_size=1)
        
        # Max pooling
        #self.pool = nn.MaxPool2d(2)
        self.pool = nn.AvgPool2d(2)

    # padding options are dilation, reflection, and zeros
        
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, features, kernel_size=3, bias=True),
            #nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, bias=True),
            #nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def _block_dilated(self, in_channels, features, name):
        return nn.Sequential(
            nn.ReflectionPad2d(2),  # Padding of 2 for dilation of 2
            nn.Conv2d(in_channels, features, kernel_size=3, dilation=2, bias=True),
            #nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),  # Padding of 2 for dilation of 2
            nn.Conv2d(features, features, kernel_size=3, dilation=2, bias=True),
            #nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features), # trying this out
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec1 = self.dec1(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='nearest'), enc4], dim=1)) # nearest or bilinear
        dec2 = self.dec2(torch.cat([F.interpolate(dec1, scale_factor=2, mode='nearest'), enc3], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec2, scale_factor=2, mode='nearest'), enc2], dim=1))
        dec4 = self.dec4(torch.cat([F.interpolate(dec3, scale_factor=2, mode='nearest'), enc1], dim=1))
        
        output = dec4

        return self.final(output)
    
    def __str__(self):
        return "Current UNet"
    
    def load_unet(self, config):
        checkpoint_path = config['training']['checkpoint_path']
        load = config['training']['load']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(in_channels=1, out_channels=1).to(device)
        if load:
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.to(device)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e
            
        return model