import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, 48, name="enc1")
        self.enc2 = self._block(48, 48, name="enc2")
        self.enc3 = self._block(48, 48, name="enc3")
        self.enc4 = self._block(48, 48, name="enc4")
        
        # Decoder (upsampling)
        self.dec1 = self._block(96, 96, name="dec1")
        self.dec2 = self._block(144, 96, name="dec2")
        self.dec3 = self._block(144, 96, name="dec3")
        
        # Final layer
        self.final = nn.Conv2d(96 + in_channels, out_channels, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Save input for skip connection
        inp = x
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with skip connections
        dec1 = self.dec1(torch.cat([F.interpolate(enc4, scale_factor=2, mode='nearest'), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec1, scale_factor=2, mode='nearest'), enc2], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec2, scale_factor=2, mode='nearest'), enc1], dim=1))
        
        # Final layer with skip connection to input
        final = torch.cat([dec3, inp], dim=1)
        
        return self.final(final)
    
    def __str__(self):
        return "UNet"
    
def load_unet(config): # checkpoint_path = r'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\baselines\n2n\checkpoints\noise2noise_final.pth',device=None, load=False
    checkpoint_path = config['training']['checkpoint_path']
    load = config['training']['load']
    if not device:
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