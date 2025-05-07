import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LargeUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(LargeUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Feature dimensions as per paper
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        # Use dilated convolutions in bottleneck for larger receptive field
        self.bottleneck = DoubleConv(512, 512)
        
        # Upsampling path
        self.up1 = Up(1024, 256, bilinear)  # 512 + 512 = 1024
        self.up2 = Up(512, 128, bilinear)   # 256 + 256 = 512
        self.up3 = Up(256, 64, bilinear)    # 128 + 128 = 256
        self.up4 = Up(128, 32, bilinear)    # 64 + 64 = 128

        #self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.final_upsample = Up(32, 32, bilinear)  # 32 + 32 = 64

       # self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        #self.final_conv = DoubleConv(32, 32)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.outc = OutConv(32, out_channels)
        
        #self.outc = OutConv(32, out_channels)

        self.final_norm = nn.InstanceNorm2d(32, affine=True)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.bottleneck(x5)
        
        # Upsampling path with skip connections
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        
        x = self.final_upsample(x)
        x = self.final_conv(x)
        #x = self.final_norm(x)
        x = self.outc(x)
        
        return x
    
    def __str__(self):
        return "LargeUNet"
    

def load_unet(config):
    checkpoint_path = config['training']['checkpoint_path']
    load = config['training']['load']
    use_instance_norm = config.get('model', {}).get('use_instance_norm', False)
    bilinear = config.get('model', {}).get('bilinear', True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LargeUNet(
        in_channels=1, 
        out_channels=1, 
        bilinear=bilinear
    ).to(device)
    
    if load:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
    return model