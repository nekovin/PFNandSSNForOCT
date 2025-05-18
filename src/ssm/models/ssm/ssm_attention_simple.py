import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedSpeckleSeparationModel(nn.Module):
    def __init__(self, input_channels=1, feature_dim=32, depth=4):
        super(SimplifiedSpeckleSeparationModel, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder_channels = []
        
        in_channels = input_channels
        for i in range(depth):
            out_channels = feature_dim * (2**min(i, 2))
            self.encoder_blocks.append(self._create_conv_block(in_channels, out_channels))
            self.encoder_channels.append(out_channels)  # Store for decoder
            in_channels = out_channels
            
        bottleneck_channels = in_channels * 2
        self.bottleneck = self._create_conv_block(in_channels, bottleneck_channels)
        
        self.decoder_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        in_channels = bottleneck_channels
        for i in range(depth):
            up_out_channels = in_channels // 2
            self.up_samples.append(nn.ConvTranspose2d(in_channels, up_out_channels, 
                                                     kernel_size=2, stride=2))

            skip_channels = self.encoder_channels[-(i+1)]
            concat_channels = up_out_channels + skip_channels
            
            out_channels = up_out_channels
            
            self.decoder_blocks.append(self._create_conv_block(concat_channels, out_channels))
            in_channels = out_channels
            
        self.output_flow = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv2d(feature_dim//2, input_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.output_noise = nn.Sequential(
            nn.Conv2d(in_channels, input_channels, kernel_size=1)
        )
    
    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_outputs.append(x)
            if i < len(self.encoder_blocks) - 1:
                x = self.pool(x)
                
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i in range(len(self.decoder_blocks)):
            # Upsample
            x = self.up_samples[i](x)
            
            # Get corresponding encoder output
            encoder_feature = encoder_outputs[-(i+1)]
            
            # Ensure sizes match for concatenation
            if x.shape[2:] != encoder_feature.shape[2:]:
                x = F.interpolate(x, size=encoder_feature.shape[2:], 
                                 mode='bilinear', align_corners=True)
            
            x = torch.cat([x, encoder_feature], dim=1)
            
            x = self.decoder_blocks[i](x)
        
        # Generate output components
        flow_component = self.output_flow(x)
        noise_component = self.output_noise(x)
        
        return {
            'flow_component': flow_component,
            'noise_component': noise_component
        }
    
def get_ssm_model_simple(checkpoint_path):

    print("Loading simplified SSM model...")

    model = SimplifiedSpeckleSeparationModel()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    return model