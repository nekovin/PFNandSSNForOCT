import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the NonLocal Block
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # Define transformations for query, key, and value
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        # Output transformation
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # g(x): [B, C, H, W] -> [B, C//2, H, W]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, C//2, H*W]
        g_x = g_x.permute(0, 2, 1)  # [B, H*W, C//2]
        
        # theta(x): [B, C, H, W] -> [B, C//2, H, W]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [B, C//2, H*W]
        theta_x = theta_x.permute(0, 2, 1)  # [B, H*W, C//2]
        
        # phi(x): [B, C, H, W] -> [B, C//2, H, W]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C//2, H*W]
        
        # Calculate attention map
        f = torch.matmul(theta_x, phi_x)  # [B, H*W, H*W]
        f_div_C = F.softmax(f, dim=-1)
        
        # Weighted sum using the attention map
        y = torch.matmul(f_div_C, g_x)  # [B, H*W, C//2]
        y = y.permute(0, 2, 1).contiguous()  # [B, C//2, H*W]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [B, C//2, H, W]
        
        # Final transformation and residual connection
        W_y = self.W(y)  # [B, C, H, W]
        W_y = self.bn(W_y)  # Apply batch normalization
        z = W_y + x  # Residual connection
        
        return z

# Generator Network (with Nonlocal blocks)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # [B, 64, H/2, W/2]
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )  # [B, 128, H/4, W/4]
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )  # [B, 256, H/8, W/8]
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )  # [B, 512, H/16, W/16]
        
        # NonLocal blocks
        self.nonlocal1 = NonLocalBlock(features * 8)
        self.nonlocal2 = NonLocalBlock(features * 4)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True)
        )  # [B, 256, H/8, W/8]
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True)
        )  # [B, 128, H/4, W/4]
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )  # [B, 64, H/2, W/2]
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )  # [B, 1, H, W]
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Apply NonLocal block at bottleneck
        e4 = self.nonlocal1(e4)
        
        # Decoder with skip connections
        d1 = self.dec1(e4)
        d1 = self.nonlocal2(d1)
        d1 = torch.cat([d1, e3], dim=1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        
        d4 = self.dec4(d3)
        
        return d4

# PatchGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: No batch norm
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# OCT Dataset class
class OCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.tif')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # For unsupervised training, return the same image as both input and target
        return image

# Custom loss functions
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, denoised, noisy):
        return self.l1(denoised, noisy)

# Training function
def train_nonlocal_gan(generator, discriminator, train_loader, num_epochs=100, 
                      lr_g=0.0002, lr_d=0.0002, save_path='models'):
    # Create directories for saving models and samples
    os.makedirs(save_path, exist_ok=True)
    samples_dir = os.path.join(save_path, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Initialize optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = ContentLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        running_loss_g = 0.0
        running_loss_d = 0.0
        
        for i, noisy_images in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            noisy_images = noisy_images.to(device)
            batch_size = noisy_images.size(0)
            
            # Ground truth labels for discriminator
            real_labels = torch.ones(batch_size, 1, 16, 16).to(device)  # Size depends on your discriminator's output
            fake_labels = torch.zeros(batch_size, 1, 16, 16).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()
            
            # Pass real images through discriminator
            real_outputs = discriminator(noisy_images)
            d_loss_real = adversarial_loss(real_outputs, real_labels)
            
            # Generate denoised images
            denoised_images = generator(noisy_images)
            
            # Pass fake (denoised) images through discriminator
            fake_outputs = discriminator(denoised_images.detach())
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optimizer_d.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad()
            
            # Pass fake (denoised) images through discriminator for generator training
            fake_outputs = discriminator(denoised_images)
            g_loss_adv = adversarial_loss(fake_outputs, real_labels)
            
            # Content loss
            g_loss_content = content_loss(denoised_images, noisy_images) * 10.0  # Weight higher for content preservation
            
            # Total generator loss
            g_loss = g_loss_adv + g_loss_content
            g_loss.backward()
            optimizer_g.step()
            
            # Track losses
            running_loss_g += g_loss.item()
            running_loss_d += d_loss.item()
            
            # Save samples every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    generator.eval()
                    sample_noisy = noisy_images[0:4].cpu()
                    sample_denoised = generator(sample_noisy.to(device)).cpu()
                    
                    # Create grid of images
                    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                    for j in range(4):
                        # Original noisy images
                        axes[0, j].imshow(sample_noisy[j].squeeze(), cmap='gray')
                        axes[0, j].set_title('Noisy')
                        axes[0, j].axis('off')
                        
                        # Denoised images
                        axes[1, j].imshow(sample_denoised[j].squeeze(), cmap='gray')
                        axes[1, j].set_title('Denoised')
                        axes[1, j].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(samples_dir, f'epoch_{epoch+1}_iter_{i}.png'))
                    plt.close()
                    generator.train()
        
        # Print epoch summary
        avg_loss_g = running_loss_g / len(train_loader)
        avg_loss_d = running_loss_d / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - D Loss: {avg_loss_d:.4f}, G Loss: {avg_loss_g:.4f}')
        
        # Save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'epoch': epoch,
            }, os.path.join(save_path, f'nonlocal_gan_epoch_{epoch+1}.pth'))

# Evaluation function
def evaluate_model(generator, test_loader, save_dir='results'):
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, noisy_images in enumerate(tqdm(test_loader, desc='Evaluating')):
            noisy_images = noisy_images.to(device)
            denoised_images = generator(noisy_images)
            
            # Save the first 10 test images for visualization
            if i < 10:
                for j in range(min(4, noisy_images.size(0))):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Original noisy image
                    ax1.imshow(noisy_images[j].squeeze().cpu(), cmap='gray')
                    ax1.set_title('Noisy Image')
                    ax1.axis('off')
                    
                    # Denoised image
                    ax2.imshow(denoised_images[j].squeeze().cpu(), cmap='gray')
                    ax2.set_title('Denoised Image')
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'test_sample_{i}_{j}.png'))
                    plt.close()

# Main function to set up and run the training
def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 100
    lr_g = 0.0002
    lr_d = 0.0002
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Create dataset and dataloader
    # Replace with your OCT dataset path
    dataset = OCTDataset(root_dir='path/to/oct/images', transform=transform)
    
    # Split into train and test sets (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Train the model
    train_nonlocal_gan(generator, discriminator, train_loader, num_epochs, lr_g, lr_d)
    
    # Evaluate the model
    evaluate_model(generator, test_loader)

# Function to calculate quantitative metrics
def calculate_metrics(original_images, denoised_images):
    """
    Calculate PSNR, SNR, CNR and ENL metrics
    
    Args:
        original_images: Tensor of original/noisy images
        denoised_images: Tensor of denoised images
    
    Returns:
        Dictionary with calculated metrics
    """
    # Convert tensors to numpy arrays
    original_np = original_images.detach().cpu().numpy()
    denoised_np = denoised_images.detach().cpu().numpy()
    
    # Initialize metrics
    psnr_list = []
    snr_list = []
    cnr_list = []
    enl_list = []
    
    for i in range(original_np.shape[0]):
        # Extract single images
        orig = original_np[i, 0]  # Assuming single-channel images
        denoise = denoised_np[i, 0]
        
        # Calculate MSE
        mse = np.mean((orig - denoise) ** 2)
        
        # PSNR calculation
        if mse == 0:
            psnr = 100
        else:
            max_pixel = 1.0
            psnr = 10 * np.log10((max_pixel ** 2) / mse)
        psnr_list.append(psnr)
        
        # SNR calculation
        signal_power = np.mean(denoise ** 2)
        noise_power = np.var(orig - denoise)
        if noise_power == 0:
            snr = 100
        else:
            snr = 10 * np.log10(signal_power / noise_power)
        snr_list.append(snr)
        
        # For CNR and ENL, we need to select regions
        # This is a simplified version - in practice, you would select 
        # appropriate foreground and background regions
        
        # Simple segmentation - top 20% pixels as foreground, bottom 20% as background
        sorted_pixels = np.sort(denoise.flatten())
        bg_threshold = sorted_pixels[int(0.2 * len(sorted_pixels))]
        fg_threshold = sorted_pixels[int(0.8 * len(sorted_pixels))]
        
        fg_mask = denoise > fg_threshold
        bg_mask = denoise < bg_threshold
        
        if np.sum(fg_mask) > 0 and np.sum(bg_mask) > 0:
            # Foreground and background statistics
            fg_mean = np.mean(denoise[fg_mask])
            fg_std = np.std(denoise[fg_mask])
            bg_mean = np.mean(denoise[bg_mask])
            bg_std = np.std(denoise[bg_mask])
            
            # Calculate CNR
            if (fg_std**2 + bg_std**2) > 0:
                cnr = 10 * np.log10(np.abs(fg_mean - bg_mean) / np.sqrt(fg_std**2 + bg_std**2))
                cnr_list.append(cnr)
            
            # Calculate ENL
            if fg_std > 0:
                enl = (fg_mean**2) / (fg_std**2)
                enl_list.append(enl)
    
    # Return average metrics
    metrics = {
        "PSNR": np.mean(psnr_list) if psnr_list else 0,
        "SNR": np.mean(snr_list) if snr_list else 0,
        "CNR": np.mean(cnr_list) if cnr_list else 0,
        "ENL": np.mean(enl_list) if enl_list else 0
    }
    
    return metrics

# Usage example
if __name__ == "__main__":
    main()