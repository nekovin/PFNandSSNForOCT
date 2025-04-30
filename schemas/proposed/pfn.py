
def predict(self, model, data):
        import glob

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint_files = glob.glob(rf'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints/{self.img_size}_checkpoint_epoch_*.pt')

        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError("No checkpoint files found")
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            for data, _ in data:
                #print(data)
                data = data.to(device)
                input_img = data[:, 0, :, :, :]
                fused_img = data[:, -1, :, :, :]

                #batch = next(iter(data))[0].to(device)
                output_img = model(input_img)

                output_img = self.normalize_to_target(output_img, fused_img)

                break

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_img.cpu().squeeze().numpy(), cmap='gray')
        axes[0].axis('off')

        axes[1].imshow(fused_img.cpu().squeeze().numpy(), cmap='gray')
        axes[1].axis('off')

        axes[2].imshow(output_img.cpu().squeeze().numpy(), cmap='gray')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        return output_img

import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from losses.pfn_loss import compute_loss, compute_dynamic_loss

class trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=rf'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints',
        img_size=512,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        # self.checkpoint_dir.mkdir(exist_ok=True)
        self.img_size = img_size

        self.vis_dir = Path(r'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\results\visualisation')
        self.vis_dir.mkdir(exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        #self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()
        
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, num_epochs, load=False):
        last_checkpoint_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\phf_last.pth"
        best_checkpoint_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\phf_best.pth"
        if load:
            checkpoint = torch.load(last_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
        else:
            start_epoch = 0
            best_val_loss = float('inf')                          
        
        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(epoch, val_loss, self.model, self.optimizer, r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\phf_best.pth", self.img_size)
            
            save_checkpoint(epoch, val_loss, self.model, self.optimizer, r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\phf_last.pth", self.img_size)
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            plot_losses(self.history, self.vis_dir)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                desc=f'Epoch {epoch}')
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            num_levels = data.shape[1] - 1
            batch_loss = 0
            
            self.optimizer.zero_grad() 
            
            for level in range(num_levels):
                input_img = data[:, level, :, :, :]
                target_images = data[:, level+1, :, :, :]
                
                output = self.model(input_img)
                output = normalize_to_target(output, target_images)
                
                level_losses = compute_dynamic_loss(output, target_images, level, num_levels)

                level_weight = (level + 1) / num_levels
                batch_loss += level_losses['total_loss'] * level_weight

            input_img = data[:, 0, :, :, :]
            target_images = data[:, -1, :, :, :]
            output = self.model(input_img)
            output = normalize_to_target(output, target_images)
            final_losses = compute_loss(output, target_images)

            final_weight = 1.0
            batch_loss += final_losses['total_loss'] * final_weight

            
            batch_loss.backward()

            self.optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            pbar.set_postfix({'loss': f"{sum(epoch_losses)/len(epoch_losses):.4f}"})
            
            if batch_idx % 50 == 0:
                visualize_batch(epoch, batch_idx, input_img, output, target_images, self.vis_dir)
        
        return sum(epoch_losses) / len(epoch_losses)

    def validate(self):
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, _ in tqdm(self.val_loader, desc='Validating'):
                data = data.to(self.device)
                
                input_images = data[:, 0, :, :, :]
                target_images = data[:, -1, :, :, :]
                
                #output = self.model(data)
                output = self.model(input_images)

                output = normalize_to_target(output, target_images)
                
                losses = compute_loss(output, target_images)
                
                val_losses.append(losses['total_loss'].item())
        
        return sum(val_losses) / len(val_losses)
    
import torch

def save_checkpoint(epoch, val_loss, model, optimizer, checkpoint_dir, img_size):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_dir)

def normalize_to_target(input_img, target_img):
    target_mean = target_img.mean()
    target_std = target_img.std()
    input_mean = input_img.mean()
    input_std = input_img.std()

    eps = 1e-8
    input_std = torch.clamp(input_std, min=eps)
    target_std = torch.clamp(target_std, min=eps)
    
    scale = torch.abs(target_std / input_std)
    
    normalized = ((input_img - input_mean) * scale) + target_mean
    
    return normalized

def compute_low_signal_mask(output, threshold_factor=0.5):
        """Compute mask for low-signal areas based on intensity."""
        mean_intensity = output.mean()
        threshold = threshold_factor * mean_intensity  # Define threshold as a fraction of the mean
        low_signal_mask = (output < threshold).float()
        return low_signal_mask

import matplotlib.pyplot as plt

def visualize_batch(epoch, batch_idx, input_images, output_images, target_images, vis_dir):
        """Visualize a batch of images"""

        fig, axes = plt.subplots(3, min(4, input_images.shape[0]), 
                                figsize=(15, 10))
        
        if input_images.shape[0] == 1:
            axes = axes.reshape(-1, 1)
            
        for i in range(min(4, input_images.shape[0])):
            input_img = input_images[i].cpu().squeeze().numpy()
            output_img = output_images[i].detach().cpu().squeeze().numpy()
            target_img = target_images[i].cpu().squeeze().numpy()
            
            axes[0, i].imshow(input_img, cmap='gray')
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(output_img, cmap='gray')
            axes[1, i].set_title(f'Output {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(target_img, cmap='gray')
            axes[2, i].set_title(f'Target {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        #plt.savefig(vis_dir / f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()

def visualize_fusion_levels(epoch, batch_idx, data, vis_dir):
    """Visualize all fusion levels for a single image"""
    n_levels = data.shape[1]
    fig, axes = plt.subplots(1, n_levels, figsize=(20, 4))
    
    for i in range(n_levels):
        img = data[0, i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Level {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / f'fusion_levels_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def plot_losses(history, vis_dir):
    """Plot training history"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(vis_dir / 'loss_history.png')
    plt.close()

import numpy as np
from pathlib import Path

def save_batch(epoch, batch_idx, input_images, output_images, target_images, data_dir):
    """Save a batch of images to a file for later visualisation."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    input_images = np.array([img.cpu().numpy() for img in input_images])
    output_images = np.array([img.detach().cpu().numpy() for img in output_images])
    target_images = np.array([img.cpu().numpy() for img in target_images])

    # Save as a compressed file
    np.savez_compressed(
        data_dir / f'epoch_{epoch}_batch_{batch_idx}.npz',
        inputs=input_images,
        outputs=output_images,
        targets=target_images
    )

def main():
    #from schemas.proposed.pfn import trainer
    from models.prog import create_progressive_fusion_dynamic_unet
    import torch
    from utils.pfn_data import get_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    levels = 6
    img_size = 256

    model = create_progressive_fusion_dynamic_unet()
    train_loader, val_loader, test_loader = get_dataset(basedir=r"C:\Datasets\OCTData\data\FusedDataset", size=img_size, levels=levels) #6,1,w,h # [0] is the images

    pfn_trainer = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        checkpoint_dir=rf'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints',
        img_size=img_size,
    )

    pfn_trainer.train(num_epochs=10)