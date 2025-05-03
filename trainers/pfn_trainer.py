
import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import json
from models.prog import create_progressive_fusion_dynamic_unet
from utils.pfn_data import get_dataset

def save_checkpoint(epoch, val_loss, model, optimizer, checkpoint_path, img_size):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

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

def visualize_batch(self, epoch, batch_idx, input_img, outputs, target_images, vis_dir):
        input_img = input_img.cpu().numpy()
        outputs = [output.cpu().numpy() for output in outputs]
        target_images = [target.cpu().numpy() for target in target_images]

        # Create a grid of images
        grid_input = make_grid(torch.tensor(input_img), nrow=1, normalize=True, scale_each=True)
        grid_output = make_grid(torch.tensor(outputs[0]), nrow=1, normalize=True, scale_each=True)
        grid_target = make_grid(torch.tensor(target_images[0]), nrow=1, normalize=True, scale_each=True)

        # Save the grid images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(grid_input.permute(1, 2, 0).numpy())
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(grid_output.permute(1, 2, 0).numpy())
        plt.title('Output Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(grid_target.permute(1, 2, 0).numpy())
        plt.title('Target Image')
        plt.axis('off')

        #plt.savefig(vis_dir / f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()
class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=rf'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints',
        img_size=300,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.img_size = img_size

        self.vis_dir = Path('../visualizations')
        self.vis_dir.mkdir(exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.l1_loss = L1Loss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, num_epochs):
        best_val_loss = float('inf')
        best_checkpoint_path = self.checkpoint_dir / f'{self.img_size}_best_checkpoint.pth'
        last_checkpoint_path = self.checkpoint_dir / f'{self.img_size}_last_checkpoint.pth'
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(epoch, val_loss, self.model, self.optimizer, best_checkpoint_path, self.img_size)
            save_checkpoint(epoch, val_loss, self.model, self.optimizer, last_checkpoint_path, self.img_size)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            #plot_losses(self.history, self.vis_dir)
        return self.history

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch}')
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.device)
            num_levels = data.shape[1] - 1
            batch_loss = 0
            
            self.optimizer.zero_grad() 
            
            input_img = data[:, 0, :, :, :]  # Input image
            target_images = [data[:, i, :, :, :] for i in range(1, num_levels + 1)]  # Targets
            
            # Forward pass with dynamic levels
            outputs = self.model(input_img, num_levels, target_images[0].shape)
            
            # Compute loss dynamically
            for output, target in zip(outputs, target_images):
                output = normalize_to_target(output, target)
                batch_loss += self.l1_loss(output, target)
            batch_loss /= num_levels
            
            # Backward pass
            batch_loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            pbar.set_postfix({'loss': f"{sum(epoch_losses) / len(epoch_losses):.4f}"})
            
        return sum(epoch_losses) / len(epoch_losses)

    def validate(self):
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, _ in tqdm(self.val_loader, desc='Validating'):
                data = data.to(self.device)
                
                input_img = data[:, 0, :, :, :]  # Input image
                num_levels = data.shape[1] - 1
                target_images = [data[:, i, :, :, :] for i in range(1, num_levels + 1)]  # Targets
                
                # Forward pass
                outputs = self.model(input_img, num_levels, target_images[0].shape)
                
                # Compute loss
                batch_loss = 0
                for output, target in zip(outputs, target_images):
                    output = normalize_to_target(output, target)
                    batch_loss += self.l1_loss(output, target)
                batch_loss /= num_levels
                
                val_losses.append(batch_loss.item())

                #visualize_batch(0, 0, input_img, outputs, target_images, self.vis_dir)
        
        return sum(val_losses) / len(val_losses)

def train_pfn():
    model = create_progressive_fusion_dynamic_unet(base_features=32, use_fusion=True)

    load = True
    if load:
        checkpoint_path = rf'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\300_best_checkpoint.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print("Model not loaded.")

    levels = None
    img_size = 300

    train_loader, val_loader, test_loader = get_dataset(basedir=r'C:\Users\CL-11\OneDrive\Repos\phf\data\FusedDataset', size=img_size, levels=levels)

    trainer = Trainer(model, train_loader, val_loader, img_size=img_size)
    history = trainer.train(num_epochs=5)

    with open('training_history.json', 'w') as f:
        json.dump(history, f) # Save the training history to a JSON file

if __name__ == "__main__":
    train_pfn()