import torch
import matplotlib.pyplot as plt

def save_checkpoint(epoch, val_loss, model, optimizer, checkpoint_dir, img_size):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / f'{img_size}_checkpoint_epoch_{epoch}.pt')

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