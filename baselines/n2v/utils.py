import torch

def create_blind_spot_input_fast(image, mask): # This creates an artificial situation: your network learns to reconstruct pixels from surrounding context, but the masking pattern (black dots) doesn't match the actual noise distribution in OCT images.
    blind_input = image.clone()
    #noise = torch.randn_like(image) * image.std() + image.mean()
    #blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    #blind_input = torch.where(mask.bool(), torch.zeros_like(image), image)
    blind_input[mask.bool()] = 0
    return blind_input

def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    #noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    return blind_input


def create_blind_spot_input_with_realistic_noise(image, mask):
    blind_input = image.clone()
    
    # Parameters for OCT-like noise (speckle)
    mean_level = image.mean()
    std_level = image.std() * 0.9  # Adjust based on your OCT noise level
    
    # Generate noise with speckle-like characteristics
    noise = torch.randn_like(image) * std_level + mean_level
    
    # Apply mask
    blind_input[mask.bool()] = noise[mask.bool()]
    return blind_input

def compute_octa(oct1, oct2):

    numerator = (oct1 - oct2)**2
    denominator = oct1**2 + oct2**2
    
    epsilon = 1e-10
    octa = numerator / (denominator + epsilon)
    
    return octa

import torch
import numpy as np

def threshold_octa(octa, oct, threshold):
    oct = oct.cpu().numpy()
    octa = octa.detach().cpu().numpy()
    background_mask = oct > np.percentile(oct, threshold)  # Bottom 20% of OCT values
    
    if np.sum(background_mask) > 0:  # Ensure we have background pixels
        background_mean = np.mean(oct[background_mask])
        background_std = np.std(oct[background_mask])
        
        threshold = background_mean + 2 * background_std
    else:
        # If no background pixels, use a fixed threshold
        print("No background pixels found, using fixed threshold")
        threshold = np.percentile(oct, 1)
    
    signal_mask = oct > threshold
    
    thresholded_octa = octa * signal_mask
    
    return thresholded_octa

def threshold_octa_torch(octa: torch.Tensor, oct: torch.Tensor, threshold_percent: float):
    """
    Apply a threshold to the OCTA image based on the statistics of the OCT image.
    
    Args:
        octa (torch.Tensor): The OCTA image tensor (should require gradients).
        oct (torch.Tensor): The corresponding OCT image tensor.
        threshold_percent (float): Percentile value (0-100) for initial thresholding.
        
    Returns:
        torch.Tensor: The thresholded OCTA image.
    """
    # Compute the percentile value using torch.quantile (percentile_percent/100.0)
    t_percentile = torch.quantile(oct, threshold_percent / 100.0)

    # Create the background mask for OCT pixels
    background_mask = oct > t_percentile

    # If we have background pixels, compute the mean and std of those pixels.
    if background_mask.sum() > 0:
        background_mean = torch.mean(oct[background_mask])
        background_std = torch.std(oct[background_mask])
        threshold_val = background_mean + 2 * background_std
    else:
        # If no background pixels, default to a fixed low percentile (e.g., 1st percentile)
        threshold_val = torch.quantile(oct, 0.01)
    
    # Create a signal mask based on the computed threshold_val.
    signal_mask = oct > threshold_val

    # Multiply the OCTA image with the signal mask; cast the boolean mask to float.
    thresholded_octa = octa * signal_mask.float()

    return thresholded_octa

def enhanced_differentiable_threshold_octa_torch(octa, oct, threshold_percentile=80, smoothness=3.0, enhancement_factor=1.2):
    """
    Enhanced differentiable thresholding for OCTA images with vessel structure preservation.
    
    Args:
        octa: The OCTA image tensor (computed from OCT differences)
        oct: The OCT tensor used for thresholding reference
        threshold_percentile: Percentile (0-100) to determine foreground/background
        smoothness: Controls the transition sharpness in the sigmoid (lower = smoother)
        enhancement_factor: Factor to enhance vessel structures (higher enhances vessels)
    
    Returns:
        Thresholded OCTA image with preserved vessel structures
    """
    # Get foreground/background threshold using percentile
    sorted_values, _ = torch.sort(oct.reshape(-1))
    threshold_idx = int(sorted_values.shape[0] * threshold_percentile / 100)
    threshold_value = sorted_values[threshold_idx]
    
    # Use pixels ABOVE threshold as foreground (vessels + tissue)
    # This is more appropriate for OCT where vessels appear bright
    foreground_mask = oct > threshold_value
    
    # Compute statistics for better thresholding
    if torch.sum(foreground_mask) > 0:
        foreground_pixels = oct[foreground_mask]
        foreground_mean = torch.mean(foreground_pixels)
        foreground_std = torch.std(foreground_pixels)
        
        # Set threshold at mean - std to include most vessel structures
        signal_threshold = foreground_mean - foreground_std
    else:
        # Fallback: use high percentile threshold
        fallback_idx = int(sorted_values.shape[0] * 0.95)  # 95th percentile
        signal_threshold = sorted_values[fallback_idx]
    
    # Create gradient-aware mask using softplus instead of sigmoid
    # Softplus provides a smoother transition and better preserves fine structures
    soft_mask = torch.log(1 + torch.exp(smoothness * (oct - signal_threshold)))
    
    # Normalize soft mask to [0,1] range
    if soft_mask.max() > soft_mask.min():
        soft_mask = (soft_mask - soft_mask.min()) / (soft_mask.max() - soft_mask.min())
    
    # Apply vessel enhancement by emphasizing high OCTA values
    enhanced_octa = octa * (1.0 + (octa * enhancement_factor))
    
    # Apply soft mask to the enhanced OCTA
    thresholded_octa = enhanced_octa * soft_mask
    
    # Final normalization to ensure output is well-scaled
    if thresholded_octa.max() > 0:
        thresholded_octa = thresholded_octa / thresholded_octa.max()
    
    return thresholded_octa

from matplotlib.colors import NoNorm
import matplotlib.pyplot as plt
from IPython.display import clear_output
def visualise_n2v(raw1, oct1, oct2, output1, output2, normalize_output1, octa_from_outputs, thresholded_octa, stage1_output):
    """
    Visualize the N2V process with mask overlay
    """
    clear_output(wait=True)
    
    # Create figure with specific axis layout
    fig = plt.figure(figsize=(24, 6))
    
    # Create a grid with specific positions
    grid = plt.GridSpec(3, 5, figure=fig)
    
    # Create only the axes you need
    ax1 = fig.add_subplot(grid[0, 0])
    axraw = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0])

    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[2, 1])
    axnorm = fig.add_subplot(grid[1, 1])

    ax5 = fig.add_subplot(grid[1, 2])
    ax6 = fig.add_subplot(grid[1, 3])
    ax7 = fig.add_subplot(grid[1, 4])
    ax8 = fig.add_subplot(grid[0, 3])
    ax9 = fig.add_subplot(grid[0, 4])
    
    # Plot on individual axes
    ax1.imshow(oct1.squeeze(), cmap='gray', norm=NoNorm())
    ax1.axis('off')
    ax1.set_title('OCT1')

    axraw.imshow(raw1.squeeze(), cmap='gray')
    axraw.axis('off')
    axraw.set_title('Raw1 (norm)')
    
    ax2.imshow(oct2.squeeze(), cmap='gray', norm=NoNorm())
    ax2.axis('off')
    ax2.set_title('OCT2')
    
    ax3.imshow(output1.squeeze(), cmap='gray')
    ax3.axis('off')
    ax3.set_title('Output1')

    ax4.imshow(output2.squeeze(), cmap='gray', norm=NoNorm())
    ax4.axis('off')
    ax4.set_title('Output2')

    axnorm.imshow(normalize_output1.squeeze(), cmap='gray')
    axnorm.axis('off')
    axnorm.set_title('Normalized Output1')

    ax5.imshow(octa_from_outputs.squeeze(), cmap='gray', norm=NoNorm())
    ax5.axis('off')
    ax5.set_title('Octa from outputs')

    ax6.imshow(thresholded_octa.squeeze(), cmap='gray')
    ax6.axis('off')
    ax6.set_title('Thresholded OCTA (normalised)')

    ax7.imshow(stage1_output.squeeze(), cmap='gray')
    ax7.axis('off')
    ax7.set_title('Stage 1 Output (normalised)')

    ax8.imshow(stage1_output.squeeze(), cmap='gray', norm=NoNorm())
    ax8.axis('off')
    ax8.set_title('Stage 1 Output')

    ax9.imshow(thresholded_octa.squeeze(), cmap='gray', norm=NoNorm())
    ax9.axis('off')
    ax9.set_title('Thresholded OCTA')

    plt.tight_layout()
    plt.show()

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

import torch
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")

def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
    """
    Normalise the input image tensor.
    
    For pixels above 0.01, computes the min and max (foreground) and scales
    those pixels to the [0, 1] range. Pixels below 0.01 are forced to 0.
    
    Args:
        t_img (torch.Tensor): Input image tensor.
        
    Returns:
        torch.Tensor: The normalized image tensor.
    """
    if t_img.max() > 0:
        # Create mask for non-background (foreground) pixels.
        foreground_mask = t_img > 0.01
        
        # Check if any foreground pixel is found.
        if torch.any(foreground_mask):
            fg_values = t_img[foreground_mask]
            fg_min = fg_values.min()
            fg_max = fg_values.max()
            
            # Normalize only if there is a valid range.
            if fg_max > fg_min:
                # Use torch.where to selectively update foreground pixels.
                t_img = torch.where(foreground_mask, (t_img - fg_min) / (fg_max - fg_min), t_img)
        
        # Force background (pixels < 0.01) to be 0
        t_img = torch.where(t_img < 0.01, torch.zeros_like(t_img), t_img)
    return t_img

def freeze():
    '''
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=1e-3
        )
        '''
    pass

def load(model, optimizer, save_path, scratch, device):

    print(f"Received save_path: {save_path}")
    print(f"Current working directory: {os.getcwd()}")

    checkpoints_dir = os.path.dirname(save_path)
    if not os.path.exists(checkpoints_dir) and not scratch:
        os.makedirs(checkpoints_dir, exist_ok=True)

    if scratch:
        try:
            checkpoint = torch.load(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v\checkpoints\stage1.pth")
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch+1}")
            return model, optimizer, old_epoch, history
        except:
            raise ValueError("Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path.")
    else:
        try:
            checkpoint = torch.load(save_path)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch+1}")
            return model, optimizer, old_epoch, history
        except:
            raise ValueError("Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path.")
    
def check_performance(val_loss, best_val_loss, model, optimizer, epoch, save_path, history, old_epoch, avg_train_loss):
    if val_loss < best_val_loss:
        print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch+1}")
        best_val_loss = val_loss
        try:
            torch.save({
                'epoch': epoch + old_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, save_path)
        
        except:
            print("Err")
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        print("-" * 50)

