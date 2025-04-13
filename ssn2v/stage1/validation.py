import torch
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")
from stage1.utils.utils import normalize_data
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from tqdm import tqdm
from matplotlib.colors import NoNorm

def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    #noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    return blind_input

def visualise_n2v(blind_input, target_img, output, mask=None):
    """
    Visualize the N2V process with mask overlay
    
    Args:
        blind_input: Input with blind spots
        target_img: Target noisy image
        output: Model prediction
        mask: Binary mask showing pixel positions for N2V
    """
    # Normalize output to match input scale
    output = torch.from_numpy(output).float()
    #output = normalize_data(output)
    
    clear_output(wait=True)
    
    # If mask is provided, show 4 images including mask
    if mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot blind spot input
        axes[0].imshow(blind_input.squeeze(), cmap='gray', norm=NoNorm())
        axes[0].axis('off')
        axes[0].set_title('Blind-spot Input')
        
        # Plot model output
        axes[1].imshow(output.squeeze(), cmap='gray', norm=NoNorm())
        axes[1].axis('off')
        axes[1].set_title('Output Image')
        
        # Plot target image
        axes[2].imshow(target_img.squeeze(), cmap='gray', norm=NoNorm())
        axes[2].axis('off')
        axes[2].set_title('Target Noisy Image')
    
    # Otherwise use original 3-image layout
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(blind_input.squeeze(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Blind-spot Input')
        
        axes[1].imshow(output.squeeze(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Output Image')
        
        axes[2].imshow(target_img.squeeze(), cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Target Noisy Image')
    
    plt.tight_layout()
    plt.show()

def validate_n2v(model, val_loader, criterion, mask_ratio, device='cuda', visualise=False):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for octa in val_loader:
            octa = octa.to(device)

            mask = torch.bernoulli(torch.full((octa.size(0), 1, octa.size(2), octa.size(3)), 
                                            mask_ratio, device=device))
            
            blind_octa = create_blind_spot_input_fast(octa, mask)
            
            outputs = model(blind_octa)
                
            #outputs = normalize_data(outputs) 

            loss = criterion(outputs, octa)
            
            total_loss += loss.item()
            
            if visualise:
                visualise_n2v(
                    blind_octa.cpu().detach().numpy(),
                    octa.cpu().detach().numpy(),
                    outputs.cpu().detach().numpy(),
                )
    
    return total_loss / len(val_loader)