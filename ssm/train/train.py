
from IPython.display import clear_output
import random
import numpy as np
import torch

import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.ssm import SpeckleSeparationModule, SpeckleSeparationUNet
from losses.ssm_loss import custom_loss
from utils.visualise import visualize_progress
import torch

from postprocessing.postprocessing import normalize_image
from utils.visualise import visualize_attention_maps

# Faster implementation with convolution-based neighborhood sampling
def blind_spot_masking(tensor, mask, kernel_size=5):
    b, c, h, w = tensor.shape
    masked_tensor = tensor.clone()
    
    # For each batch and channel
    for bi in range(b):
        for ci in range(c):
            # Get masked positions for this channel
            masked_positions = torch.nonzero(mask[bi, ci], as_tuple=True)
            
            if len(masked_positions[0]) == 0:
                continue
                
            # For each masked position
            for i in range(len(masked_positions[0])):
                y, x = masked_positions[0][i], masked_positions[1][i]
                
                # Extract neighborhood
                half_k = kernel_size // 2
                y_min, y_max = max(0, y - half_k), min(h, y + half_k + 1)
                x_min, x_max = max(0, x - half_k), min(w, x + half_k + 1)
                
                # Get neighborhood excluding center pixel
                neighborhood = tensor[bi, ci, y_min:y_max, x_min:x_max].flatten()
                
                # Calculate the index of the center pixel
                center_y, center_x = y - y_min, x - x_min
                center_idx = center_y * (x_max - x_min) + center_x
                
                # Create a mask to exclude the center pixel
                valid_indices = torch.ones(neighborhood.shape[0], dtype=torch.bool, device=tensor.device)
                if 0 <= center_idx < valid_indices.shape[0]:
                    valid_indices[center_idx] = False
                
                # Select a random non-center pixel
                valid_neighborhood = neighborhood[valid_indices]
                if valid_neighborhood.shape[0] > 0:
                    # Random selection
                    rand_idx = torch.randint(0, valid_neighborhood.shape[0], (1,), device=tensor.device)
                    masked_tensor[bi, ci, y, x] = valid_neighborhood[rand_idx]
    
    return masked_tensor

def fast_blind_spot(tensor, mask, kernel_size=5):
    # Create output tensor
    masked_tensor = tensor.clone()
    
    # For each batch
    for b in range(tensor.size(0)):
        # Get all masked positions at once (across all channels)
        masked_positions = torch.nonzero(mask[b], as_tuple=False)
        
        if len(masked_positions) == 0:
            continue
            
        # For each masked position
        for pos in masked_positions:
            c, y, x = pos[0], pos[1], pos[2]
            
            # Get random position offset (-2 to 2 for kernel_size=5)
            offset = kernel_size // 2
            dy = torch.randint(-offset, offset+1, (1,)).item()
            dx = torch.randint(-offset, offset+1, (1,)).item()
            
            # Ensure we don't pick (0,0) offset
            if dy == 0 and dx == 0:
                dy = 1  # Simple fix: just move one pixel up
                
            # Sample from neighborhood with bounds checking
            ny, nx = y + dy, x + dx
            ny = max(0, min(ny, tensor.size(2)-1))
            nx = max(0, min(nx, tensor.size(3)-1))
            
            # Replace masked pixel
            masked_tensor[b, c, y, x] = tensor[b, c, ny, nx]
    
    return masked_tensor

def blind_spot_masking_fast(tensor, mask, kernel_size=5):
    device = tensor.device
    b, c, h, w = tensor.shape
    masked_tensor = tensor.clone()
    half_k = kernel_size // 2
    
    # Process all batches and channels in parallel
    # Get all masked positions
    for bi in range(b):
        for ci in range(c):
            y_coords, x_coords = torch.where(mask[bi, ci])
            
            if len(y_coords) == 0:
                continue
                
            # For efficiency, process in batches of masked pixels
            batch_size = 1000
            for i in range(0, len(y_coords), batch_size):
                y_batch = y_coords[i:i+batch_size]
                x_batch = x_coords[i:i+batch_size]
                
                for idx in range(len(y_batch)):
                    y, x = y_batch[idx].item(), x_batch[idx].item()
                    
                    # Define patch boundaries
                    y_min, y_max = max(0, y - half_k), min(h, y + half_k + 1)
                    x_min, x_max = max(0, x - half_k), min(w, x + half_k + 1)
                    
                    # Create patch mask excluding center
                    patch = tensor[bi, ci, y_min:y_max, x_min:x_max]
                    patch_mask = torch.ones_like(patch, dtype=torch.bool)
                    center_y, center_x = y - y_min, x - x_min
                    if 0 <= center_y < patch_mask.shape[0] and 0 <= center_x < patch_mask.shape[1]:
                        patch_mask[center_y, center_x] = False
                    
                    # Get valid values and pick one randomly
                    valid_values = patch[patch_mask]
                    if len(valid_values) > 0:
                        idx = torch.randint(0, len(valid_values), (1,), device=device)
                        masked_tensor[bi, ci, y, x] = valid_values[idx]
    
    return masked_tensor


def subset_blind_spot_masking(tensor, mask_ratio=0.1, kernel_size=5):
    """
    Apply blind spot masking to only a subset of pixels for faster training
    
    Args:
        tensor: Input tensor of shape [B, C, H, W]
        mask_ratio: Fraction of pixels to mask (between 0 and 1)
        kernel_size: Size of neighborhood kernel
    
    Returns:
        masked_tensor: Tensor with masked pixels filled with neighborhood values
        mask: Boolean mask showing which pixels were masked
    """
    device = tensor.device
    b, c, h, w = tensor.shape
    masked_tensor = tensor.clone()
    half_k = kernel_size // 2
    
    # Create random mask (True = pixels to be masked)
    mask = torch.rand(b, c, h, w, device=device) < mask_ratio
    
    # Use unfold to extract local neighborhoods efficiently
    padded = torch.nn.functional.pad(tensor, (half_k, half_k, half_k, half_k), mode='reflect')
    neighborhoods = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    
    # Create mask for neighborhood (excluding center pixel)
    center_mask = torch.ones((kernel_size, kernel_size), dtype=torch.bool, device=device)
    center_mask[half_k, half_k] = False
    
    for bi in range(b):
        for ci in range(c):
            # Get coordinates of masked pixels
            y_coords, x_coords = torch.where(mask[bi, ci])
            
            if len(y_coords) == 0:
                continue
            
            # Process in batches to avoid OOM
            batch_size = 10000
            for i in range(0, len(y_coords), batch_size):
                y_batch = y_coords[i:i+batch_size]
                x_batch = x_coords[i:i+batch_size]
                batch_len = len(y_batch)
                
                # Get neighborhoods for all masked pixels in batch
                pixel_neighborhoods = neighborhoods[bi, ci, y_batch, x_batch]  # [batch_size, kernel_size, kernel_size]
                
                # Apply center mask to each neighborhood
                masked_neighborhoods = pixel_neighborhoods.reshape(batch_len, -1)[:, center_mask.reshape(-1)]
                
                # For each masked pixel, select a random value from its valid neighbors
                rand_indices = torch.randint(0, masked_neighborhoods.shape[1], (batch_len,), device=device)
                selected_values = masked_neighborhoods[torch.arange(batch_len, device=device), rand_indices]
                
                # Assign the selected values
                masked_tensor[bi, ci, y_batch, x_batch] = selected_values
    
    return masked_tensor, mask

def train_speckle_separation_module_n2n(dataset, 
                                   num_epochs=50, 
                                   batch_size=8, 
                                   learning_rate=1e-4,
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   model=None,
                                   loss_parameters=None,
                                   load_model=False,
                                   debug=False,
                                   fast = False):
    """
    Train the SpeckleSeparationModule using the provided input-target data
    
    Args:
        input_target_data: List of tuples (input, target) where both are numpy arrays
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    print(f"Using device: {device}")


    torch.autograd.set_detect_anomaly(True)
    
    # Prepare the dataset
    input_tensors = []
    target_tensors = []
    
    for patient in dataset:
        patient_data = dataset[patient]
        for input_img, target_img in patient_data:
            # Convert to tensor and add channel dimension if needed
            if len(input_img.shape) == 2:
                input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)
                target_tensor = torch.from_numpy(target_img).float().unsqueeze(0)
            else:
                input_tensor = torch.from_numpy(input_img).float()
                target_tensor = torch.from_numpy(target_img).float()
                
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        
    # Stack into batch dimension
    inputs = torch.stack(input_tensors).to(device)
    targets = torch.stack(target_tensors).to(device)

    # Replace with n2v_loss function
    loss_fn = custom_loss
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create the model
    if model is None:
        return None, None
    
    history = {
        'loss': [],
        'flow_loss': [],
        'noise_loss': []
    }
    best_loss = float('inf')
    epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        try:
            print("Loading model from checkpoint...")
            checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\{repr(model)}_best.pth"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
            set_epoch = checkpoint['epoch']
            history = checkpoint['history']
            num_epochs = num_epochs + set_epoch
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e 
    else:
        best_loss = float('inf')
        set_epoch = 0

    checkpoint = {
        'epoch': set_epoch,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
        'history': history,
        }

    n2v_weight = loss_parameters['n2v_loss'] if loss_parameters else 1.0
    
    # Training loop
    for epoch in range(set_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_flow_loss = 0.0
        running_noise_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")
        for batch_inputs, batch_targets in progress_bar:
            # Zero the gradients
            optimizer.zero_grad()
            
            mask = torch.rand_like(batch_inputs) > 0.9  # Mask ~75% but original paper suggets 5%
            masked_inputs = batch_inputs.clone()

            if fast:
                roll_amount = torch.randint(-5, 5, (2,))
                shifted = torch.roll(batch_inputs, shifts=(roll_amount[0].item(), roll_amount[1].item()), dims=(2, 3))
                masked_inputs[mask] = shifted[mask]
            else: # proper masking
                #masked_inputs = blind_spot_masking(batch_inputs, mask, kernel_size=5)
                #masked_inputs = fast_blind_spot(batch_inputs, mask, kernel_size=5)
                #masked_inputs = blind_spot_masking_fast(batch_inputs, mask, kernel_size=5)
                masked_inputs = subset_blind_spot_masking(batch_inputs, mask, kernel_size=5)[0]

            #plt.imshow(masked_inputs[0][0].cpu().numpy(), cmap='gray')
            #plt.title("Masked Input")
            #plt.show()

            outputs = model(masked_inputs)

            flow_component = outputs['flow_component']
            noise_component = outputs['noise_component']

            mse = nn.MSELoss(reduction='none')
            n2v_loss = mse(flow_component[mask], batch_targets[mask]).mean()
            
            total_loss = loss_fn(flow_component, noise_component, batch_inputs, batch_targets, loss_parameters=loss_parameters, debug=debug)

            total_loss = total_loss + n2v_loss * n2v_weight

            print(f"Total Loss: {total_loss.item()}")

            # Backward pass and optimize
            total_loss.backward()

            if debug and epoch == 0:
                params_before = [p.clone().detach() for p in model.parameters()]

            optimizer.step()

            if debug and epoch == 0:
                params_after = [p.clone().detach() for p in model.parameters()]
                any_change = any(torch.any(b != a) for b, a in zip(params_before, params_after))
                print(f"Parameters changed: {any_change}")

            noise_loss = 0
            
            # Update running losses
            running_loss += total_loss
            running_flow_loss += total_loss.item()
            running_noise_loss += 0 #noise_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'flow_loss': total_loss.item(),
                'noise_loss': noise_loss
            })
        
        # Calculate average losses for the epoch
        avg_loss = running_loss / len(dataloader)
        avg_flow_loss = running_flow_loss / len(dataloader)
        avg_noise_loss = running_noise_loss / len(dataloader)
        
        # Update history
        history['loss'].append(avg_loss)
        history['flow_loss'].append(avg_flow_loss)
        history['noise_loss'].append(avg_noise_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Noise Loss: {avg_noise_loss:.6f}")
        
        #if not debug:
        clear_output(wait=True)
        #visualize_progress(model, inputs[0:1], targets[0:1], epoch+1)
        random.seed(epoch)
        #random_n = random.randint(0, len(inputs)-1)
        #visualize_progress(model, inputs[random_n:random_n+1], targets[random_n:random_n+1], masked_tensor=masked_inputs[random_n:random_n+1].cpu().numpy(), epoch=epoch+1)
        random_idx = random.randint(0, batch_inputs.size(0)-1)  # Get a random index within the current batch
        visualize_progress(model, batch_inputs[random_idx:random_idx+1], batch_targets[random_idx:random_idx+1], 
                        masked_tensor=masked_inputs[random_idx:random_idx+1][0][0].cpu().numpy(), epoch=epoch+1)


    
        plt.close()

        visualize_attention_maps(model, batch_inputs[random_idx:random_idx+1][0][0].cpu().numpy())

        if debug:
            return model, history

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            print(f"New best model found at epoch {best_epoch} with loss {best_loss:.6f}")
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
                'history': history,
            }
            checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\{repr(model)}_best.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # save last
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
                'history': history,
            }
        checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\{repr(model)}_last.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
    
    return model, history