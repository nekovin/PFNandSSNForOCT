import torch.optim as optim
import time
import os
import torch
import numpy as np
from data_loading import get_loaders
from models.unet import UNet
from models.unet_2 import UNet2
from visualise import plot_images, plot_computation_graph

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
        foreground_mask = t_img > 0.01
        
        if torch.any(foreground_mask):
            fg_values = t_img[foreground_mask]
            fg_min = fg_values.min()
            fg_max = fg_values.max()
            if fg_max > fg_min:
                t_img = torch.where(foreground_mask, (t_img - fg_min) / (fg_max - fg_min), t_img)
        
        # Force background (pixels < 0.01) to be 0
        t_img = torch.where(t_img < 0.01, torch.zeros_like(t_img), t_img)
    return t_img

def create_blind_spot_mask(batch_size, channels, height, width, device, blind_spot_ratio=0.1):
    """
    Create a random blind spot mask for Noise2Void.
    
    Args:
        batch_size (int): Number of images in batch
        channels (int): Number of channels per image
        height (int): Image height
        width (int): Image width
        device (str): Device to create tensor on
        blind_spot_ratio (float): Fraction of pixels to mask
        
    Returns:
        torch.Tensor: Binary mask with 0s at blind spot locations
    """
    mask = torch.ones((batch_size, channels, height, width), device=device)
    
    # Create random mask for each image in batch
    for b in range(batch_size):
        for c in range(channels):
            # Calculate number of pixels to mask
            num_pixels = int(height * width * blind_spot_ratio)
            
            # Generate random indices
            flat_indices = torch.randperm(height * width, device=device)[:num_pixels]
            y_indices = flat_indices // width
            x_indices = flat_indices % width
            
            # Set mask values to 0 at blind spot locations
            mask[b, c, y_indices, x_indices] = 0
    
    return mask

def create_blind_spot_input(input_imgs, mask):
    """
    Create inputs for Noise2Void by replacing masked pixels with neighborhood means.
    
    Args:
        input_imgs (torch.Tensor): Original input images
        mask (torch.Tensor): Binary mask with 0s at blind spot locations
        
    Returns:
        torch.Tensor: Input with masked pixels replaced by neighborhood means
    """
    # Create a copy of the input to modify
    masked_input = input_imgs.clone()
    batch_size, channels, height, width = input_imgs.shape
    
    # For each image in the batch
    for b in range(batch_size):
        for c in range(channels):
            # Find the masked pixel coordinates
            y_coords, x_coords = torch.where(mask[b, c] == 0)
            
            # For each masked pixel
            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                
                # Calculate the neighborhood mean (3x3 window excluding center)
                neighborhood = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue  # Skip the center pixel
                        
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighborhood.append(input_imgs[b, c, ny, nx].item())
                
                # Replace the masked pixel with the neighborhood mean
                if neighborhood:
                    masked_input[b, c, y, x] = sum(neighborhood) / len(neighborhood)
                    
    return masked_input

def process_batch_n2v(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module=None, alpha=1.0):
    """
    Process a batch using the Noise2Void approach.
    
    Args:
        data_loader: DataLoader providing the batch
        model: Neural network model
        criterion: Loss function
        optimizer: Optimizer for model weights
        epoch: Current epoch number
        epochs: Total number of epochs
        device: Device to run computations on
        visualise: Whether to visualize results
        speckle_module: Optional speckle separation module
        alpha: Weight for speckle loss component
        
    Returns:
        float: Average loss for this batch
    """
    mode = 'train' if model.training else 'val'
    
    epoch_loss = 0
    for batch_idx, (input_imgs, _) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        batch_size, channels, height, width = input_imgs.shape
        
        # Create blind spot mask
        mask = create_blind_spot_mask(batch_size, channels, height, width, device)
        
        # Create masked input for N2V
        masked_input = create_blind_spot_input(input_imgs, mask)
        
        # Forward pass
        outputs = model(masked_input)
        
        # Calculate loss only at masked pixels
        masked_targets = input_imgs * (1 - mask)  # Original values at masked locations
        masked_outputs = outputs * (1 - mask)     # Predicted values at masked locations
        
        # N2V loss: prediction error at blind spots
        n2v_loss = criterion(masked_outputs, masked_targets)
        loss = n2v_loss
        
        # Add speckle module constraint if available
        if speckle_module is not None:
            flow_inputs = speckle_module(input_imgs)
            flow_inputs = flow_inputs['flow_component'].detach()
            flow_inputs = normalize_image_torch(flow_inputs)
            
            flow_outputs = speckle_module(outputs)
            flow_outputs = flow_outputs['flow_component'].detach()
            flow_outputs = normalize_image_torch(flow_outputs)
            
            flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
            loss = loss + flow_loss * alpha
        
        # Backpropagation if in training mode
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"N2V {mode.capitalize()} Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.6f}")

        # Visualization for the first batch if enabled
        if visualise and batch_idx == 0:
            if speckle_module is not None:
                titles = ['Input Image', 'Masked Input', 'Flow Input', 'Flow Output', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    masked_input[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    outputs[0][0].cpu().detach().numpy()
                ]
                losses = {
                    'N2V Loss': n2v_loss.item(),
                    'Flow Loss': flow_loss.item() if speckle_module else 0,
                    'Total Loss': loss.item()
                }
            else:
                titles = ['Input Image', 'Masked Input', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    masked_input[0][0].cpu().numpy(),
                    outputs[0][0].cpu().detach().numpy()
                ]
                losses = {'N2V Loss': loss.item()}
                
            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)

def lognormal_consistency_loss(denoised, noisy, epsilon=1e-6):
    """
    Physics-informed loss term that ensures denoised and noisy images 
    maintain the expected log-normal relationship for OCT speckle.
    """
    # Clamp values to prevent zeros and negatives
    denoised_safe = torch.clamp(denoised, min=epsilon)
    noisy_safe = torch.clamp(noisy, min=epsilon)
    
    # Calculate ratio between images (speckle should be multiplicative)
    ratio = noisy_safe / denoised_safe
    
    # Clamp ratio to reasonable range to prevent extreme values
    ratio_safe = torch.clamp(ratio, min=epsilon, max=10.0)
    
    # Log-transform the ratio which should follow normal distribution
    log_ratio = torch.log(ratio_safe)
    
    # For log-normal statistics, calculate parameters
    mu = torch.mean(log_ratio)
    sigma = torch.std(log_ratio)
    
    # Check for NaN values
    if torch.isnan(mu) or torch.isnan(sigma):
        return torch.tensor(0.0, device=denoised.device, requires_grad=True)
    
    # Expected values for log-normal OCT speckle
    expected_mu = 0.0  # Calibrate this
    expected_sigma = 0.5  # Calibrate this
    
    # Penalize deviation from expected log-normal statistics
    loss = torch.abs(mu - expected_mu) + torch.abs(sigma - expected_sigma)
    return loss

def train(model, train_loader, val_loader, optimizer, criterion, starting_epoch, epochs, batch_size, lr, 
          best_val_loss, checkpoint_path=None, save_dir='checkpoints', device='cuda', visualise=False, 
          speckle_module=None, alpha=1, save=False, method='n2v'):
    """
    Train function that handles both Noise2Void and Noise2Self approaches.
    
    Args:
        method (str): 'n2v' for Noise2Void or 'n2s' for Noise2Self
    """
    os.makedirs(save_dir, exist_ok=True)

    last_checkpoint_path = checkpoint_path + f'{model}_{method}_last_checkpoint.pth'
    best_checkpoint_path = checkpoint_path + f'{model}_{method}_best_checkpoint.pth'

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch+epochs):
        model.train()

        # Use N2V processing
        train_loss = process_batch_n2v(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        
        model.eval()
        with torch.no_grad():
            val_loss = process_batch_n2v(val_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {train_loss:.6f}")
        
        if val_loss < best_val_loss and save:
            best_val_loss = val_loss
            print(f"Saving best model with val loss: {val_loss:.6f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, best_checkpoint_path)
    
    if save:
        print(f"Saving last model with val loss: {val_loss:.6f}")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
            }, last_checkpoint_path)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")
    
    return model

def train_denoising(config):
    """
    Main training function that can train with Noise2Void or Noise2Self.
    """
    train_config = config['training']

    n_patients = train_config['n_patients']
    n_images_per_patient = train_config['n_images_per_patient']
    batch_size = train_config['batch_size']
    start = train_config['start_patient'] if train_config['start_patient'] else 1
    method = 'n2v'  # Fixed to Noise2Void 

    train_loader, val_loader = get_loaders(start, n_patients, n_images_per_patient, batch_size)

    if config['speckle_module']['use'] is True:
        checkpoint_path = train_config['base_checkpoint_path_speckle']
    else:
        checkpoint_path = train_config['base_checkpoint_path'] if train_config['base_checkpoint_path'] else None

    save_dir = train_config['save_dir'] if train_config['save_dir'] else f'{method}/checkpoints'
    if "n2n" in save_dir:
        save_dir = save_dir.replace("n2n", method)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_config['model'] == 'UNet':
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif train_config['model'] == 'UNet2':
        model = UNet2(in_channels=1, out_channels=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    visualise = train_config['visualise']

    alpha = 1
    starting_epoch = 0
    best_val_loss = float('inf')

    save = train_config['save']

    if config['speckle_module']['use'] is True:
        from ssm.models.ssm_attention import SpeckleSeparationUNetAttention
        speckle_module = SpeckleSeparationUNetAttention(input_channels=1, feature_dim=32).to(device)
        try:
            print("Loading speckle module from checkpoint...")
            ssm_checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\SpeckleSeparationUNetAttention_custom_loss_best.pth"
            ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
            speckle_module.load_state_dict(ssm_checkpoint['model_state_dict'])
            speckle_module.to(device)
            alpha = config['speckle_module']['alpha']
        except Exception as e:
            print(f"Error loading speckle module: {e}")
            print("Starting training without speckle module.")
            speckle_module = None
    else:
        speckle_module = None

    if train_config['load']:
        try:
            checkpoint = torch.load(checkpoint_path + f'{model}_{method}_best_checkpoint.pth', map_location=device)
            print(f"Loading {method} model from checkpoint...")
            print(checkpoint_path + f'{model}_{method}_best_checkpoint.pth')
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully")
            print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['best_val_loss']}")
            starting_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            print("Starting training from scratch.")

    if train_config['train']:
        print(f"Training Noise2Void model...")
        model = train(
            model,
            train_loader,
            val_loader,
            optimizer=optimizer,
            criterion=train_config['criterion'],
            starting_epoch=starting_epoch,
            epochs=train_config['epochs'], 
            batch_size=train_config['batch_size'], 
            lr=train_config['learning_rate'],
            best_val_loss=best_val_loss,
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            device=device,
            visualise=visualise,
            speckle_module=speckle_module,
            alpha=alpha,
            save=save,
            method=method)