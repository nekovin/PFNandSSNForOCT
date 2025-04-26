import torch.optim as optim
import time
import os
import torch
import numpy as np
from data_loading import get_loaders
from models.unet import UNet
from models.unet_2 import UNet2
from visualise import plot_images, plot_computation_graph
from IPython.display import clear_output
from baselines.n2v.utils import normalize_image_torch, load, visualise_n2v, plot_loss, enhanced_differentiable_threshold_octa_torch, compute_octa, create_blind_spot_input_with_realistic_noise


def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
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
    mask = torch.ones((batch_size, channels, height, width), device=device)
    
    # Determine number of pixels to mask per image
    num_pixels = int(height * width * blind_spot_ratio)
    
    for b in range(batch_size):
        for c in range(channels):
            # Generate random pixel coordinates
            coords = torch.randperm(height * width, device=device)[:num_pixels]
            y_coords = coords // width
            x_coords = coords % width
            
            # Set mask to 0 at blind spot locations
            mask[b, c, y_coords, x_coords] = 0
            
    return mask

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

def process_batch_n2v(
        model, loader, criterion, mask_ratio,
        optimizer=None,  # Optional parameter - present for training, None for evaluation
        device='cuda',
        visualize=False
        ):
    """
    Process a batch using the standard Noise2Void approach.
    
    Args:
        model: The neural network model
        loader: DataLoader providing batches (returns raw1, raw2)
        criterion: Loss function for N2V training
        mask_ratio: Ratio of pixels to mask
        optimizer: Optimizer (if None, we're in evaluation mode)
        device: Device to run computations on
        visualize: Whether to visualize results
        
    Returns:
        float: Average loss for the epoch
    """
    from contextlib import nullcontext
    
    # Set model mode based on whether we're training or evaluating
    if optimizer:  # If optimizer is provided, we're in training mode
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    
    # Use nullcontext when training, torch.no_grad when evaluating
    context_manager = torch.no_grad() if not optimizer else nullcontext()
    
    with context_manager:
        for batch in loader:
            raw1, raw2 = batch

            raw1 = raw1.to(device)
            raw2 = raw2.to(device)

            # Create blind spot mask using Bernoulli distribution
            mask = torch.bernoulli(torch.full((raw1.size(0), 1, raw1.size(2), raw1.size(3)), 
                                            mask_ratio, device=device))

            # Create masked inputs with neighborhood means
            blind1 = create_blind_spot_input_with_realistic_noise(raw1, mask).requires_grad_(True)
            blind2 = create_blind_spot_input_with_realistic_noise(raw2, mask).requires_grad_(True)
            
            # Zero gradients if we're training
            if optimizer:
                optimizer.zero_grad()

            # Forward pass through the model
            outputs1 = model(blind1)
            outputs2 = model(blind2)
            
            # Calculate N2V losses - only on masked pixels
            n2v_loss1 = criterion(outputs1[mask > 0], raw1[mask > 0])
            n2v_loss2 = criterion(outputs2[mask > 0], raw2[mask > 0])

            # Combine losses (just the N2V losses)
            loss = n2v_loss1 + n2v_loss2
            
            # Backprop if we're training
            if optimizer:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if visualize:
                # clear output
                clear_output(wait=True)
                visualise_n2v(
                    raw1=raw1.cpu().detach().numpy(),
                    blind1=blind1.cpu().detach().numpy(),
                    blind2=blind2.cpu().detach().numpy(),
                    output1=outputs1.cpu().detach().numpy(),
                    output2=outputs2.cpu().detach().numpy()
                )
    
    return total_loss / len(loader)

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

def visualise_n2v(raw1, blind1, blind2, output1, output2):
    """
    Visualization function for Noise2Void results.
    This should be replaced with your actual visualization code.
    
    Args:
        raw1: Original images
        blind1: First masked input
        blind2: Second masked input 
        output1: First model output
        output2: Second model output
    """
    import matplotlib.pyplot as plt
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot the images
    axes[0, 0].imshow(raw1[0, 0], cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(blind1[0, 0], cmap='gray')
    axes[0, 1].set_title('Masked Input 1')
    
    axes[0, 2].imshow(output1[0, 0], cmap='gray')
    axes[0, 2].set_title('Output 1')
    
    axes[1, 0].imshow(raw1[0, 0] - output1[0, 0], cmap='gray')
    axes[1, 0].set_title('Residual 1')
    
    axes[1, 1].imshow(blind2[0, 0], cmap='gray')
    axes[1, 1].set_title('Masked Input 2')
    
    axes[1, 2].imshow(output2[0, 0], cmap='gray')
    axes[1, 2].set_title('Output 2')
    
    plt.tight_layout()
    plt.show()

def create_partitioning_function(shape, n_partitions=2):
    height, width = shape
    
    def partition_function(i, j):
        return (i + j) % n_partitions
    
    return partition_function

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
          speckle_module=None, alpha=1, save=False, method='n2v', octa_criterion=None, threshold=0.0, mask_ratio=0.1):
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

        print(model)

        train_loss = process_batch_n2v(model, train_loader, criterion, mask_ratio,
            optimizer=optimizer, 
            device='cuda',
            visualize=False)
        
        model.eval()
        with torch.no_grad():
            val_loss = process_batch_n2v(model, val_loader, criterion, mask_ratio,
                optimizer=None, 
                device='cuda',
                visualize=True)

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
    method = train_config.get('method', 'n2v')  # Default to Noise2Void if not specified

    train_loader, val_loader = get_loaders(start, n_patients, n_images_per_patient, batch_size)

    # assert
    sample = next(iter(train_loader))
    assert len(sample) == 2


    if config['speckle_module']['use'] is True:
        checkpoint_path = train_config['base_checkpoint_path_speckle']
    else:
        checkpoint_path = train_config['base_checkpoint_path'] if train_config['base_checkpoint_path'] else None

    save_dir = train_config['save_dir'] if train_config['save_dir'] else f'{method}/checkpoints'
    save_dir.replace("n2n", method)

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
        print(f"Training {method} model...")
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
            method=method,
            octa_criterion=compute_octa,
            threshold=train_config['threshold'],
            mask_ratio=train_config['mask_ratio'])