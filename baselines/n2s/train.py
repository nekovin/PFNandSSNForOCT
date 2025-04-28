import torch.optim as optim
import time
import os
import torch
import numpy as np
from scripts.data_loading import get_loaders
from models.unet import UNet
from models.unet_2 import UNet2
from scripts.visualise import plot_images, plot_computation_graph

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

def create_partitioning_function(shape, n_partitions=2):
    height, width = shape
    
    def partition_function(i, j):
        return (i + j) % n_partitions
    
    return partition_function

def process_batch_n2s(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module=None, alpha=1.0):
    mode = 'train' if model.training else 'val'
    
    epoch_loss = 0
    for batch_idx, (input_imgs, _) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        batch_size, channels, height, width = input_imgs.shape
        
        partition_fn = create_partitioning_function((height, width))
        
        partition_masks = []
        for p in range(2):
            mask = torch.zeros((height, width), device=device)
            for i in range(height):
                for j in range(width):
                    if partition_fn(i, j) == p:
                        mask[i, j] = 1
            partition_masks.append(mask)
        
        total_loss = 0
        outputs = None
        
        for p in range(len(partition_masks)):
            curr_mask = partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
            
            comp_mask = 1 - curr_mask
            
            masked_input = input_imgs * comp_mask
            
            curr_outputs = model(masked_input)
            
            if p == 0:
                outputs = curr_outputs
            
            target = input_imgs * curr_mask
            
            pred = curr_outputs * curr_mask
            
            loss = criterion(pred, target)
            total_loss += loss
        
        loss = total_loss / len(partition_masks)
        
        if speckle_module is not None and outputs is not None:
            flow_inputs = speckle_module(input_imgs)
            flow_inputs = flow_inputs['flow_component'].detach()
            flow_inputs = normalize_image_torch(flow_inputs)
            
            flow_outputs = speckle_module(outputs)
            flow_outputs = flow_outputs['flow_component'].detach()
            flow_outputs = normalize_image_torch(flow_outputs)
            
            flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
            loss = loss + flow_loss * alpha
        
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"N2S {mode.capitalize()} Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.6f}")

        if visualise and batch_idx == 0 and outputs is not None:
            if speckle_module is not None:
                titles = ['Input Image', 'Flow Input', 'Flow Output', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    outputs[0][0].cpu().detach().numpy()
                ]
                losses = {
                    'N2S Loss': loss.item() - (flow_loss.item() * alpha if speckle_module else 0),
                    'Flow Loss': flow_loss.item() if speckle_module else 0,
                    'Total Loss': loss.item()
                }
            else:
                titles = ['Input Image', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    outputs[0][0].cpu().detach().numpy()
                ]
                losses = {'Total Loss': loss.item()}
                
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
          best_val_loss, checkpoint_path=None, device='cuda', visualise=False, 
          speckle_module=None, alpha=1, save=False, method='n2v'):
    """
    Train function that handles both Noise2Void and Noise2Self approaches.
    
    Args:
        method (str): 'n2v' for Noise2Void or 'n2s' for Noise2Self
    """

    last_checkpoint_path = checkpoint_path + f'_last_checkpoint.pth'
    best_checkpoint_path = checkpoint_path + f'_best_checkpoint.pth'

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch+epochs):
        model.train()

        train_loss = process_batch_n2s(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        
        model.eval()
        with torch.no_grad():
            val_loss = process_batch_n2s(val_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)

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
    method = train_config['method']
    model = train_config['model']

    train_loader, val_loader = get_loaders(start, n_patients, n_images_per_patient, batch_size)
    

    if config['speckle_module']['use'] is True:
        #checkpoint_path = train_config['base_checkpoint_path_speckle']
        #checkpoint_path = train_config['baselines_checkpoint_path'] + f'{method}/checkpoints/{model}_ssm_best_checkpoint.pth'
        checkpoint_path = train_config['baselines_checkpoint_path'] + f'{method}/checkpoints/{model}_ssm'
    else:
        #checkpoint_path = train_config['base_checkpoint_path'] if train_config['base_checkpoint_path'] else None
        checkpoint_path = train_config['baselines_checkpoint_path'] + f'{method}/checkpoints/{model}'
    
    if not os.path.exists(train_config['baselines_checkpoint_path'] + f'{method}/checkpoints'):
        os.makedirs(train_config['baselines_checkpoint_path'] + f'{method}/checkpoints')

    #save_dir = train_config['save_dir'] if train_config['save_dir'] else f'{method}/checkpoints'
    #save_dir.replace("n2n", method)

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
            checkpoint = torch.load(checkpoint_path + f'_best_checkpoint.pth', map_location=device)
            print(f"Loading {method} model from checkpoint...")
            print(checkpoint_path + f'_best_checkpoint.pth')
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
            device=device,
            visualise=visualise,
            speckle_module=speckle_module,
            alpha=alpha,
            save=save,
            method=method)