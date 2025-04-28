import torch.optim as optim
import time
import os
import torch
from scripts.data_loading import get_loaders
from models.unet import UNet
from models.unet_2 import UNet2
from scripts.visualise import plot_images, plot_computation_graph
from ssm.models.ssm_attention import SpeckleSeparationUNetAttention
from ssm.postprocessing.postprocessing import normalize_image

def process_batch(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise): 
    """
    Process a batch of data through the model, compute loss, and update weights.
    WITHOUT speckle module.
    """
    epoch_loss = 0
    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        outputs = model(input_imgs)
        loss = criterion(outputs, target_imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.6f}")

        if visualise:
            assert input_imgs[0][0].shape == (256, 256)
            assert target_imgs[0][0].shape == (256, 256)
            assert outputs[0][0].shape == (256, 256)
            images = [input_imgs[0][0].cpu().numpy(), target_imgs[0][0].cpu().numpy(), outputs[0][0].cpu().detach().numpy()]
            plot_images(images)

        return epoch_loss
    
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

    
def process_batch(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha):
    mode = 'train' if model.training else 'val'
    
    epoch_loss = 0
    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        if speckle_module is not None:
            flow_inputs = speckle_module(input_imgs)
            flow_inputs = flow_inputs['flow_component'].detach()
            flow_inputs = normalize_image_torch(flow_inputs)
            outputs = model(input_imgs)
            flow_outputs = speckle_module(outputs)
            flow_outputs = flow_outputs['flow_component'].detach()
            flow_outputs = normalize_image_torch(flow_outputs)
            flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
            
            loss = criterion(outputs, target_imgs) + flow_loss * alpha
        else:
            outputs = model(input_imgs)
            loss = criterion(outputs, target_imgs)

            #physics_loss = lognormal_consistency_loss(outputs, target_imgs)
            #loss += physics_loss * 0.01

        
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"{mode.capitalize()} Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.6f}")

        if visualise and batch_idx == 0:
            assert input_imgs[0][0].shape == (256, 256)
            assert target_imgs[0][0].shape == (256, 256)
            assert outputs[0][0].shape == (256, 256)
            
            if speckle_module is not None:
                titles = ['Input Image', 'Flow Input', 'Flow Output', 'Target Image', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(), 
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    target_imgs[0][0].cpu().numpy(), 
                    outputs[0][0].cpu().detach().numpy()
                ]
                losses = {
                    'Flow Loss': flow_loss.item(),
                    'Total Loss': loss.item()
                }
            else:
                titles = ['Input Image', 'Target Image', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(), 
                    target_imgs[0][0].cpu().numpy(), 
                    outputs[0][0].cpu().detach().numpy()
                ]
                losses = {
                    'Total Loss': loss.item()
                }
                
            plot_images(images, titles, losses)

            if epoch == 0 and mode == 'train' and speckle_module is not None:
                plot_computation_graph(model, loss, speckle_module)

    return epoch_loss / len(data_loader)

def train(model, train_loader, val_loader, optimizer, criterion, starting_epoch, epochs, batch_size, lr, best_val_loss, checkpoint_path = None,device='cuda', visualise=False, speckle_module=None, alpha=1, save=False):

    last_checkpoint_path = checkpoint_path + f'_last_checkpoint.pth'
    best_checkpoint_path = checkpoint_path + f'_best_checkpoint.pth'

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch+epochs):
        model.train()

        train_loss = process_batch(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        
        #avg_epoch_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = process_batch(val_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)

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



def train_n2n(config):

    train_config = config['training']
    train_config['method'] = 'n2n'

    n_patients = train_config['n_patients']
    n_images_per_patient = train_config['n_images_per_patient']
    batch_size = train_config['batch_size']
    start = train_config['start_patient'] if train_config['start_patient'] else 1

    train_loader, val_loader = get_loaders(start, n_patients, n_images_per_patient, batch_size)

    method = train_config['method']
    baselines_checkpoint_path = train_config['baselines_checkpoint_path']
    checkpoint_path = baselines_checkpoint_path + rf"{method}/checkpoints/"
    model = train_config['model']

    if config['speckle_module']['use'] is True:
        checkpoint_path = checkpoint_path + rf"{model}_ssm"
        
    else:
        #checkpoint_path = train_config['base_checkpoint_path'] if train_config['base_checkpoint_path'] else None
        checkpoint_path = checkpoint_path + rf"{model}"

    #save_dir = train_config['save_dir'] if train_config['save_dir'] else 'n2n/checkpoints'

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
        speckle_module = SpeckleSeparationUNetAttention(input_channels=1, feature_dim=32).to(device)
        try:
            print("Loading model from checkpoint...")
            ssm_checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\SpeckleSeparationUNetAttention_custom_loss_best.pth"
            ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
            speckle_module.load_state_dict(ssm_checkpoint['model_state_dict'])
            speckle_module.to(device)
            alpha = config['speckle_module']['alpha']
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e 
    else:
        speckle_module = None

    if train_config['load']:
        try:
            checkpoint = torch.load(checkpoint_path + f'_best_checkpoint.pth', map_location=device)
            print("Loading model from checkpoint...")
            print(checkpoint_path + f'_best_checkpoint.pth')
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully")
            print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['best_val_loss']}")
            starting_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")

    if train_config['train']:

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
            save=save)

###

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