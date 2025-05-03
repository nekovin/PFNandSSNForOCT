
import time
import torch
from utils.visualise import plot_images
from tqdm import tqdm

def process_batch(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise): 
    """
    Process a batch of data through the model, compute loss, and update weights.
    WITHOUT speckle module.
    """
    epoch_loss = 0
    for batch_idx, (input_imgs, target_imgs) in tqdm(enumerate(data_loader)):
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

            '''
            if epoch == 0 and mode == 'train' and speckle_module is not None:
                plot_computation_graph(model, loss, speckle_module)
            '''

    return epoch_loss / len(data_loader)

def train_n2n(model, train_loader, val_loader, optimizer, criterion, starting_epoch, epochs, batch_size, lr, best_val_loss, checkpoint_path = None,device='cuda', visualise=False, speckle_module=None, alpha=1, save=False):

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
