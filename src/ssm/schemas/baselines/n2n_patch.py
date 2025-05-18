
import time
import torch
from ssm.utils.eval_utils.visualise import plot_images
    
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

def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
    """
    Normalise the input image tensor to [0, 1] range.
    
    Args:
        t_img (torch.Tensor): Input image tensor.
        
    Returns:
        torch.Tensor: The normalized image tensor.
    """
    min_val = t_img.min()
    max_val = t_img.max()
    
    if max_val > min_val:
        return (t_img - min_val) / (max_val - min_val)
    else:
        # If all values are the same, return zeros
        return torch.zeros_like(t_img)


def threshold_flow_component(t_img: torch.Tensor, threshold: float = 0.01, bottom_percent: float = 0.2) -> torch.Tensor:
    """
    Creates a binary mask of the flow component with bottom portion blacked out.
    
    Args:
        t_img (torch.Tensor): Input flow component tensor
        threshold (float): Threshold value for binarization
        bottom_percent (float): Percentage of image height to black out from bottom
        
    Returns:
        torch.Tensor: Binary tensor with bottom portion set to zero
    """
    # Create binary threshold
    binary_mask = (t_img > threshold).float()
    
    # Create mask for bottom portion
    batch_size, channels, height, width = binary_mask.shape
    bottom_pixels = int(height * bottom_percent)
    
    # Create a mask where bottom 20% is zero and rest is one
    bottom_mask = torch.ones_like(binary_mask)
    bottom_mask[:, :, -bottom_pixels:, :] = 0
    
    # Apply both masks
    return binary_mask * bottom_mask

def extract_patches(image, patch_size=64, stride=32):
    """Extract patches from an image with given patch size and stride."""
    stride = patch_size // 4
    # Handle different image formats
    if len(image.shape) == 4:  # (B, C, H, W)
        b, c, h, w = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension
        b, c, h, w = image.shape
    elif len(image.shape) == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        b, c, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    all_patches = []
    all_locations = []
    
    for i in range(b):
        img_patches = []
        img_locations = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[i, :, y:y+patch_size, x:x+patch_size]
                img_patches.append(patch)
                img_locations.append((y, x))
        
        all_patches.extend(img_patches)
        all_locations.extend([(i, y, x) for y, x in img_locations])
    
    return torch.stack(all_patches), all_locations

def reconstruct_from_patches(patches, locations, image_shape, patch_size=64):
    """Reconstruct an image from patches based on their locations."""
    # Handle different shape formats
    if len(image_shape) == 4:  # (B, C, H, W)
        b, c, h, w = image_shape
    elif len(image_shape) == 3:  # (C, H, W)
        c, h, w = image_shape
        b = 1
    else:
        raise ValueError(f"Unexpected image shape: {image_shape}")
    
    # Check location format
    sample_location = locations[0]
    if len(sample_location) == 3:  # (batch_idx, y, x)
        # Need to group by batch
        batch_reconstructed = []
        
        # Group patches by batch index
        batch_patches = [[] for _ in range(b)]
        batch_locations = [[] for _ in range(b)]
        
        for i, (patch, location) in enumerate(zip(patches, locations)):
            batch_idx, y, x = location
            batch_patches[batch_idx].append(patch)
            batch_locations[batch_idx].append((y, x))
        
        # Reconstruct each batch item
        for i in range(b):
            if len(batch_patches[i]) > 0:
                # Call recursively with simpler locations
                reconstructed = reconstruct_from_patches(
                    torch.stack(batch_patches[i]), 
                    batch_locations[i],
                    (c, h, w),  # Single image shape
                    patch_size
                )
                batch_reconstructed.append(reconstructed)
            else:
                # Empty tensor if no patches for this batch
                batch_reconstructed.append(torch.zeros((c, h, w), device=patches[0].device))
        
        return torch.stack(batch_reconstructed)
    
    elif len(sample_location) == 2:  # (y, x)
        # Simple reconstruction for a single image
        reconstructed = torch.zeros((c, h, w), device=patches[0].device)
        weights = torch.zeros((h, w), device=patches[0].device)
        
        for patch, (y, x) in zip(patches, locations):
            reconstructed[:, y:y+patch_size, x:x+patch_size] += patch
            weights[y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping regions
        weights = weights.unsqueeze(0).repeat(c, 1, 1)
        weights[weights == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / weights
        
        return reconstructed
    
    else:
        raise ValueError(f"Unexpected location format: {sample_location}")

def process_batch(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha, scheduler):
    mode = 'train' if model.training else 'val'
    
    epoch_loss = 0
    patch_size = 64  # Choose appropriate patch size
    stride = 32      # Choose appropriate stride
    
    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        # Extract patches
        input_patches, patch_locations = extract_patches(input_imgs, patch_size, stride)
        target_patches, _ = extract_patches(target_imgs, patch_size, stride)
        
        # Process patches in sub-batches to avoid memory issues
        sub_batch_size = 32  # Adjust based on your GPU memory
        total_loss = 0
        all_output_patches = []
        
        for i in range(0, len(input_patches), sub_batch_size):
            input_sub_batch = input_patches[i:i+sub_batch_size]
            target_sub_batch = target_patches[i:i+sub_batch_size]

                
            if speckle_module is not None:
                flow_inputs = speckle_module(input_sub_batch)
                flow_inputs = flow_inputs['flow_component'].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                
                outputs = model(input_sub_batch)
                all_output_patches.extend(outputs)
                
                flow_outputs = speckle_module(outputs)
                flow_outputs = flow_outputs['flow_component'].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                
                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                patch_loss = criterion(outputs, target_sub_batch) + flow_loss * alpha
            else:
                outputs = model(input_sub_batch)
                all_output_patches.extend(outputs)
                patch_loss = criterion(outputs, target_sub_batch)
            
            total_loss += patch_loss.item() * len(input_sub_batch)
            
            if mode == 'train':
                optimizer.zero_grad()
                patch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Reconstruct full images from patches for visualization
        if visualise and batch_idx % 10 == 0:
            output_patches = torch.stack(all_output_patches)
            reconstructed_outputs = reconstruct_from_patches(
                output_patches, patch_locations, input_imgs.shape, patch_size
            )
            
            if speckle_module is not None:
                flow_inputs_full = speckle_module(input_imgs)['flow_component'].detach()
                flow_outputs_full = speckle_module(reconstructed_outputs)['flow_component'].detach()
                
                titles = ['Input Image', 'Flow Input', 'Flow Output', 'Target Image', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(), 
                    flow_inputs_full[0][0].cpu().numpy(),
                    flow_outputs_full[0][0].cpu().numpy(),
                    target_imgs[0][0].cpu().numpy(), 
                    reconstructed_outputs[0][0].cpu().numpy()
                ]
                losses = {
                    'Flow Loss': flow_loss.item(),
                    'Total Loss': total_loss / len(input_patches)
                }
            else:
                titles = ['Input Image', 'Target Image', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(), 
                    target_imgs[0][0].cpu().numpy(), 
                    reconstructed_outputs[0][0].cpu().numpy()
                ]
                losses = {
                    'Total Loss': total_loss / len(input_patches)
                }
                
            plot_images(images, titles, losses)
            
        loss_value = total_loss / len(input_patches)
        epoch_loss += loss_value
        
        if mode != 'train':
            scheduler.step(loss_value)

    return epoch_loss / len(data_loader)

def train_n2n_patch(model, train_loader, val_loader, optimizer, criterion, starting_epoch, epochs, 
              batch_size, lr, best_val_loss, checkpoint_path = None,device='cuda', visualise=False, 
              speckle_module=None, alpha=1, save=False, scheduler=None):

    last_checkpoint_path = checkpoint_path + f'_last_checkpoint.pth'
    best_checkpoint_path = checkpoint_path + f'_best_checkpoint.pth'

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch+epochs):
        model.train()
        visualise = False
        train_loss = process_batch(train_loader, model, criterion, optimizer, epoch, starting_epoch+epochs, device, visualise, speckle_module, alpha, scheduler)

        model.eval()
        visualise = True
        with torch.no_grad():
            val_loss = process_batch(val_loader, model, criterion, optimizer, epoch, starting_epoch+epochs, device, visualise, speckle_module, alpha, scheduler)

        print(f"Epoch [{epoch+1}/{starting_epoch+epochs}], Average Loss: {train_loss:.6f}")
        
        if val_loss < best_val_loss and save:
            best_val_loss = val_loss
            print(f"Saving best model with val loss: {val_loss:.6f}")
            print(f"Best checkpoint path: {best_checkpoint_path}")
            print(f"Epoch: {epoch}, Best val loss: {best_val_loss:.6f}")
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

