import torch.optim as optim
import time
import torch
from ssm.utils.eval_utils.visualise import plot_images
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

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

def create_partition_masks(shape, n_partitions=2, device='cuda'):
    height, width = shape
    
    y_coords = torch.arange(height, device=device).view(-1, 1).repeat(1, width)
    x_coords = torch.arange(width, device=device).repeat(height, 1)

    coord_sum = (y_coords + x_coords) % n_partitions
    
    partition_masks = []
    for p in range(n_partitions):
        mask = (coord_sum == p).float()
        partition_masks.append(mask)
    
    return partition_masks

def create_random_partition_masks(shape, n_partitions=4, device='cuda'):
    height, width = shape
    all_indices = torch.arange(height * width, device=device)
    shuffled = all_indices[torch.randperm(len(all_indices))]
    
    partition_size = len(all_indices) // n_partitions
    masks = []
    
    for i in range(n_partitions):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < n_partitions - 1 else len(all_indices)
        
        mask = torch.zeros(height * width, device=device)
        mask[shuffled[start_idx:end_idx]] = 1
        mask = mask.reshape(height, width)
        masks.append(mask)
    
    return masks

def _process_batch_n2s(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module=None, alpha=1.0):
    mode = 'train' if model.training else 'val'
    
    epoch_loss = 0

    partition_masks = create_partition_masks((256, 256), n_partitions=2, device=device)
    
    for batch_idx, (input_imgs, _) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        batch_size, channels, height, width = input_imgs.shape

        total_loss = 0
        outputs = None
        
        for p in range(len(partition_masks)):
            curr_mask = partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
            
            comp_mask = 1 - curr_mask
            
            masked_input = input_imgs * comp_mask
            
            curr_outputs = model(masked_input)
            
            if p == 0:
                outputs = curr_outputs
            
            pred = curr_outputs * curr_mask
            target = input_imgs * curr_mask
            loss = criterion(pred, target)
            total_loss += loss
        
        loss = total_loss / len(partition_masks)

        full_output = model(input_imgs)
        
        if speckle_module is not None and outputs is not None:
            flow_inputs = speckle_module(input_imgs)
            flow_inputs = flow_inputs['flow_component'].detach()
            flow_inputs = normalize_image_torch(flow_inputs)
            
            flow_outputs = speckle_module(full_output)
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
                    full_output[0][0].cpu().detach().numpy()
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
                    #outputs[0][0].cpu().detach().numpy()
                    full_output[0][0].cpu().detach().numpy()
                ]
                losses = {'Total Loss': loss.item()}
                
            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)

def process_batch_n2s(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module=None, alpha=1.0):
    from torch.cuda.amp import autocast, GradScaler
    
    mode = 'train' if model.training else 'val'
    epoch_loss = 0
    
    scaler = GradScaler() if mode == 'train' else None
    
    # Standard N2S uses 2 partitions
    #partition_masks = create_partition_masks((256, 256), n_partitions=4, device=device)
    partition_masks = create_random_partition_masks((256, 256), n_partitions=8, device=device)
    
    
    for batch_idx, (input_imgs, _) in enumerate(tqdm(data_loader)):
        input_imgs = input_imgs.to(device)

        with autocast():
            total_loss = 0
            final_output = torch.zeros_like(input_imgs)
            
            for p in range(len(partition_masks)):
                curr_mask = partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
                comp_mask = 1 - curr_mask
                
                # Mask current partition pixels for input
                masked_input = input_imgs * comp_mask
                
                # Predict only the masked pixels
                curr_outputs = model(masked_input)
                
                # Accumulate predictions for final output
                final_output += curr_outputs * curr_mask
                
                # Calculate loss only for current partition
                pred = curr_outputs * curr_mask
                target = input_imgs * curr_mask
                loss = criterion(pred, target)
                total_loss += loss
            
            # Average loss across partitions
            loss = total_loss / len(partition_masks)

            # SSM loss if enabled
            if speckle_module is not None:
                flow_inputs = speckle_module(input_imgs)
                flow_inputs = flow_inputs['flow_component'].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                
                flow_outputs = speckle_module(final_output)
                flow_outputs = flow_outputs['flow_component'].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                
                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                loss = loss + flow_loss * alpha
        
        if mode == 'train':
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        epoch_loss += loss.item()

        if visualise and batch_idx == 0:
            # Use final_output for visualization, not the partial outputs
            if speckle_module is not None:
                titles = ['Input Image', 'Flow Input', 'Flow Output', 'Output Image']
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    final_output[0][0].cpu().detach().numpy()
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
                    final_output[0][0].cpu().detach().numpy()
                ]
                losses = {'Total Loss': loss.item()}
                
            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)

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

def process_batch_n2s_with_clean_inference(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module=None, alpha=1.0):
    """
    N2S training with periodic clean inference training
    """
    from torch.cuda.amp import autocast, GradScaler
    
    mode = 'train' if model.training else 'val'
    epoch_loss = 0
    
    scaler = GradScaler() if mode == 'train' else None
    
    partition_masks = create_random_partition_masks((256, 256), n_partitions=8, device=device)
    
    for batch_idx, (input_imgs, _) in tqdm_notebook(enumerate(data_loader)):
        input_imgs = input_imgs.to(device)
        
        if mode == 'train' and batch_idx % 10 == 0:
            with autocast():
                clean_output = model(input_imgs)
                
                final_output = torch.zeros_like(input_imgs)
                for p in range(len(partition_masks)):
                    curr_mask = partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
                    comp_mask = 1 - curr_mask
                    masked_input = input_imgs * comp_mask
                    curr_outputs = model(masked_input)
                    final_output += curr_outputs * curr_mask

                consistency_loss = criterion(clean_output, final_output.detach())
                
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(consistency_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    consistency_loss.backward()
                    optimizer.step()
        
        with autocast():
            total_loss = 0
            final_output = torch.zeros_like(input_imgs)
            
            for p in range(len(partition_masks)):
                curr_mask = partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
                comp_mask = 1 - curr_mask
                masked_input = input_imgs * comp_mask
                curr_outputs = model(masked_input)
                final_output += curr_outputs * curr_mask
                
                pred = curr_outputs * curr_mask
                target = input_imgs * curr_mask
                loss = criterion(pred, target)
                total_loss += loss
            
            loss = total_loss / len(partition_masks)
            
            if speckle_module is not None:
                flow_inputs = speckle_module(input_imgs)['flow_component'].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                flow_outputs = speckle_module(final_output)['flow_component'].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                loss = loss + flow_loss * alpha
        
        if mode == 'train':
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        epoch_loss += loss.item()
        
        # Visualization with clean output comparison
        if visualise and batch_idx == 0:
            clean_output = model(input_imgs)
            
            titles = ['Input', 'N2S Output', 'Clean Output', 'Clean vs N2S']
            images = [
                input_imgs[0][0].cpu().numpy(),
                final_output[0][0].cpu().detach().numpy(),
                clean_output[0][0].cpu().detach().numpy(),
                (clean_output[0][0] - final_output[0][0]).abs().cpu().detach().numpy()
            ]
            
            losses = {'Total Loss': loss.item()}
            plot_images(images, titles, losses)
    
    return epoch_loss / len(data_loader)


def train_n2s(model, train_loader, val_loader, optimizer, criterion, starting_epoch, epochs, batch_size, lr, 
          best_val_loss, checkpoint_path=None, device='cuda', visualise=False, 
          speckle_module=None, alpha=1, save=False):

    last_checkpoint_path = checkpoint_path + f'_last_checkpoint.pth'
    best_checkpoint_path = checkpoint_path + f'_best_checkpoint.pth'

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()

    for epoch in tqdm_notebook(range(starting_epoch, starting_epoch+epochs)):
        model.train()
        #train_loss = process_batch_n2s(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        train_loss = process_batch_n2s_with_clean_inference(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        
        model.eval()
        with torch.no_grad():
            #val_loss = process_batch_n2s(val_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
            val_loss = process_batch_n2s_with_clean_inference(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)

        print(f"Epoch [{starting_epoch+epoch+1}/{epochs}], Average Loss: {train_loss:.6f}")
        
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