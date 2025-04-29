
from IPython.display import clear_output
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from losses.ssm_loss import custom_loss
from utils.visualise_ssm import visualize_progress
import torch
from utils.visualise_ssm import visualize_attention_maps
from utils.masking import subset_blind_spot_masking
from models.ssm_attention import get_ssm_model
from losses.ssm_loss import custom_loss
from utils.data_loading import preprocessing_v2
import numpy as np
from torch.utils.data import random_split

def process_batch(dataloader, model, history, epoch, num_epochs, optimizer, loss_fn, loss_parameters, debug, n2v_weight, fast, visualise, mode='train'):
    running_loss = 0.0
    running_flow_loss = 0.0
    running_noise_loss = 0.0
    
    # Determine if we're in training or validation mode
    is_training = mode == 'train'
    
    progress_bar = tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch+1}/{num_epochs}")
    print(f"{mode.capitalize()}...")
    
    for batch_inputs, batch_targets in progress_bar:
        # Set model mode appropriately
        if is_training:
            model.train()
        else:
            model.eval()
        
        # Only zero gradients during training
        if is_training and optimizer:
            optimizer.zero_grad()
            
        # Create masked inputs for N2V
        mask = torch.rand_like(batch_inputs) > 0.9  # Mask ~10% of pixels
        masked_inputs = batch_inputs.clone()
        
        if fast:
            roll_amount = torch.randint(-5, 5, (2,))
            shifted = torch.roll(batch_inputs, shifts=(roll_amount[0].item(), roll_amount[1].item()), dims=(2, 3))
            masked_inputs[mask] = shifted[mask]
        else: # proper masking
            masked_inputs = subset_blind_spot_masking(batch_inputs, mask, kernel_size=5)[0]

        # Forward pass
        with torch.set_grad_enabled(is_training):
            outputs = model(masked_inputs)
            flow_component = outputs['flow_component']
            noise_component = outputs['noise_component']

            mse = nn.MSELoss(reduction='none')
            n2v_loss = mse(flow_component[mask], batch_targets[mask]).mean()
            
            if loss_fn:
                total_loss = loss_fn(
                    flow_component, 
                    noise_component, 
                    batch_inputs, 
                    batch_targets, 
                    loss_parameters=loss_parameters, 
                    debug=debug)
                total_loss = total_loss + n2v_loss * n2v_weight
            else:
                total_loss = n2v_loss * n2v_weight

        if is_training and optimizer:
            # Debug parameter changes before step (first epoch only)
            if debug and epoch == 0:
                params_before = [p.clone().detach() for p in model.parameters()]
            
            total_loss.backward()
            optimizer.step()
            
            # Debug parameter changes after step (first epoch only)
            if debug and epoch == 0:
                params_after = [p.clone().detach() for p in model.parameters()]
                any_change = any(torch.any(b != a) for b, a in zip(params_before, params_after))
                print(f"Parameters changed: {any_change}")

        # Track losses
        noise_loss = 0  # Placeholder, adjust if you calculate this elsewhere
        running_loss += total_loss.item()
        running_flow_loss += total_loss.item()
        running_noise_loss += 0
        
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
    
    # Only update history in training mode
    if is_training:
        history['loss'].append(avg_loss)
        history['flow_loss'].append(avg_flow_loss)
        history['noise_loss'].append(avg_noise_loss)

    print(f"{mode.capitalize()} Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Noise Loss: {avg_noise_loss:.6f}")

    if visualise and mode == 'val':
        clear_output(wait=True)
        random.seed(epoch)
        random_idx = random.randint(0, batch_inputs.size(0)-1)
        
        visualize_progress(
            model, 
            batch_inputs[random_idx:random_idx+1], 
            batch_targets[random_idx:random_idx+1], 
            masked_tensor=masked_inputs[random_idx:random_idx+1][0][0].cpu().numpy(), 
            epoch=epoch+1
        )
        plt.close()
        
        visualize_attention_maps(model, batch_inputs[random_idx:random_idx+1][0][0].cpu().numpy())

    return avg_loss

def train(train_dataloader, val_dataloader, checkpoint, checkpoint_path, model, history, optimizer, 
          set_epoch, num_epochs, loss_fn, loss_parameters, debug, n2v_weight, fast, visualise):
    
    # Setup checkpoint paths
    last_checkpoint = checkpoint_path.replace('.pth', f'_last.pth')
    best_checkpoint = checkpoint_path.replace('.pth', f'_best.pth')
    
    # Get best loss from checkpoint if available
    best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else float('inf')
    best_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    
    # Enable anomaly detection for debugging if needed
    torch.autograd.set_detect_anomaly(True)
    
    # Add validation loss to history if not present
    if 'val_loss' not in history:
        history['val_loss'] = []
    
    for epoch in range(set_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_loss = process_batch(
            train_dataloader, model, history, 
            epoch, num_epochs, optimizer, 
            loss_fn, loss_parameters, debug, 
            n2v_weight, fast, visualise,
            mode='train'
        )
        
        # Validation phase
        val_loss = process_batch(
            val_dataloader, model, history, 
            epoch, num_epochs, None,  # No optimizer for validation 
            loss_fn, loss_parameters, False,  # No debug during validation
            n2v_weight, fast, visualise,
            mode='val'
        )
        
        # Store validation loss in history
        history['val_loss'].append(val_loss)
        
        # Check if this is the best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            print(f"New best model found at epoch {best_epoch} with validation loss {best_loss:.6f}")
            
            # Save best model checkpoint
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }
            torch.save(checkpoint, best_checkpoint)
            print(f"Best model checkpoint saved at {best_checkpoint}")

        # Save last checkpoint
        checkpoint = {
            'epoch': epoch + 1,  # Save epoch + 1 so we can resume from next epoch
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        
        torch.save(checkpoint, last_checkpoint)
        print(f"Latest model checkpoint saved at {last_checkpoint}")
    
    return model, history

def get_loaders(dataset, batch_size, val_split=0.2, device='cuda', seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
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
    
    # Stack all tensors
    inputs = torch.stack(input_tensors).to(device)
    targets = torch.stack(target_tensors).to(device)
    
    full_dataset = TensorDataset(inputs, targets)

    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False  # No need to shuffle validation data
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    return train_loader, val_loader
    


def train_speckle_separation_module_n2n(train_config, loss_fn, loss_name):

    device = train_config['device']

    n_patients = train_config['n_patients']

    start = train_config['start']

    dataset = preprocessing_v2(start, n_patients, 50, n_neighbours = 10, threshold=65, sample=False, post_process_size=10)

    batch_size = train_config['batch_size']
    
    #dataloader = get_loaders(dataset, batch_size, device)
    train_loader, val_loader = get_loaders(dataset, batch_size, val_split=0.2, device=device)
    
    history = {
        'loss': [],
        'flow_loss': [],
        'noise_loss': []
    }

    learning_rate = train_config['learning_rate']
    num_epochs = train_config['num_epochs']

    base_checkpoint_path = train_config['checkpoint'].format(loss_fn=loss_name)

    if train_config['load_model']:
        try:
            print(train_config['checkpoint'])
            custom_loss_trained_path = train_config['load_checkpoint'].format(loss_fn=loss_name)
            model = get_ssm_model(checkpoint=custom_loss_trained_path)
            print("Loading model from checkpoint...")
            #checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\{repr(model)}_best.pth"
            
            checkpoint_path = train_config['load_checkpoint'].format(loss_fn=loss_name)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
            set_epoch = checkpoint['epoch']
            history = checkpoint['history']
            num_epochs = num_epochs + set_epoch
            print(f"Model loaded from {checkpoint_path} at epoch {set_epoch} with loss {best_loss:.6f}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e 
    else:
        model = get_ssm_model(checkpoint=None)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float('inf')
        set_epoch = 0

    checkpoint = {
        'epoch': set_epoch,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
        'history': history,
        }

    n2v_weight = train_config['n2v_weight']
    #loss_fn = train_config['loss_fn']
    debug = train_config['debug']
    fast = train_config['fast']
    visualise = train_config['visualise']
    loss_parameters = train_config['loss_parameters']

    train(train_loader, val_loader, checkpoint, base_checkpoint_path, model, history, 
          optimizer, set_epoch, num_epochs, 
          loss_fn, loss_parameters, debug, 
          n2v_weight, fast, visualise)
    
def main():
    #from ssm.train.train import train_speckle_separation_module_n2n
    #from ssm.losses.ssm_loss import custom_loss
    from utils import get_config

    config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\ssm_config.yaml"

    config = get_config(config_path)

    loss_names = ['custom_loss', 'mse']
    loss_name = 'custom_loss'  # 'mse' or 'custom_loss'

    if loss_name == 'mse':
        loss_fn = None
    else:
        loss_fn = custom_loss

    train_speckle_separation_module_n2n(config['training'], loss_fn, loss_name)

    #from ssm.models.ssm_attention import get_ssm_model
    #from torchviz import make_dot
    #import torch

    #model = get_ssm_model(checkpoint=None)
    #x = torch.randn(1, 1, 256, 256)  # Example input tensor
    #outputs = model(x)  # Forward pass through the model

    # Get both outputs
    #flow_output = outputs['flow_component']
    #noise_output = outputs['noise_component']

    # Visualize both outputs together
    #both_outputs = (flow_output, noise_output)
    #graph = make_dot(both_outputs, params=dict(model.named_parameters()))
    #graph.render("ssm_model", format="png")

