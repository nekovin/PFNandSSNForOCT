
from IPython.display import clear_output
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from ssm.losses.ssm_loss import custom_loss
from ssm.utils.visualise import visualize_progress
import torch
from ssm.utils.visualise import visualize_attention_maps

def process_batch(dataloader):
    running_loss = 0.0
    running_flow_loss = 0.0
    running_noise_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    print("Training...")
    for batch_inputs, batch_targets in progress_bar:
        model.train()
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

        outputs = model(masked_inputs)
        flow_component = outputs['flow_component']
        noise_component = outputs['noise_component']
        mse = nn.MSELoss(reduction='none')
        n2v_loss = mse(flow_component[mask], batch_targets[mask]).mean()
        total_loss = loss_fn(
            flow_component, 
            noise_component, 
            batch_inputs, 
            batch_targets, 
            loss_parameters=loss_parameters, 
            debug=debug)
        total_loss = total_loss + n2v_loss * n2v_weight

        print(f"Total Loss: {total_loss.item()}")

        total_loss.backward()

        if debug and epoch == 0:
            params_before = [p.clone().detach() for p in model.parameters()]

        optimizer.step()

        if debug and epoch == 0:
            params_after = [p.clone().detach() for p in model.parameters()]
            any_change = any(torch.any(b != a) for b, a in zip(params_before, params_after))
            print(f"Parameters changed: {any_change}")

        noise_loss = 0
        
        running_loss += total_loss
        running_flow_loss += total_loss.item()
        running_noise_loss += 0
        
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Noise Loss: {avg_noise_loss:.6f}")

        if visualise:
            clear_output(wait=True)

            random.seed(epoch)

            random_idx = random.randint(0, batch_inputs.size(0)-1)  # Get a random index within the current batch
            visualize_progress(model, batch_inputs[random_idx:random_idx+1], batch_targets[random_idx:random_idx+1], 
                            masked_tensor=masked_inputs[random_idx:random_idx+1][0][0].cpu().numpy(), epoch=epoch+1)
        
            plt.close()

            visualize_attention_maps(model, batch_inputs[random_idx:random_idx+1][0][0].cpu().numpy())


def train(): # dataset, num_epochs=50, batch_size=8, learning_rate=1e-4,device='cuda' if torch.cuda.is_available() else 'cpu',model=None,loss_parameters=None,load_model=False,debug=False,fast = False

    print(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(set_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        process_batch(dataloader)

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
                'history': history
            }
        checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\{repr(model)}_last.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
    
    return model, history

def get_loader(dataset, batch_size, device):
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
        
    inputs = torch.stack(input_tensors).to(device)
    targets = torch.stack(target_tensors).to(device)

    # Replace with n2v_loss function
    loss_fn = custom_loss
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
from data_loading import preprocessing_v2

def train_speckle_separation_module_n2n(train_config):

    device = train_config['device']

    dataset = preprocessing_v2(1, 20, n_neighbours = 8, threshold=70, sample=False, post_process_size=5)
    
    dataloader = get_loader(dataset, device)

    model = train_config['model']
    
    history = {
        'loss': [],
        'flow_loss': [],
        'noise_loss': []
    }

    best_loss = float('inf')
    epoch = 0

    learning_rate = train_config['learning_rate']
    num_epochs = train_config['num_epochs']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if train_config['load_model']:
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

    n2v_weight = 1.0# alpha

    #train(dataloader)