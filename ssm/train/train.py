
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
from ssm.utils.masking import subset_blind_spot_masking
from ssm.models.ssm_attention import get_ssm_model
from ssm.losses.ssm_loss import custom_loss
from data_loading import preprocessing_v2

def process_batch(dataloader, model, history, epoch, num_epochs, optimizer, loss_fn, loss_parameters, debug, n2v_weight, fast, visualise):
    running_loss = 0.0
    running_flow_loss = 0.0
    running_noise_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    print("Training...")
    print(progress_bar)
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

    return avg_loss


def train(dataloader, checkpoint, checkpoint_path, model, history, optimizer, set_epoch, num_epochs, loss_fn, loss_parameters, debug, n2v_weight, 
          fast, visualise): # dataset, num_epochs=50, batch_size=8, learning_rate=1e-4,device='cuda' if torch.cuda.is_available() else 'cpu',model=None,loss_parameters=None,load_model=False,debug=False,fast = False
    last_checkpoint = checkpoint_path.replace('.pth', f'_last.pth')
    best_checkpoint = checkpoint_path.replace('.pth', f'_best.pth')
    best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else float('inf')
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(set_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        loss = process_batch(
            dataloader, model, history, 
            set_epoch, num_epochs, optimizer, 
            loss_fn, loss_parameters, debug, 
            n2v_weight, fast, visualise)

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch + 1
            print(f"New best model found at epoch {best_epoch} with loss {best_loss:.6f}")
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
                'history': history,
            }
            torch.save(checkpoint, best_checkpoint)
            print(f"Model checkpoint saved at {best_checkpoint}")

        # save last
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
                'history': history
            }
        
        torch.save(checkpoint, last_checkpoint)
        print(f"Model checkpoint saved at {last_checkpoint}")
    
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
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
    


def train_speckle_separation_module_n2n(train_config, loss_fn, loss_name):

    device = train_config['device']

    n_patients = train_config['n_patients']

    dataset = preprocessing_v2(n_patients, 50, n_neighbours = 10, threshold=65, sample=False, post_process_size=10)

    batch_size = train_config['batch_size']
    
    dataloader = get_loader(dataset, batch_size, device)
    
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

    train(dataloader, checkpoint, base_checkpoint_path, model, history, 
          optimizer, set_epoch, num_epochs, 
          loss_fn, loss_parameters, debug, 
          n2v_weight, fast, visualise)