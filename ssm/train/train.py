
from IPython.display import clear_output
import random

def train_speckle_separation_module_n2n(dataset, 
                                   num_epochs=50, 
                                   batch_size=8, 
                                   learning_rate=1e-4,
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   unet=False,
                                   loss_parameters=None,
                                   load_model=False):
    """
    Train the SpeckleSeparationModule using the provided input-target data
    
    Args:
        input_target_data: List of tuples (input, target) where both are numpy arrays
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    print(f"Using device: {device}")


    torch.autograd.set_detect_anomaly(True)
    
    # Prepare the dataset
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
        
    # Stack into batch dimension
    inputs = torch.stack(input_tensors).to(device)
    targets = torch.stack(target_tensors).to(device)

    # Replace with n2v_loss function
    loss_fn = custom_loss
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create the model
    if unet:
        model = SpeckleSeparationUNet(input_channels=1, feature_dim=32).to(device)
    else:
        model = SpeckleSeparationModule(input_channels=1, feature_dim=32).to(device)

    history = {
        'loss': [],
        'flow_loss': [],
        'noise_loss': []
    }
    best_loss = float('inf')
    epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        try:
            print("Loading model from checkpoint...")
            checkpoint_path = r"checkpoints/speckle_separation_model_best.pth"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
            epoch = checkpoint['epoch']
            history = checkpoint['history']
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e 
    else:
        best_loss = float('inf')
        epoch = 0

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
        'history': history,
        }

    n2v_weight = loss_parameters['n2v_loss'] if loss_parameters else 1.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_flow_loss = 0.0
        running_noise_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")
        for batch_inputs, batch_targets in progress_bar:
            # Zero the gradients
            optimizer.zero_grad()
            
            mask = torch.rand_like(batch_inputs) > 0.9  # Mask ~75% but original paper suggets 5%
            masked_inputs = batch_inputs.clone()

            # Replace masked pixels with random nearby pixel values
            roll_amount = torch.randint(-5, 5, (2,))
            shifted = torch.roll(batch_inputs, shifts=(roll_amount[0].item(), roll_amount[1].item()), dims=(2, 3))
            masked_inputs[mask] = shifted[mask]

            outputs = model(masked_inputs)

            flow_component = outputs['flow_component']
            noise_component = outputs['noise_component']

            mse = nn.MSELoss(reduction='none')
            n2v_loss = mse(flow_component[mask], batch_targets[mask]).mean()
            
            total_loss = loss_fn(flow_component, noise_component, batch_inputs, batch_targets, loss_parameters=loss_parameters)

            total_loss = total_loss + n2v_loss * n2v_weight

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            noise_loss = 0
            
            # Update running losses
            running_loss += total_loss
            running_flow_loss += total_loss.item()
            running_noise_loss += 0 #noise_loss.item()
            
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
        
        # Update history
        history['loss'].append(avg_loss)
        history['flow_loss'].append(avg_flow_loss)
        history['noise_loss'].append(avg_noise_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Noise Loss: {avg_noise_loss:.6f}")
        
        clear_output(wait=True)
        #visualize_progress(model, inputs[0:1], targets[0:1], epoch+1)
        random.seed(epoch)
        #random_n = random.randint(0, len(inputs)-1)
        #visualize_progress(model, inputs[random_n:random_n+1], targets[random_n:random_n+1], masked_tensor=masked_inputs[random_n:random_n+1].cpu().numpy(), epoch=epoch+1)
        random_idx = random.randint(0, batch_inputs.size(0)-1)  # Get a random index within the current batch
        visualize_progress(model, batch_inputs[random_idx:random_idx+1], batch_targets[random_idx:random_idx+1], 
                        masked_tensor=masked_inputs[random_idx:random_idx+1][0][0].cpu().numpy(), epoch=epoch+1)

        plt.close()

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
            checkpoint_path = "checkpoints/speckle_separation_model_best.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # save last
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),  # optional, but useful
                'history': history,
            }
        checkpoint_path = r"checkpoints/speckle_separation_model_last.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
    
    return model, history