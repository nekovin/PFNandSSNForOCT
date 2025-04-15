import os 
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.colors import NoNorm

from .utils import create_blind_spot_input_fast, plot_loss


def run_batch():
    pass

def train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda', scratch=False, save_path=None, mask_ratio = 0.1, visualise=False):

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if scratch:
        model = model
        history = {'train_loss': [], 'val_loss': []}
        old_epoch = 0
        print("Training from scratch")
    else:
        try:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch+1}")
        except:
            print("No model found, training from scratch")
            model = model
            history = {'train_loss': [], 'val_loss': []}
            old_epoch = 0

    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        if torch.cuda.is_available():
            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Training phase
        model.train()
        running_loss = 0.0

        
        for batch_idx, octa in enumerate(train_loader):
            octa = octa.to(device)

            mask = torch.bernoulli(torch.full((octa.size(0), 1, octa.size(2), octa.size(3)), 
                                            mask_ratio, device=device))
            
            blind_octa = create_blind_spot_input_fast(octa, mask)

            optimizer.zero_grad()

            outputs = model(blind_octa)

            #outputs = normalize_data(outputs, octa)

            loss = criterion(outputs, octa)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1} finished")
        
        avg_train_loss = running_loss / len(train_loader)
        
        print("Validating")
        val_loss = validate_n2v(model, val_loader, criterion, mask_ratio, device, visualise=visualise)
        print("Validation finished")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        if visualise:
            plot_loss(history['train_loss'], history['val_loss'])
        
        if val_loss < best_val_loss:
            print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch+1}")
            best_val_loss = val_loss
            try:
                torch.save({
                    'epoch': epoch + old_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, save_path)
            
            except:
                print("Err")
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        print("-" * 50)

    return model, history



def visualise_n2v(blind_input, target_img, output, mask=None):
    """
    Visualize the N2V process with mask overlay
    
    Args:
        blind_input: Input with blind spots
        target_img: Target noisy image
        output: Model prediction
        mask: Binary mask showing pixel positions for N2V
    """
    # Normalize output to match input scale
    output = torch.from_numpy(output).float()
    #output = normalize_data(output)
    
    clear_output(wait=True)
    
    # If mask is provided, show 4 images including mask
    if mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot blind spot input
        axes[0].imshow(blind_input.squeeze(), cmap='gray', norm=NoNorm())
        axes[0].axis('off')
        axes[0].set_title('Blind-spot Input')
        
        # Plot model output
        axes[1].imshow(output.squeeze(), cmap='gray', norm=NoNorm())
        axes[1].axis('off')
        axes[1].set_title('Output Image')
        
        # Plot target image
        axes[2].imshow(target_img.squeeze(), cmap='gray', norm=NoNorm())
        axes[2].axis('off')
        axes[2].set_title('Target Noisy Image')
    
    # Otherwise use original 3-image layout
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes[0].imshow(blind_input.squeeze(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Blind-spot Input')
        
        axes[1].imshow(output.squeeze(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Output Image')
        
        axes[2].imshow(target_img.squeeze(), cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Target Noisy Image')
    
    plt.tight_layout()
    plt.show()

def validate_n2v(model, val_loader, criterion, mask_ratio, device='cuda', visualise=False):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for octa in val_loader:
            octa = octa.to(device)

            mask = torch.bernoulli(torch.full((octa.size(0), 1, octa.size(2), octa.size(3)), 
                                            mask_ratio, device=device))
            
            blind_octa = create_blind_spot_input_fast(octa, mask)
            
            outputs = model(blind_octa)
                
            #outputs = normalize_data(outputs) 

            loss = criterion(outputs, octa)
            
            total_loss += loss.item()
            
            if visualise:
                visualise_n2v(
                    blind_octa.cpu().detach().numpy(),
                    octa.cpu().detach().numpy(),
                    outputs.cpu().detach().numpy(),
                )
    
    return total_loss / len(val_loader)