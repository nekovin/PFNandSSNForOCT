import torch
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")

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
        # Create mask for non-background (foreground) pixels.
        foreground_mask = t_img > 0.01
        
        # Check if any foreground pixel is found.
        if torch.any(foreground_mask):
            fg_values = t_img[foreground_mask]
            fg_min = fg_values.min()
            fg_max = fg_values.max()
            
            # Normalize only if there is a valid range.
            if fg_max > fg_min:
                # Use torch.where to selectively update foreground pixels.
                t_img = torch.where(foreground_mask, (t_img - fg_min) / (fg_max - fg_min), t_img)
        
        # Force background (pixels < 0.01) to be 0
        t_img = torch.where(t_img < 0.01, torch.zeros_like(t_img), t_img)
    return t_img

def freeze():
    '''
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=1e-3
        )
        '''
    pass

def load(model, optimizer, save_path, scratch, device):

    print(f"Received save_path: {save_path}")
    print(f"Current working directory: {os.getcwd()}")

    checkpoints_dir = os.path.dirname(save_path)
    if not os.path.exists(checkpoints_dir) and not scratch:
        os.makedirs(checkpoints_dir, exist_ok=True)

    if scratch:
        try:
            checkpoint = torch.load(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v\checkpoints\stage1.pth")
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch+1}")
            return model, optimizer, old_epoch, history
        except:
            raise ValueError("Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path.")
    else:
        try:
            checkpoint = torch.load(save_path)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch+1}")
            return model, optimizer, old_epoch, history
        except:
            raise ValueError("Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path.")
    
def check_performance(val_loss, best_val_loss, model, optimizer, epoch, save_path, history, old_epoch, avg_train_loss):
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

