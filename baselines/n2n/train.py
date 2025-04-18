import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import torch
from data_loading import get_loaders
from models.unet import UNet
from torch import nn

def process_batch():
    pass

def train(model, train_loader, optimizer=None, criterion=None, epochs=100, batch_size=8, lr=0.001, save_dir='n2n/checkpoints',device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_imgs, target_imgs) in enumerate(train_loader):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            outputs = model(input_imgs)
            loss = criterion(outputs, target_imgs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss
            }, os.path.join(save_dir, f'n2n_model_epoch_{epoch+1}.pth'))
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'n2n_final.pth'))
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")
    
    return model

def train_noise2noise():

    train_config = {
        "model": UNet,
        'batch_size': 8,
        'epochs': 5,
        'learning_rate': 1e-4,
        'criterion': nn.MSELoss(),
        'optimizer': optim.Adam,
    }

    train_loader = get_loaders(n_patients=2, n_images_per_patient=50, batch_size=train_config['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = train_config['model'](in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    model = train(
        model,
        train_loader,
        optimizer=optimizer,
        criterion=train_config['criterion'],
        epochs=train_config['epochs'], 
        batch_size=train_config['batch_size'], 
        lr=train_config['learning_rate'],
        save_dir='n2n/checkpoints',
        device=device)

if __name__ == "__main__":
    train_noise2noise()