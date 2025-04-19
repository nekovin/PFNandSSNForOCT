import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import torch
from data_loading import get_loaders
from models.unet import UNet
from torch import nn
from visualise import plot_images

def process_batch(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise):
    epoch_loss = 0
    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
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
    
def process_batch(data_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module):
    epoch_loss = 0
    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        if speckle_module is not None:
            processed_inputs = speckle_module(input_imgs)
            processed_inputs = processed_inputs['flow_component']
            outputs = model(processed_inputs)
        else:
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
            
            if speckle_module is not None:
                images = [
                    input_imgs[0][0].cpu().numpy(), 
                    processed_inputs[0][0].cpu().detach().numpy(),
                    target_imgs[0][0].cpu().numpy(), 
                    outputs[0][0].cpu().detach().numpy()
                ]
            else:
                images = [
                    input_imgs[0][0].cpu().numpy(), 
                    target_imgs[0][0].cpu().numpy(), 
                    outputs[0][0].cpu().detach().numpy()
                ]
            plot_images(images)

    return epoch_loss

def train(model, train_loader, val_loader, optimizer=None, criterion=None, epochs=100, batch_size=8, lr=0.001, save_dir='n2n/checkpoints',device='cuda', visualise=False, speckle_module=None):
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        epoch_loss = process_batch(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module)
        
        avg_epoch_loss = epoch_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss
            }, os.path.join(save_dir, f'n2n_model_epoch_{epoch+1}.pth'))
        
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss
        }, os.path.join(save_dir, f'n2n_model_last.pth'))
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")
    
    return model

from ssm.models.ssm_attention import SpeckleSeparationUNetAttention

def train_noise2noise(config):

    train_config = config['training']

    n_patients = train_config['n_patients']
    n_images_per_patient = train_config['n_images_per_patient']
    batch_size = train_config['batch_size']

    train_loader, val_loader = get_loaders(n_patients, n_images_per_patient, batch_size)

    checkpoint_path = train_config['checkpoint_path'] if train_config['checkpoint_path'] else None

    save_dir = train_config['save_dir'] if train_config['save_dir'] else 'n2n/checkpoints'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if train_config['model'] == 'UNet':
        model = UNet(in_channels=1, out_channels=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    visualise = train_config['visualise']

    if config['speckle_module']['use'] is True:
        speckle_module = SpeckleSeparationUNetAttention(input_channels=1, feature_dim=32).to(device)
        try:
            print("Loading model from checkpoint...")
            ssm_checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\{repr(speckle_module)}_best.pth"
            ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
            speckle_module.load_state_dict(ssm_checkpoint['model_state_dict'])
            speckle_module.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e 
    else:
        speckle_module = None

    if train_config['load']:
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model loaded successfully")

    if train_config['train']:

        model = train(
            model,
            train_loader,
            val_loader,
            optimizer=optimizer,
            criterion=train_config['criterion'],
            epochs=train_config['epochs'], 
            batch_size=train_config['batch_size'], 
            lr=train_config['learning_rate'],
            save_dir=save_dir,
            device=device,
            visualise=visualise,
            speckle_module=speckle_module)

if __name__ == "__main__":
    train_noise2noise()