import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")
from stage1.utils.utils import normalize_image, get_stage1_loaders, get_unet_model, normalize_data

from stage1.utils.utils import plot_loss
import torch
import time
import matplotlib.pyplot as plt
from stage1.validation import validate_n2v

import random
import numpy as np

from torch.utils.data import Dataset
from models.enhanced_n2v_unet import get_e_n2n_unet_model
from models.blind_n2v_unet import get_blind_n2v_unet_model
from models.n2v_unet import get_n2n_unet_model

def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    #noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    return blind_input


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


def process_stage1(flow_masks, epochs=10, stage1_path='checkpoints/stage1.pth', visualise=False, scratch=False, mask_ratio=0.5):

    img_size = 256

    normalised_flow_masks = [normalize_image(flow_mask) for flow_mask in flow_masks]

    train_loader, val_loader, test_loader = get_stage1_loaders(normalised_flow_masks, img_size)

    #device, model, criterion, optimizer = get_n2n_unet_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_e_n2n_unet_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    model, history = train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs, device, scratch, save_path='checkpoints/stage1.pth', mask_ratio=mask_ratio, visualise=visualise)

    return model, history, train_loader, val_loader, test_loader