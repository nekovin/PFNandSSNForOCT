from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stage2.mask import create_blind_spot_input_with_realistic_noise
from stage2.octa import compute_octa
from stage2.threshold import enhanced_differentiable_threshold_octa_torch
from stage2.vis import visualise_n2v, plot_loss
from stage2.utils import normalize_image_torch, load
from stage2.loss import octa_criterion

def process_batch(
        model, loader, criterion, threshold, octa_criterion, mask_ratio, alpha,
        optimizer=None,  # Add optimizer as optional parameter
        device='cuda',
        visualize=False
        ):
    
    # Set model mode based on whether we're training or evaluating
    if optimizer:  # If optimizer is provided, we're in training mode
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    
    context_manager = torch.no_grad() if not optimizer else nullcontext()
    
    with context_manager:
        for batch in loader:
            raw1, raw2, octa_calc, stage1_output = batch

            raw1 = raw1.to(device)
            raw2 = raw2.to(device)
            octa_calc = octa_calc.to(device)
            stage1_output = stage1_output.to(device)

            mask = torch.bernoulli(torch.full((raw1.size(0), 1, raw1.size(2), raw1.size(3)), 
                                            mask_ratio, device=device))

            blind1 = create_blind_spot_input_with_realistic_noise(raw1, mask).requires_grad_(True) #change based on what you want to do
            #blind1 = create_blind_spot_input_fast(raw1, mask).requires_grad_(True)
            #blind2 = create_blind_spot_input(raw1, mask).requires_grad_(True)
            blind2 = create_blind_spot_input_with_realistic_noise(raw2, mask).requires_grad_(True)
            
            # Zero gradients if we're training
            if optimizer:
                optimizer.zero_grad()

            outputs1 = model(blind1)
            outputs2 = model(blind2) #dont need to mask 2nd

            norm_outputs1 = normalize_image_torch(outputs1.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)
            norm_outputs2 = normalize_image_torch(outputs2.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)

            octa_output = compute_octa(norm_outputs1, norm_outputs2)

            if threshold > 0.0:
                #post_octa = differentiable_threshold_octa_torch(octa_output, raw1, threshold, smoothness=50.0) + 1e-8
                post_octa = enhanced_differentiable_threshold_octa_torch(octa_output, stage1_output, threshold, smoothness=2.0, enhancement_factor=1.5) + 1e-8
            else:
                post_octa = octa_output + 1e-8
            
            if optimizer:
                post_octa.retain_grad()
            
            n2v_loss1 = criterion(outputs1[mask > 0], raw1[mask > 0])
            n2v_loss2 = criterion(outputs2[mask > 0], raw2[mask > 0])

            octa_constraint_loss = octa_criterion(post_octa, stage1_output)

            loss = n2v_loss1 + n2v_loss2 + alpha * octa_constraint_loss
            
            # Backprop if we're training
            if optimizer:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if visualize:
                visualise_n2v(
                    raw1=raw1.cpu().detach().numpy(),
                    oct1=blind1.cpu().detach().numpy(),
                    oct2=blind2.cpu().detach().numpy(),
                    output1=outputs1.cpu().detach().numpy(),
                    output2=outputs2.cpu().detach().numpy(),
                    normalize_output1=norm_outputs1.cpu().detach().numpy(),
                    octa_from_outputs=octa_output.cpu().detach().numpy(),
                    thresholded_octa=post_octa.cpu().detach().numpy(),
                    stage1_output=stage1_output.cpu().detach().numpy()
                )
    
    return total_loss / len(loader)

def train_stage2(img_size, model, train_loader, val_loader, criterion, optimizer, 
                epochs=10, device='cuda', scratch=False, save_path=None, 
                mask_ratio=0.1, alpha=1, threshold=0.9, visualise=False, debug=False):

    if scratch:
        history = {'train_loss': [], 'val_loss': []}
        old_epoch = 0
    else:
        print(f"Loading model from {save_path}")
        model, optimizer, old_epoch, history = load(model, optimizer, save_path, scratch, device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        print("Training...")
        train_loss = process_batch(
            model=model, 
            loader=train_loader, 
            criterion=criterion, 
            threshold=threshold,
            octa_criterion=octa_criterion,
            mask_ratio=mask_ratio, 
            alpha=alpha,
            optimizer=optimizer,  # Pass optimizer to indicate we're training
            device=device,
            visualize=False
        )
        history['train_loss'].append(train_loss)
        print(f"Training loss: {train_loss:.6f}")
        
        print("Validating...")
        val_loss = process_batch(
            model=model, 
            loader=val_loader, 
            criterion=criterion, 
            threshold=threshold,
            octa_criterion=octa_criterion,
            mask_ratio=mask_ratio, 
            alpha=alpha,
            optimizer=None,  # mo optimizer means we're evaluating
            device=device,
            visualize=visualise
        )
        history['val_loss'].append(val_loss)
        print(f"Validation loss: {val_loss:.6f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            print(f"New best model! Val loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
            best_val_loss = val_loss
            
            # Save checkpoint
            try:
                torch.save({
                    'epoch': epoch + old_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history
                }, save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Visualize loss curves if requested
        if visualise:
            plot_loss(history['train_loss'], history['val_loss'])
        
        print("-" * 50)
    
    return model, history