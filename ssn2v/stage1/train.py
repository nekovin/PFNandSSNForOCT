import os 
import torch

from .utils import create_blind_spot_input_fast, plot_loss, visualise_n2v, save_model


def run_batch(loader, model, criterion, optimizer, device, mask_ratio, visualise):


    running_loss = 0.0

    for batch_idx, octa in enumerate(loader):
        octa = octa.to(device)

        mask = torch.bernoulli(torch.full((octa.size(0), 1, octa.size(2), octa.size(3)), 
                                        mask_ratio, device=device))
        
        blind_octa = create_blind_spot_input_fast(octa, mask)

        if model.training:
            optimizer.zero_grad()

        outputs = model(blind_octa)

        #outputs = normalize_data(outputs, octa)

        loss = criterion(outputs, octa)
        
        if model.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        running_loss += loss.item()

        if visualise:
            visualise_n2v(
                blind_octa.cpu().detach().numpy(),
                octa.cpu().detach().numpy(),
                outputs.cpu().detach().numpy(),
            )

        return running_loss / len(loader)

def train_stage1(img_size, model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cuda', scratch=False, save_path=None, mask_ratio = 0.1, visualise=False):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

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

        model.train()

        train_loss = run_batch(train_loader, model, criterion, optimizer, device, mask_ratio, visualise)
        
        avg_train_loss = train_loss / len(train_loader)

        val_loss = validate_n2v(model, val_loader, criterion, mask_ratio, device, visualise)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        if visualise:
            plot_loss(history['train_loss'], history['val_loss'])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                save_model(model, optimizer, epoch, avg_train_loss, val_loss, history, save_path)
            except:
                print("Err")
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        print("-" * 50)

    return model, history

def validate_n2v(model, val_loader, criterion, mask_ratio, device='cuda', visualise=False):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for octa in val_loader:
            val_loss = run_batch(val_loader, model, criterion, None, device, mask_ratio, visualise)
    
    return val_loss / len(val_loader)