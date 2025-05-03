from utils.data_loading import get_loaders
from models.unet import UNet
from models.unet_2 import UNet2
from models.ssm_attention import SpeckleSeparationUNetAttention
import torch.optim as optim
import torch
from schemas.baselines.n2n import train_n2n
from schemas.baselines.n2v import train_n2v
from schemas.baselines.n2s import train_n2s

def train(config, method, ssm):

    train_config = config['training']
    if method is None:
        raise ValueError("Method must be specified either in config or function.")

    print(f"Training method: {method}")

    n_patients = train_config['n_patients']
    n_images_per_patient = train_config['n_images_per_patient']
    batch_size = train_config['batch_size']
    start = train_config['start_patient'] if train_config['start_patient'] else 1

    train_loader, val_loader = get_loaders(start, n_patients, n_images_per_patient, batch_size)

    baselines_checkpoint_path = train_config['baselines_checkpoint_path']
    
    model = train_config['model']

    checkpoint_path = baselines_checkpoint_path + rf"{method}_"

    if config['speckle_module']['use'] is True:
        checkpoint_path = checkpoint_path + rf"{model}_ssm"
    else:
        checkpoint_path = checkpoint_path + rf"{model}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_config['model'] == 'UNet':
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif train_config['model'] == 'UNet2':
        model = UNet2(in_channels=1, out_channels=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    visualise = train_config['visualise']

    alpha = 1
    starting_epoch = 0
    best_val_loss = float('inf')

    save = train_config['save']

    if config['speckle_module']['use'] is True or ssm:
        speckle_module = SpeckleSeparationUNetAttention(input_channels=1, feature_dim=32).to(device)
        try:
            print("Loading ssm model from checkpoint...")
            ssm_checkpoint_path = rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\SpeckleSeparationUNetAttention_custom_loss_best.pth"
            ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
            speckle_module.load_state_dict(ssm_checkpoint['model_state_dict'])
            speckle_module.to(device)
            alpha = config['speckle_module']['alpha']
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e 
    else:
        speckle_module = None

    if train_config['load']:
        try:
            checkpoint = torch.load(checkpoint_path + f'_best_checkpoint.pth', map_location=device)
            print("Loading model from checkpoint...")
            print(checkpoint_path + f'_best_checkpoint.pth')
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully")
            print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['best_val_loss']}")
            starting_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
    
    if train_config['train']:
        if method == "n2n":
            model = train_n2n(
                model,
                train_loader,
                val_loader,
                optimizer=optimizer,
                criterion=train_config['criterion'],
                starting_epoch=starting_epoch,
                epochs=train_config['epochs'], 
                batch_size=train_config['batch_size'], 
                lr=train_config['learning_rate'],
                best_val_loss=best_val_loss,
                checkpoint_path=checkpoint_path,
                device=device,
                visualise=visualise,
                speckle_module=speckle_module,
                alpha=alpha,
                save=save)
            
        elif method == "n2v":
            model = train_n2v(
                model,
                train_loader,
                val_loader,
                optimizer=optimizer,
                criterion=train_config['criterion'],
                starting_epoch=starting_epoch,
                epochs=train_config['epochs'], 
                batch_size=train_config['batch_size'], 
                lr=train_config['learning_rate'],
                best_val_loss=best_val_loss,
                checkpoint_path=checkpoint_path,
                device=device,
                visualise=visualise,
                speckle_module=speckle_module,
                alpha=alpha,
                save=save,
                method=method,
                octa_criterion=False,
                threshold=train_config['threshold'],
                mask_ratio=train_config['mask_ratio'])
        elif method == "n2s":
            model = train_n2s(
                model,
                train_loader,
                val_loader,
                optimizer=optimizer,
                criterion=train_config['criterion'],
                starting_epoch=starting_epoch,
                epochs=train_config['epochs'], 
                batch_size=train_config['batch_size'], 
                lr=train_config['learning_rate'],
                best_val_loss=best_val_loss,
                checkpoint_path=checkpoint_path,
                device=device,
                visualise=visualise,
                speckle_module=speckle_module,
                alpha=alpha,
                save=save)

            
    return model