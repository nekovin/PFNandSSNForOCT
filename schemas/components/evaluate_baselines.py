from utils.config import get_config
from utils.evaluate import evaluate
import torch
from models.unet_2 import UNet2
from models.unet import UNet
from models.large_unet import LargeUNet, LargeUNetAttention

def load_model(config, verbose=False):
    use_speckle = config['speckle_module']['use']
    eval_config = config['eval']
    base_checkpoint_path = eval_config['baselines_checkpoint_path']
    method = eval_config['method']
    model = eval_config['model']
    if use_speckle:
        best = config['speckle_module']['best']
        if best:
            checkpoint_path = base_checkpoint_path + rf"{method}_{model}_ssm_best_checkpoint.pth"
        else:
            checkpoint_path = base_checkpoint_path + rf"{method}_{model}_ssm_last_checkpoint.pth"
    else:
        checkpoint_path = base_checkpoint_path + rf"{method}_{model}_best_checkpoint.pth"
    
    device = eval_config['device']
    
    if model == "UNet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    if model == "UNet2":
        model = UNet2(in_channels=1, out_channels=1).to(device)
    if model == "LargeUNet":
        model = LargeUNet(in_channels=1, out_channels=1).to(device)
    if model == "LargeUNetAttention":
        model = LargeUNetAttention(in_channels=1, out_channels=1).to(device)

    print(f"Loading model: {model}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    #checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = load_checkpoint(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    if verbose:
        print(f"Loading {model} model...")
        print(f"Checkpoint path: {checkpoint_path}")
        for key, value in checkpoint.items():
            #if key != 'model_state_dict' or key != 'optimizer_state_dict':
                #print(f"{key}: {value}")
            if key == 'epoch':
                print(f"Epoch: {value}")
            if key == 'best_val_loss':
                print(f"Loss: {value}")
        print(f"Model loaded successfully")
    return model

def load_checkpoint(config):
    eval_config = config['eval']
    base_checkpoint_path = eval_config['baselines_checkpoint_path']
    ablation = eval_config['ablation']
    method = eval_config['method']
    model = eval_config['model']
    if config['speckle_module']['use']:
        best = config['speckle_module']['best']
        if best:
            checkpoint_path = base_checkpoint_path + ablation + rf"/{method}_{model}_ssm_best_checkpoint.pth"
        else:
            checkpoint_path = base_checkpoint_path + ablation + rf"/{method}_{model}_ssm_last_checkpoint.pth"
    else:
        checkpoint_path = base_checkpoint_path + ablation + rf"/{method}_{model}_best_checkpoint.pth"
    
    print(f"Checkpoint path: {checkpoint_path}")
    
    device = eval_config['device']
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    return checkpoint

def evaluate_baseline(image, reference, method, config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml", override_config = None):
    
    config = get_config(config_path, override_config)
    
    config['eval']['method'] = method
    
    verbose = config['eval']['verbose']

    exclude = config['eval']['exclude'][method]

    if exclude:
        print(f"Method {method} is excluded from evaluation.")
        return None, None

    model = load_model(config, verbose)
    checkpoint = load_checkpoint(config)
    
    metrics, denoised = evaluate(image, reference, model, method)

    metrics['epochs'] = checkpoint['epoch']
    metrics['loss'] = checkpoint['best_val_loss']
    metrics['model'] = str(model)

    return metrics, denoised

def evaluate_ssm_constraint(image, reference, method, config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml", override_dict = None):
    
    config = get_config(config_path, override_dict)

    exclude = config['eval']['exclude'][method]
    if exclude:
        return None, None
    
    config['speckle_module']['use'] = True
    config['eval']['method'] = method
    verbose = config['eval']['verbose']

    model = load_model(config, verbose)
    checkpoint = load_checkpoint(config)

    metrics, denoised = evaluate(image, reference, model, method)

    metrics['epochs'] = checkpoint['epoch']
    metrics['loss'] = checkpoint['best_val_loss']
    metrics['model'] = str(model)

    return metrics, denoised