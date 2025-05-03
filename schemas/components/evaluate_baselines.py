from utils.config import get_config
from utils.evaluate import evaluate
import torch
from models.unet_2 import UNet2

def load_model(config, verbose=False):
    use_speckle = config['speckle_module']['use']
    eval_config = config['eval']
    base_checkpoint_path = eval_config['baselines_checkpoint_path']
    method = eval_config['method']
    model = eval_config['model']
    if use_speckle:
        checkpoint_path = base_checkpoint_path + rf"{method}_{model}_ssm_best_checkpoint.pth"
    else:
        checkpoint_path = base_checkpoint_path + rf"{method}_{model}_best_checkpoint.pth"
    
    device = eval_config['device']
    if model == "UNet2":
        model = UNet2(in_channels=1, out_channels=1).to(device)

    if verbose:
        print(f"Loading {model} model...")
        print(f"Checkpoint path: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_baseline(image, method, config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"):
    
    config = get_config(config_path)
    
    config['eval']['method'] = method

    model = load_model(config)

    return evaluate(image, model, method)

def evaluate_ssm_constraint(image, method, config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"):
    
    config = get_config(config_path)
    
    config['speckle_module']['use'] = True
    config['eval']['method'] = method

    model = load_model(config)

    return evaluate(image, model, method)