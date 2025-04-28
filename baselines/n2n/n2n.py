
import torch
from models.unet_2 import UNet2

def load_model(config):
    eval_config = config['eval']
    base_checkpoint_path = eval_config['baselines_checkpoint_path']
    method = eval_config['method']
    model = eval_config['model']
    checkpoint_path = base_checkpoint_path + rf"{method}/checkpoints/{model}_best_checkpoint.pth"
    
    device = eval_config['device']
    if model == "UNet2":
        model = UNet2(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Loading model from checkpoint...")
    print(checkpoint_path + rf'/{model}_best_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model