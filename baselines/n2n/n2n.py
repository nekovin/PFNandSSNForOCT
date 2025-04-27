
import torch
from models.unet_2 import UNet2

def load_model(config):
    eval_config = config['eval']
    checkpoint_path = eval_config['checkpoint_path']
    model = eval_config['model']
    device = eval_config['device']

    #print(checkpoint_path)

    # checkpoint_path=rf"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm\checkpoints\SpeckleSeparationUNetAttention_custom_loss_best.pth", model="UNet2", device='cuda'
    if model == "UNet2":
        model = UNet2(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(checkpoint_path + f'{model}_best_checkpoint.pth', map_location=device)
    print("Loading model from checkpoint...")
    print(checkpoint_path + f'{model}_best_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model