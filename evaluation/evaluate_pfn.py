from models.prog import create_progressive_fusion_dynamic_unet
from models.prog_unet import ProgUNet
from models.prog_custom import ProgLargeUNet
from utils.postprocessing import normalize_image
from utils.evaluate import evaluate_oct_denoising
import matplotlib.pyplot as plt
import torch
from utils.config import get_config

def denoise_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image, n_targets=1, target_size=image.shape[-2:]) 
        denoised_image = outputs[0] 
    return denoised_image

from utils.evaluate import evaluate

def evaluate_progressssive_fusion_unet(image, reference, device, config_path=r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\pfn_config.yaml", prog_override_dict=None):

    config = get_config(config_path, prog_override_dict)

    eval_config = config['evaluation']

    #temp_checkpoint_path = eval_config['temp_checkpoint_path']

    base_checkpoint_path = eval_config['base_checkpoint_path']
    ablation = eval_config['ablation']
    model_name = eval_config['model_name']

    large = True #eval_config['large']

    #model = create_progressive_fusion_dynamic_unet().to(device)
    if model_name == "ProgUNet":
        if large:
            
            model = ProgLargeUNet(in_channels=1, out_channels=1).to(device)
            checkpoint_path = base_checkpoint_path + f"{ablation}/ProgUNet_{model}_best_checkpoint.pth"
        else:
            model = ProgUNet(in_channels=1, out_channels=1).to(device)
            checkpoint_path = base_checkpoint_path + f"{ablation}/ProgUNet_{model}_best_checkpoint.pth"
        print(f"Loading checkpoint from {checkpoint_path}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    metrics, denoised = evaluate(image, reference, model, "pfn")

    metrics['epochs'] = checkpoint['epoch']
    metrics['loss'] = checkpoint['val_loss']
    
    metrics['model'] = 'pfn'
    return metrics, denoised