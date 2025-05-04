from models.prog import create_progressive_fusion_dynamic_unet
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

def evaluate_progressssive_fusion_unet(image, device):

    config = get_config(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\pfn_config.yaml")

    eval_config = config['evaluation']

    temp_checkpoint_path = eval_config['temp_checkpoint_path']

    model = create_progressive_fusion_dynamic_unet().to(device)
    checkpoint = torch.load(temp_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    denoised = denoise_image(model, image, device)
    denoised = denoised.cpu().numpy()[0][0]
    sample_image = image.cpu().numpy()[0][0]
    #denoised = normalize_image(denoised)
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #ax[0].imshow(sample_image, cmap='gray')
    #ax[1].imshow(denoised, cmap='gray')
    metrics = evaluate_oct_denoising(sample_image, denoised)
    metrics['epochs'] = checkpoint['epoch']
    metrics['loss'] = checkpoint['val_loss']
    
    metrics['model'] = 'pfn'
    return metrics, denoised