import torch
from dotenv import load_dotenv
import os
import pandas as pd

from utils.postprocessing import normalize_image
from utils.metrics import evaluate_oct_denoising

import matplotlib.pyplot as plt
import numpy as np

def get_sample_image(dataloader, device):
    sample = next(iter(dataloader))
    image = sample[0].to(device)
    return image

def plot_sample(image, denoised, model, method):
    

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    fig.suptitle(f"Model: {model.__class__.__name__} - {method}", fontsize=16)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    normalised_denoised = normalize_image(denoised)

    ax[1].imshow(denoised, cmap='gray')
    ax[1].set_title('Denoised Image')
    ax[1].axis('off')

    ax[2].imshow(normalised_denoised, cmap='gray')
    ax[2].set_title('Normalized Denoised Image')
    ax[2].axis('off')

    plt.show()
    

def denoise_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        denoised_image = model(image)
    return denoised_image



load_dotenv()

device = os.getenv("DEVICE")
device

def evaluate(image, model, method):
    denoised = denoise_image(model, image, device='cuda')[0][0]
    sample_image = image.cpu().numpy()[0][0]
    denoised = denoised.cpu().numpy()
    #normalised_denoised = normalize_image(denoised)
    #plot_sample(sample_image, denoised, model, method)
    if len(denoised.shape) == 3 and denoised.shape[0] == 1: # this is because my pfn model retusn a 3d tensor
        denoised = denoised[-1]
    metrics = evaluate_oct_denoising(sample_image, denoised)

    return metrics, denoised