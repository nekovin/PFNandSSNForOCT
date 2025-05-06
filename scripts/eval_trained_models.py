from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch
from utils.metrics import display_metrics, display_grouped_metrics

from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
from evaluation.evaluate_n2_baselines import evaluate_n2, evaluate_n2_with_ssm

# import rando
import random
import matplotlib.pyplot as plt

def plot_images(images, metrics_df):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for i, (key, image) in enumerate(images.items()):
        #ax[i].imshow(image, cmap='gray')
        #ax[i].set_title(key)
        #ax[i].axis('off')
        ax[i // 4, i % 4].imshow(image, cmap='gray')
        ax[i // 4, i % 4].set_title(key)
        ax[i // 4, i % 4].axis('off')
        
    plt.show()

def main():
    #random.seed(42)
    start = random.randint(30, 35)
    train_loader, val_loader = get_loaders(start, 1, 50, 8)

    print("Loading data...")
    print("Patient ID: ", start)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = get_sample_image(val_loader, device)

    metrics = {}
    denoised_images = {
        "original": image.cpu().numpy()[0][0],
    }

    metrics, denoised_images = evaluate_n2(metrics, denoised_images, image)

    prog_metrics, prog_image = evaluate_progressssive_fusion_unet(image, device)
    metrics["pfn"] = prog_metrics
    denoised_images["pfn"] = prog_image

    metrics, denoised_images = evaluate_n2_with_ssm(metrics, denoised_images, image)


    metrics_df = display_metrics(metrics)
    
    display_grouped_metrics(metrics)
    
    plot_images(denoised_images, metrics_df)

if __name__ == "__main__":
    main()