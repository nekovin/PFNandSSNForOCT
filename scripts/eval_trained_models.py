from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch
from utils.metrics import display_metrics

from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
from evaluation.evaluate_n2_baselines import evaluate_n2, evaluate_n2_with_ssm

# import rando
import random
import matplotlib.pyplot as plt

def plot_images(images, metrics_df):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 10))
    for i, (key, image) in enumerate(images.items()):
        ax[i].imshow(image, cmap='gray')
        ax[i].set_title(key)
        ax[i].axis('off')
    plt.show()

def main():
    random.seed(42)
    random_number = random.randint(2, 20)
    train_loader, val_loader = get_loaders(random_number, 1, 50, 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = get_sample_image(val_loader, device)

    metrics = {}
    n2_metrics, n2_base_images = evaluate_n2(image)
    metrics["n2n"] = n2_metrics['n2n']
    metrics["n2s"] = n2_metrics['n2s']
    metrics["n2v"] = n2_metrics['n2v']

    n2_metrics, n2_ssm_images = evaluate_n2_with_ssm(image)
    metrics["n2n_ssm"] = n2_metrics['n2n_ssm']
    metrics["n2s_ssm"] = n2_metrics['n2s_ssm']
    metrics["n2v_ssm"] = n2_metrics['n2v_ssm']

    prog_metrics, prog_image = evaluate_progressssive_fusion_unet(image, device)
    metrics["pfn"] = prog_metrics

    metrics_df = display_metrics(metrics)

    images = {
        "original": image.cpu().numpy()[0][0],
        "n2n": n2_base_images[0],
        "n2s": n2_base_images[1],
        "n2v": n2_base_images[2],
        "n2n_ssm": n2_ssm_images[0],
        "n2s_ssm": n2_ssm_images[1],
        "n2v_ssm": n2_ssm_images[2],
        "pfn": prog_image
    }
    
    plot_images(images, metrics_df)

if __name__ == "__main__":
    main()