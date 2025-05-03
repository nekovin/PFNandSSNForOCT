from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch
from utils.metrics import display_metrics

from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
from evaluation.evaluate_n2_baselines import evaluate_n2

# import rando
import random

def main():
    random.seed(42)
    random_number = random.randint(2, 20)
    train_loader, val_loader = get_loaders(random_number, 1, 50, 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = get_sample_image(val_loader, device)

    metrics = {}
    n2_metrics = evaluate_n2(image)
    metrics["n2n"] = n2_metrics['n2n']
    metrics["n2s"] = n2_metrics['n2s']
    metrics["n2v"] = n2_metrics['n2v']

    prog_metrics = evaluate_progressssive_fusion_unet(image, device)
    metrics["pfn"] = prog_metrics

    metrics_df = display_metrics(metrics)

    print("Metrics for N2N:")
    print(metrics_df)

if __name__ == "__main__":
    main()