from schemas.components.evaluate_baselines import evaluate_baseline
from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch

def evaluate_n2(image=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

    if image is None:

        train_loader, val_loader = get_loaders(15, 1, 50, 8)
        image = get_sample_image(val_loader, device)

    metrics = {}
    n2n_metrics = evaluate_baseline(image, "n2n")
    metrics["n2n"] = n2n_metrics
    n2s_metrics = evaluate_baseline(image, "n2s")
    metrics["n2s"] = n2s_metrics
    n2v_metrics = evaluate_baseline(image, "n2v")
    metrics["n2v"] = n2v_metrics

    return metrics