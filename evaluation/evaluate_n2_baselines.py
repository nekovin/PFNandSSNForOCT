from schemas.components.evaluate_baselines import evaluate_baseline, evaluate_ssm_constraint
from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch

def evaluate_n2(image=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

    if image is None:

        train_loader, val_loader = get_loaders(15, 1, 50, 8)
        image = get_sample_image(val_loader, device)

    metrics = {}
    n2n_metrics, n2n_denoised = evaluate_baseline(image, "n2n")
    metrics["n2n"] = n2n_metrics
    n2s_metrics, n2s_denoised = evaluate_baseline(image, "n2s")
    metrics["n2s"] = n2s_metrics
    n2v_metrics, n2v_denoised = evaluate_baseline(image, "n2v")
    metrics["n2v"] = n2v_metrics

    return metrics, (n2n_denoised, n2s_denoised, n2v_denoised)

def evaluate_n2_with_ssm(image=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

    if image is None:

        train_loader, val_loader = get_loaders(15, 1, 50, 8)
        image = get_sample_image(val_loader, device)

    metrics = {}
    n2n_metrics, n2n_denoised = evaluate_ssm_constraint(image, "n2n")
    metrics["n2n_ssm"] = n2n_metrics
    n2s_metrics, n2s_denoised = evaluate_ssm_constraint(image, "n2s")
    metrics["n2s_ssm"] = n2s_metrics
    n2v_metrics, n2v_denoised = evaluate_ssm_constraint(image, "n2v")
    metrics["n2v_ssm"] = n2v_metrics

    return metrics, (n2n_denoised, n2s_denoised, n2v_denoised)