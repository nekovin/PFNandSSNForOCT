from schemas.components.evaluate_baselines import evaluate_baseline, evaluate_ssm_constraint
from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch

def evaluate_n2(metrics, denoised_images, config_path, eval_override, image=None, reference=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

    if image is None:

        train_loader, val_loader = get_loaders(15, 1, 50, 8)
        image = get_sample_image(val_loader, device)
        
    images = []
    try:
        print("Evaluating n2n")
        n2n_metrics, n2n_denoised = evaluate_baseline(image, reference, "n2n", config_path, eval_override)
        if n2n_metrics is None:
            raise ValueError("Metrics for n2n are None")
        else:
            #print("Adding n2n metrics")
            metrics["n2n"] = n2n_metrics
            images.append(n2n_denoised)
            denoised_images['n2n'] = n2n_denoised
    except Exception as e:
        print(f"Error evaluating n2n: {e}")

    try:
        n2s_metrics, n2s_denoised = evaluate_baseline(image, reference, "n2s", config_path, eval_override)
        if n2s_metrics is None:
            raise ValueError("Metrics for n2s are None")
        else:
            #print("Adding n2s metrics")
            metrics["n2s"] = n2s_metrics
            images.append(n2s_denoised)
            denoised_images['n2s'] = n2s_denoised
    except Exception as e:
        print(f"Error evaluating n2s: {e}")

    try:
        n2v_metrics, n2v_denoised = evaluate_baseline(image, reference, "n2v", config_path, eval_override)
        if n2v_metrics is None:
            raise ValueError("Metrics for n2v are None")
        else:
            #print("Adding n2v metrics")
            metrics["n2v"] = n2v_metrics
            images.append(n2v_denoised)
            denoised_images['n2v'] = n2v_denoised
    except Exception as e:
        print(f"Error evaluating n2v: {e}")

    return metrics, denoised_images

def evaluate_n2_with_ssm(metrics, denoised_images, config_path, eval_override, image=None, reference=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

    if image is None:

        train_loader, val_loader = get_loaders(15, 1, 50, 8)
        image = get_sample_image(val_loader, device)

    images = []
    try:
        n2n_metrics, n2n_denoised = evaluate_ssm_constraint(image, reference, "n2n", config_path, eval_override)
        if n2n_metrics is None:
            raise ValueError("Metrics for n2s are None")
        else:
            #print("Adding n2n metrics")
            metrics["n2n_ssm"] = n2n_metrics
            images.append(n2n_denoised)
            denoised_images['n2n_ssm'] = n2n_denoised
    except Exception as e:
        print(f"Error evaluating n2n with SSM: {e}")
    
    try:
        n2s_metrics, n2s_denoised = evaluate_ssm_constraint(image, reference, "n2s", config_path, eval_override)
        if n2s_metrics is None:
            raise ValueError("Metrics for n2s are None")
        else:
            #print("Adding n2s metrics")
            metrics["n2s_ssm"] = n2s_metrics
            images.append(n2s_denoised)
            denoised_images['n2s_ssm'] = n2s_denoised
    except Exception as e:
        print(f"Error evaluating n2s with SSM: {e}")
    
    try:
        n2v_metrics, n2v_denoised = evaluate_ssm_constraint(image, reference, "n2v", config_path, eval_override)
        if n2v_metrics is None:
            raise ValueError("Metrics for n2v are None")
        else:
            #print("Adding n2v metrics")
            metrics["n2v_ssm"] = n2v_metrics
            images.append(n2v_denoised)
            denoised_images['n2v_ssm'] = n2v_denoised
    except Exception as e:
        print(f"Error evaluating n2v with SSM: {e}")

    return metrics, denoised_images