import matplotlib.pyplot as plt
from ssm.evaluation import evaluate_baseline, evaluate_ssm_constraint, evaluate_progressssive_fusion_unet
from ssm.utils import load_sdoct_dataset, display_metrics, display_grouped_metrics
from tqdm import tqdm
import torch
import os
import random
from ssm.utils.config import get_config

def main():

    
    config_path = os.getenv("N2_CONFIG_PATH")

    config = get_config(config_path)

    n_patients = config['training']['n_patients']
    
    override_config = {
        "eval" : {
            "ablation": f"patient_count/{n_patients}_patients",
            "n_patients" : n_patients
            }
        }

    all_metrics = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    dataset = load_sdoct_dataset(sdoct_path)

    # random sample
    sample = random.choice(list(dataset.keys()))
    raw_image = dataset[sample]["raw"]
    reference = dataset[sample]["avg"][0][0]


    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
    ax[0].set_title("Raw Image")
    ax[1].imshow(reference.cpu().numpy(), cmap="gray")
    ax[1].set_title("Reference Image")
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    metrics = {}
    metrics_last = {}

    best_images = []
    try:    
        n2n_metrics, n2n_denoised = evaluate_baseline(raw_image, reference, "n2v",config_path, override_config=override_config)
        metrics['n2v'] = n2n_metrics
        all_metrics['n2v'] = n2n_metrics
        ax[0][0].imshow(n2n_denoised, cmap="gray")
        ax[0][0].set_title("N2N Denoised")
    except Exception as e:
        print(f"Error evaluating n2n: {e}")
        n2n_metrics = None
        n2n_denoised = None
    try:
        n2n_ssm_metrics, n2n_ssm_denoised = evaluate_ssm_constraint(raw_image, reference, "n2v", config_path, override_config=override_config)
        metrics['n2v_ssm'] = n2n_ssm_metrics
        ax[0][1].imshow(n2n_ssm_denoised, cmap="gray")
        ax[0][1].set_title("N2N SSM Denoised")
    except Exception as e:
        print(f"Error evaluating n2n ssm: {e}")
        n2n_ssm_metrics = None
        n2n_ssm_denoised = None

    last_images = []
    try:
        n2n_metrics_last, n2n_denoised_last = evaluate_baseline(raw_image, reference, "n2v",config_path, override_config=override_config, last=True)
        metrics_last['n2v'] = n2n_metrics_last
        all_metrics['n2v_last'] = n2n_metrics_last
        ax[1][0].imshow(n2n_denoised_last, cmap="gray")
        ax[1][0].set_title("N2N Denoised Last")
    except Exception as e:
        print(f"Error evaluating n2n last: {e}")
        n2n_metrics_last = None
        n2n_denoised_last = None
    try:
        n2n_ssm_metrics_last, n2n_ssm_denoised_last = evaluate_ssm_constraint(raw_image, reference, "n2v", config_path, override_config=override_config, last=True)
        all_metrics['n2v_ssm'] = n2n_ssm_metrics
        metrics_last['n2v_ssm'] = n2n_ssm_metrics_last
        all_metrics['n2v_ssm_last'] = n2n_ssm_metrics_last
        ax[1][1].imshow(n2n_ssm_denoised_last, cmap="gray")
        ax[1][1].set_title("N2N SSM Denoised Last")
    except Exception as e:
        print(f"Error evaluating n2n ssm last: {e}")
        n2n_ssm_metrics_last = None
        n2n_ssm_denoised_last = None
    
    display_metrics(metrics)
    display_metrics(metrics_last)
    
    fig.tight_layout()
    plt.show()
    
    

if __name__ == "__main__":
    main()