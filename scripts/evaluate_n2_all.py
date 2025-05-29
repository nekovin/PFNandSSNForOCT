import matplotlib.pyplot as plt
from ssm.evaluation import evaluate_baseline, evaluate_ssm_constraint, evaluate_progressssive_fusion_unet
from ssm.utils import load_sdoct_dataset, display_metrics, display_grouped_metrics
from tqdm import tqdm
import torch
import os
import random
from ssm.utils.config import get_config

def main(method=None):
    import os
    import torch
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    config_path = os.getenv("N2_CONFIG_PATH")
    config = get_config(config_path)
    n_patients = config['training']['n_patients']
    
    override_config = {
        "eval": {

        }
    }

    all_metrics = defaultdict(list)
    combined_metrics = {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the dataset
    sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    dataset = load_sdoct_dataset(sdoct_path)
    
    def normalise_sample(raw_image, reference):
        import cv2
        from ssm.utils import normalize_image_np
        
        # Normalise the raw image
        raw_image = raw_image.cpu().numpy()
        resized = cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        resized = normalize_image_np(resized)
        raw_image = torch.from_numpy(resized).float()
        raw_image = raw_image.unsqueeze(0).unsqueeze(0)
        raw_image = raw_image.to(device)

        # Normalise the reference image
        reference = reference.cpu().numpy()
        resized_ref = cv2.resize(reference, (256, 256), interpolation=cv2.INTER_LINEAR)
        resized_ref = normalize_image_np(resized_ref)
        reference = torch.from_numpy(resized_ref).float()
        reference = reference.unsqueeze(0).unsqueeze(0)
        reference = reference.to(device)

        return raw_image, reference
    
    # Process all samples in the dataset
    print(f"Processing {len(dataset)} samples from the SDOCT dataset...")
    
    # Initialize visualization for a random sample
    random_keys = random.sample(list(dataset.keys()), min(5, len(dataset)))
    vis_sample_idx = 0
    vis_figs = []
    
    for sample_key in dataset.keys():
        print(f"Processing sample: {sample_key}")
        try:
            raw_image = dataset[sample_key]["raw"][0][0]
            reference = dataset[sample_key]["avg"][0][0]
            
            # Normalize the sample
            raw_image, reference = normalise_sample(raw_image, reference)
            
            # Create visualization for random samples
            if sample_key in random_keys:
                fig, ax = plt.subplots(3, 3, figsize=(18, 15))
                ax[0][0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
                ax[0][0].set_title("Raw Image")
                ax[0][1].imshow(reference.cpu().numpy()[0][0], cmap="gray")
                ax[0][1].set_title("Reference Image")
                vis_figs.append(fig)
                
                # Run models and evaluations
                try:
                    baseline_metrics, baseline_denoised = evaluate_baseline(
                        raw_image, reference, method, config_path, override_config=override_config
                    )
                    all_metrics[f'{method}'].append(baseline_metrics)
                    ax[1][0].imshow(baseline_denoised, cmap="gray")
                    ax[1][0].set_title(f"{method} Denoised")
                except Exception as e:
                    print(f"Error evaluating baseline for {sample_key}: {e}")
                
                try:
                    ssm_metrics, ssm_denoised = evaluate_ssm_constraint(
                        raw_image, reference, method, config_path, override_config=override_config
                    )
                    all_metrics[f'{method}_ssm'].append(ssm_metrics)
                    ax[1][1].imshow(ssm_denoised, cmap="gray")
                    ax[1][1].set_title(f"{method} SSM Denoised")
                except Exception as e:
                    print(f"Error evaluating SSM for {sample_key}: {e}")
                
                try:
                    best_metrics, best_denoised = evaluate_baseline(
                        raw_image, reference, method, config_path, override_config=override_config, best=True
                    )
                    all_metrics[f'{method}_best'].append(best_metrics)
                    ax[2][0].imshow(best_denoised, cmap="gray")
                    ax[2][0].set_title(f"{method} Best Denoised")
                except Exception as e:
                    print(f"Error evaluating best baseline for {sample_key}: {e}")
                
                try:
                    ssm_best_metrics, ssm_best_denoised = evaluate_ssm_constraint(
                        raw_image, reference, method, config_path, override_config=override_config, best=True
                    )
                    all_metrics[f'{method}_ssm_best'].append(ssm_best_metrics)
                    ax[2][1].imshow(ssm_best_denoised, cmap="gray")
                    ax[2][1].set_title(f"{method} SSM Best Denoised")
                except Exception as e:
                    print(f"Error evaluating best SSM for {sample_key}: {e}")
                
                # Display ROI masks for visualization samples
                from ssm.utils import auto_select_roi_using_flow
                masks = auto_select_roi_using_flow(raw_image[0][0].cpu().numpy(), device)
                ax[0][2].imshow(masks[0], cmap="gray")
                ax[0][2].set_title("Foreground Mask")
                ax[1][2].imshow(masks[1], cmap="gray")
                ax[1][2].set_title("Background Mask")
                
                fig.tight_layout()
                vis_sample_idx += 1
            else:
                # For non-visualization samples, just compute metrics
                try:
                    baseline_metrics, _ = evaluate_baseline(
                        raw_image, reference, method, config_path, override_config=override_config
                    )
                    all_metrics[f'{method}'].append(baseline_metrics)
                except Exception as e:
                    print(f"Error evaluating baseline for {sample_key}: {e}")
                
                try:
                    ssm_metrics, _ = evaluate_ssm_constraint(
                        raw_image, reference, method, config_path, override_config=override_config
                    )
                    all_metrics[f'{method}_ssm'].append(ssm_metrics)
                except Exception as e:
                    print(f"Error evaluating SSM for {sample_key}: {e}")
        
        except Exception as e:
            print(f"Error processing sample {sample_key}: {e}")
    
    for model_name, metrics_list in all_metrics.items():
        if not metrics_list:
            continue
            
        # Initialize a dictionary to store the average metrics
        avg_metrics = {}
        
        # Get all metric keys from the first sample
        metric_keys = metrics_list[0].keys()
        
        # Calculate average for each metric
        # Inside your averaging loop
        for key in metric_keys:
            values = []
            print(f"\nChecking values for metric: {key}")
            
            for i, metrics in enumerate(metrics_list):
                if key in metrics:
                    val = metrics[key]
                    print(f"  Sample {i}: {key} = {val} (type: {type(val)})")
                    values.append(val)
            
            if values:
                try:
                    avg_metrics[key] = sum(values) / len(values)
                except TypeError as e:
                    print(f"ERROR: Cannot sum values for {key}: {values}")
                    print(f"Exception: {e}")

        #for key in metric_keys:
            #values = [metrics[key] for metrics in metrics_list if key in metrics]
            #if values:
                #avg_metrics[key] = sum(values) / len(values)
        
        combined_metrics[model_name] = avg_metrics
    
    # Display the average metrics
    print("\n===== Average Metrics Across All Samples =====")
    display_metrics(combined_metrics)
    
    # Save results to file
    import json
    result_file = f"sdoct_results_{method}_full_dataset.json"
    with open(result_file, "w") as f:
        json.dump(combined_metrics, f, indent=4)
    
    print(f"Results saved to {result_file}")
    
    # Show visualizations
    for fig in vis_figs:
        plt.figure(fig.number)
        plt.show()

if __name__ == "__main__":
    import sys
    method = "n2n" if len(sys.argv) <= 1 else sys.argv[1]
    main(method)
        

if __name__ == "__main__":
    main()