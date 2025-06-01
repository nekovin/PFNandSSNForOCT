import matplotlib.pyplot as plt
from fpss.evaluate.evaluate import evaluate_baseline, evaluate_ssm_constraint
from fpss.utils import load_sdoct_dataset, display_metrics, display_grouped_metrics
from tqdm import tqdm
import torch
import os
import random
from fpss.utils.config import get_config
from fpss.data import get_paired_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"

def normalise_sample(raw_image, reference):
        '''
        sample = random.choice(list(dataset.keys()))
        raw_image = dataset[sample]["raw"][0][0]
        raw_image = raw_image.cpu().numpy()
        print(f"Raw image shape: {raw_image.shape}")
        resized = cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        print(f"Resized image shape: {resized.shape}")
        resized = normalize_image_np(resized)
        #raw_image = resized.to(device)
        raw_image = torch.from_numpy(resized).float()
        raw_image = raw_image.unsqueeze(0).unsqueeze(0)
        raw_image = raw_image.to(device)
    '''
        import cv2
        from fpss.utils import normalize_image_np
        
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

def main(method=None, soct=True):

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

    if soct:

        sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
        dataset = load_sdoct_dataset(sdoct_path)

        # random sample
        #sample = random.choice(list(dataset.keys()))
        sample = '1'
        raw_image = dataset[sample]["raw"][0][0]
        reference = dataset[sample]["avg"][0][0]
        raw_image, reference = normalise_sample(raw_image, reference)
    
    else:
        n_images_per_patient = config['training']['n_images_per_patient']
        batch_size = config['training']['batch_size']
        start = 1
        train_loader, val_loader = get_paired_loaders(start, 2, 5, batch_size)
        print(f"Train loader size: {len(train_loader.dataset)}")
        
        # Get actual tensor data, not shape
        batch_data = next(iter(train_loader))
        sample_batch = batch_data[0]  # Get input tensor batch
        print(f"Sample batch shape: {sample_batch.shape}")
        
        # Get first sample from batch
        raw_image = sample_batch[0]  # Shape: (channels, height, width)
        reference = sample_batch[0]  # Using same as reference
        
        # Add batch dimension
        raw_image = raw_image.unsqueeze(0).to(device)  # Shape: (1, channels, height, width)
        reference = reference.unsqueeze(0).to(device)
    

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a in ax:
        a.axis('off')
    ax[0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
    ax[0].set_title("Raw Image")
    ax[1].imshow(reference.cpu().numpy()[0][0], cmap="gray")
    ax[1].set_title("Reference Image")
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(3, 2, figsize=(15, 15))

    metrics = {}
    metrics_last = {}
    metrics_best = {}
    all_metrics = {}

    # Best checkpoints
    try:    
        n2v_metrics, n2v_denoised = evaluate_baseline(raw_image, reference, method, config_path, override_config=override_config)
        metrics[f'{method}'] = n2v_metrics
        all_metrics[f'{method}'] = n2v_metrics
        ax[0][0].imshow(n2v_denoised, cmap="gray")
        ax[0][0].set_title(f"{method} Denoised")
    except Exception as e:
        print(f"Error evaluating n2v: {e}")
        n2v_metrics = None
        n2v_denoised = None

    try:
        n2v_ssm_metrics, n2v_ssm_denoised = evaluate_ssm_constraint(raw_image, reference, method, config_path, override_config=override_config)
        metrics[f'{method}_ssm'] = n2v_ssm_metrics
        all_metrics[f'{method}_ssm'] = n2v_ssm_metrics
        ax[0][1].imshow(n2v_ssm_denoised, cmap="gray")
        ax[0][1].set_title(f"{method} SSM Denoised")
    except Exception as e:
        print(f"Error evaluating n2v ssm: {e}")
        n2v_ssm_metrics = None
        n2v_ssm_denoised = None

    # Last checkpoints
    try:
        n2v_metrics_last, n2v_denoised_last = evaluate_baseline(raw_image, reference, method, config_path, override_config=override_config, last=True)
        metrics_last[f'{method}'] = n2v_metrics_last
        all_metrics[f'{method}_last'] = n2v_metrics_last
        ax[1][0].imshow(n2v_denoised_last, cmap="gray", vmin=0, vmax=1)
        ax[1][0].set_title(f"{method} Denoised Last")
    except Exception as e:
        print(f"Error evaluating n2v last: {e}")
        n2v_metrics_last = None
        n2v_denoised_last = None

    try:
        n2v_ssm_metrics_last, n2v_ssm_denoised_last = evaluate_ssm_constraint(raw_image, reference, method, config_path, override_config=override_config, last=True)
        metrics_last[f'{method}_ssm'] = n2v_ssm_metrics_last
        all_metrics[f'{method}_ssm_last'] = n2v_ssm_metrics_last
        ax[1][1].imshow(n2v_ssm_denoised_last, cmap="gray")
        ax[1][1].set_title(f"{method} SSM Denoised Last")
    except Exception as e:
        print(f"Error evaluating n2v ssm last: {e}")
        n2v_ssm_metrics_last = None
        n2v_ssm_denoised_last = None

    # Best metrics checkpoints
    try:
        n2v_metrics_best, n2v_denoised_best = evaluate_baseline(raw_image, reference, method, config_path, override_config=override_config, best=True)
        metrics_best[f'{method}'] = n2v_metrics_best
        all_metrics[f'{method}_best'] = n2v_metrics_best
        ax[2][0].imshow(n2v_denoised_best, cmap="gray")
        ax[2][0].set_title(f"{method} Denoised Best")
    except Exception as e:
        print(f"Error evaluating n2v best: {e}")
        n2v_metrics_best = None
        n2v_denoised_best = None

    try:
        n2v_ssm_metrics_best, n2v_ssm_denoised_best = evaluate_ssm_constraint(raw_image, reference, method, config_path, override_config=override_config, best=True)
        metrics_best[f'{method}_ssm'] = n2v_ssm_metrics_best
        all_metrics[f'{method}_ssm_best'] = n2v_ssm_metrics_best
        ax[2][1].imshow(n2v_ssm_denoised_best, cmap="gray")
        ax[2][1].set_title(f"{method} SSM Denoised Best")
    except Exception as e:
        print(f"Error evaluating n2v ssm best: {e}")
        n2v_ssm_metrics_best = None
        n2v_ssm_denoised_best = None

    display_metrics(metrics)
    display_metrics(metrics_last)
    display_metrics(metrics_best)

    fig.tight_layout()
    plt.show()

    from fpss.utils import auto_select_roi_using_flow

    masks = auto_select_roi_using_flow(raw_image[0][0].cpu().numpy(), device)
    mask_fig, ax_mask = plt.subplots(1, 3, figsize=(15, 5))
    ax_mask[0].imshow(raw_image[0][0].cpu().numpy(), cmap='gray')
    ax_mask[0].set_title('Original Image')
    ax_mask[1].imshow(masks[0], cmap='gray')
    ax_mask[1].set_title('Foreground Mask')
    ax_mask[2].imshow(masks[1], cmap='gray')
    ax_mask[2].set_title('Background Mask')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a in ax:
        a.axis('off')
    ax[0].imshow(n2v_denoised, cmap='gray')
    ax[0].set_title(f"{method} Denoised")
    ax[1].imshow(n2v_ssm_denoised, cmap='gray')
    ax[1].set_title(f"{method} SSM Denoised")
    plt.tight_layout()
    plt.show()

    ###

    print("Comparing image patches...")
    print(f"Raw image shape: {raw_image.shape}")
    print(f"Reference image shape: {reference.shape}")
    print(f"N2V denoised shape: {n2v_denoised.shape if n2v_denoised is not None else 'N/A'}")

    def compare_image_patches(img1, img2, figsize=(15, 8)):
        """
        Split two images into 3x3 patches and visualize comparison
        
        Args:
            img1, img2: Input images (H, W) or (H, W, C)
            figsize: Figure size for matplotlib
        """
        # Ensure images are same size
        assert img1.shape == img2.shape, "Images must have same dimensions"
        
        h, w = img1.shape[:2]
        patch_h, patch_w = h // 3, w // 3
        
        fig, axes = plt.subplots(3, 6, figsize=figsize)
        fig.suptitle('Image Patch Comparison (Left: Image 1, Right: Image 2)')
        
        for i in range(3):
            for j in range(3):
                # Extract patches
                y_start, y_end = i * patch_h, (i + 1) * patch_h
                x_start, x_end = j * patch_w, (j + 1) * patch_w
                
                patch1 = img1[y_start:y_end, x_start:x_end]
                patch2 = img2[y_start:y_end, x_start:x_end]
                
                # Plot patch from image 1
                axes[i][j*2].imshow(patch1, cmap='gray')
                axes[i][j*2].set_title(f'Img1 P{i}{j}')
                axes[i][j*2].axis('off')
                
                # Plot patch from image 2
                axes[i][j*2+1].imshow(patch2, cmap='gray')
                axes[i][j*2+1].set_title(f'Img2 P{i}{j}')
                axes[i][j*2+1].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Usage example:
    compare_image_patches(raw_image[0][0].cpu().numpy(), n2v_denoised)
    compare_image_patches(raw_image[0][0].cpu().numpy(), n2v_ssm_denoised)

    def export_results_to_overleaf_table(metrics1, metrics2):
        table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
        table += f"Metric & {str(method).upper()} & {str(method).upper()}-FPSS \\\\\n\\hline\n"
        
        for m in zip(metrics1.keys(), metrics2.keys()):

            # if metric not a number, skip
            
            metric_name = m[0]
            if metric_name not in ['psnr', 'ssim', 'snr', 'cnr', 'enl', 'epi']:
                continue
            metric1 = float(metrics1[metric_name])
            metric2 = float(metrics2[metric_name])
            #metric_values = f"{metric1:.4f} & {metric2:.4f}"
            if metric1 > metric2:
                metric_values = f"\\textbf{{{metric1:.4f}}} & {metric2:.4f}"
            else:
                metric_values = f"{metric1:.4f} & \\textbf{{{metric2:.4f}}}"
            
            # Add to table
            table += f"{str(metric_name).upper()} & {metric_values} \\\\\n"

        #table += f"{metric_name} & {metric_values} & {metric_values} \\\\\n"
        
        table += "\\hline\n\\end{tabular}\n\\caption{Comparison of "
        #{method} and {method}}-FPSS Performance Metrics}\n\\end{table}"
        table += f"{method} and {method}-FPSS Performance Metrics"
        table += "}\n\\end{table}"

        table += "\n\label{tab:"
        table += f"{method}_comparison"
        table += "}\n"
        print(table)
    
    print(metrics)
    export_results_to_overleaf_table(metrics[method], metrics[f'{method}_ssm'])
    
    return metrics

if __name__ == "__main__":
    main()