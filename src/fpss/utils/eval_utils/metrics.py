import numpy as np
from skimage.metrics import structural_similarity
from scipy import ndimage
import cv2
import torch
import os
import time
import matplotlib.pyplot as plt
from fpss.utils.data_utils.paired_preprocessing import paired_preprocessing

def calculate_psnr(img1, img2, max_value=1.0):

    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same dimensions: {img1.shape} vs {img2.shape}")
    
    mse = np.mean((img1 - img2) ** 2)
    #if mse == 0:
        #return float('inf')
    
    return 20 * np.log10(max_value / np.sqrt(mse))

def calculate_ssim(img1, img2, max_value=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Value ranges from -1 to 1, higher is better.
    
    Args:
        img1, img2: Images to compare
        max_value: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        SSIM value
    """
    # Handle multi-channel images
    if img1.ndim == 3 and img1.shape[2] == 1:
        img1 = img1[:,:,0]
    if img2.ndim == 3 and img2.shape[2] == 1:
        img2 = img2[:,:,0]
    
    return structural_similarity(img1, img2, data_range=max_value)

def calculate_snr(img, background_mask=None):
    """
    Calculate Signal-to-Noise Ratio (SNR) for OCT images.
    Higher is better.
    
    Args:
        img: Input image
        background_mask: Optional mask of background regions. If None, estimates background.
    
    Returns:
        SNR value in dB
    """
    # Ensure image is in the right format
    img = np.asarray(img, dtype=np.float32)
    
    if background_mask is None:
        # Assume bottom 10% intensity pixels are background
        threshold = np.percentile(img, 50)
        background_mask = img <= threshold
    
    # Calculate signal and noise regions
    signal = img[~background_mask]
    noise = img[background_mask]
    
    # Calculate SNR
    signal_mean = np.mean(signal)
    noise_std = np.std(noise)
    
    if noise_std == 0:
        print("Noise standard deviation is zero, SNR cannot be calculated.")
        return np.nan
    
    epsilon = 1e-10  # Small value to avoid log(0)
    #return 20 * np.log10(signal_mean / noise_std + epsilon)
    ratio = abs(signal_mean) / noise_std
    return 20 * np.log10(ratio + epsilon)

def calculate_cnr(img, foreground_mask=None, background_mask=None):
    """
    Calculate Contrast-to-Noise Ratio (CNR) for OCT images.
    Higher is better.
    
    Args:
        img: Input image
        foreground_mask: Optional mask of foreground regions. If None, estimates foreground.
        background_mask: Optional mask of background regions. If None, estimates background.
    
    Returns:
        CNR value in dB
    """
    # Ensure image is in the right format
    img = np.asarray(img, dtype=np.float32)
    
    # If no masks provided, estimate them
    if foreground_mask is None or background_mask is None:
        # Use Otsu's thresholding to separate foreground and background
        if img.max() <= 1.0:
            img_8bit = (img * 255).astype(np.uint8)
        else:
            img_8bit = img.astype(np.uint8)
        
        _, binary = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_mask = binary > 0
        background_mask = ~foreground_mask
    
    # Calculate foreground and background statistics
    foreground = img[foreground_mask]
    background = img[background_mask]
    
    if len(foreground) == 0 or len(background) == 0:
        return 0.0
    
    foreground_mean = np.mean(foreground)
    background_mean = np.mean(background)
    foreground_std = np.std(foreground)
    background_std = np.std(background)
    
    # Calculate CNR
    denominator = np.sqrt(foreground_std**2 + background_std**2)
    if denominator == 0:
        return float('inf')
    
    return 10 * np.log10(np.abs(foreground_mean - background_mean) / denominator)

def calculate_enl(img, region_mask=None):
    """
    Calculate Equivalent Number of Looks (ENL) for speckle assessment.
    Higher is better, indicates more smoothing of speckle noise.
    
    Args:
        img: Input image
        region_mask: Optional mask of region to calculate ENL. If None, uses whole image.
    
    Returns:
        ENL value
    """
    # Ensure image is in the right format
    img = np.asarray(img, dtype=np.float32)
    
    # If no mask provided, use the whole image
    if region_mask is None:
        region = img
    else:
        region = img[region_mask]
    
    if len(region) == 0:
        return 0.0
    
    # Calculate ENL
    mean_val = np.mean(region)
    var_val = np.var(region)
    
    if var_val == 0:
        return float('inf')
    
    return (mean_val**2) / var_val

def calculate_epi(img1, img2):
    """
    Calculate Edge Preservation Index (EPI).
    Value ranges from 0 to 1, higher is better.
    
    Args:
        img1: Original image
        img2: Denoised image
    
    Returns:
        EPI value
    """
    # Ensure images are in the same shape and type
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    
    # Apply Sobel operator to detect edges
    edges1_h = ndimage.sobel(img1, axis=0)
    edges1_v = ndimage.sobel(img1, axis=1)
    edges1 = np.sqrt(edges1_h**2 + edges1_v**2)
    
    edges2_h = ndimage.sobel(img2, axis=0)
    edges2_v = ndimage.sobel(img2, axis=1)
    edges2 = np.sqrt(edges2_h**2 + edges2_v**2)
    
    # Calculate correlation between edge maps
    edges1_mean = np.mean(edges1)
    edges2_mean = np.mean(edges2)
    
    numerator = np.sum((edges1 - edges1_mean) * (edges2 - edges2_mean))
    denominator = np.sqrt(np.sum((edges1 - edges1_mean)**2) * np.sum((edges2 - edges2_mean)**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def auto_select_roi(img, n_regions=3, min_size=100):
    """
    Automatically select regions of interest for CNR calculation
    
    Args:
        img: Input image
        n_regions: Number of regions to select
        min_size: Minimum region size
    
    Returns:
        List of masks for ROIs
    """
    if img.max() <= 1.0:
        img_8bit = (img * 255).astype(np.uint8)
    else:
        img_8bit = img.astype(np.uint8)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(img_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Select regions based on size
    region_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
    valid_regions = [i+1 for i, size in enumerate(region_sizes) if size >= min_size]
    
    # Sort regions by size and select the n_regions largest (excluding background)
    valid_regions = sorted(valid_regions, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:n_regions]
    
    # Create masks for selected regions
    masks = []
    for region_idx in valid_regions:
        mask = (labels == region_idx)
        masks.append(mask)
    
    return masks

def calculate_cnr_whole(image):
    """
    Calculate CNR using a percentile-based approach that doesn't require manual ROI selection.
    Works well for evaluating the whole image quality.
    
    Args:
        image: Input image (2D numpy array)
        
    Returns:
        cnr: Contrast-to-noise ratio
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image / image.max()
    
    # Use percentiles to identify signal and background regions
    # Signal: top 10% brightest pixels
    # Background: bottom 10% darkest pixels
    signal_threshold = np.percentile(image, 90)
    background_threshold = np.percentile(image, 10)
    
    signal_mask = image >= signal_threshold
    background_mask = image <= background_threshold
    
    # Extract regions
    signal_region = image[signal_mask]
    background_region = image[background_mask]
    
    # Calculate statistics
    signal_mean = np.mean(signal_region)
    background_mean = np.mean(background_region)
    signal_std = np.std(signal_region)
    background_std = np.std(background_region)
    
    # Calculate CNR
    # Avoid division by zero
    denominator = np.sqrt(signal_std**2 + background_std**2)
    if denominator == 0:
        return 0
    
    cnr = abs(signal_mean - background_mean) / denominator
    
    # Convert to dB for consistency with other metrics
    cnr_db = 20 * np.log10(cnr) if cnr > 0 else -np.inf
    
    return cnr_db

def auto_select_roi_using_flow(img, device='cuda'):

    from fpss.models import SpeckleSeparationUNetAttention

    speckle_module = SpeckleSeparationUNetAttention(input_channels=1, feature_dim=32).to(device)
    try:
        print("Loading ssm model from checkpoint...")
        ssm_checkpoint_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\checkpoints\SSM_mse_best.pth"
        ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
        speckle_module.load_state_dict(ssm_checkpoint['model_state_dict'])
        speckle_module.to(device)
        speckle_module.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting training from scratch.")
        raise e 

    # Convert numpy array to tensor
    if len(img.shape) == 2:
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = speckle_module(img_tensor)
        flow = output['flow_component'].cpu().numpy()[0, 0]
    
    foreground_mask = flow > 0.1 * flow.max()
    background_mask = flow < 0.1 * flow.max()
    #foreground_mask = flow  # Use raw flow values as "weights" rather than binary mask
    #background_mask = 1.0 - flow

    
    return [foreground_mask, background_mask]

def evaluate_oct_denoising(original, denoised, reference=None):

    metrics = {}

    if reference is None:
        metrics['psnr'] = np.nan
        metrics['ssim'] = np.nan
    else:
        metrics['psnr'] = calculate_psnr(denoised, reference)
        metrics['ssim'] = calculate_ssim(denoised, reference)
    
    metrics['snr'] = calculate_snr(denoised) - calculate_snr(original)
    
    try:
        roi_masks = auto_select_roi_using_flow(denoised)
        print("Using auto-selected ROIs Using Flow for CNR calculation.")
    except Exception as e:
        raise e
        roi_masks = auto_select_roi(denoised)
        print(f"Using AutoSelect ROI{e}")

    if len(roi_masks) >= 2:
        print("Using auto-selected ROIs for CNR calculation.")
        metrics['cnr'] = calculate_cnr(denoised, roi_masks[0], roi_masks[1]) - calculate_cnr(original, roi_masks[0], roi_masks[1])
    else:
        metrics['cnr'] = calculate_cnr_whole(denoised) - calculate_cnr_whole(original)
    
    if len(roi_masks) > 0:
        metrics['enl'] = calculate_enl(denoised, roi_masks[0]) - calculate_enl(original, roi_masks[0])
    
    metrics['epi'] = calculate_epi(original, denoised)
    
    return metrics

def denoise_image(model, image, device=None):
    """Apply model to denoise a single image"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    
    input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    output_image = output.squeeze().cpu().numpy()
    
    if len(output_image.shape) == 2:
        return output_image
    else:
        return output_image.transpose(1, 2, 0)

def validate_model(model, n_patients=5, n_images_per_patient=10, device=None, save_results=True):
    """Validate model on a separate set of images"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load validation data
    val_dataset = paired_preprocessing(n_patients, n_images_per_patient, sample=True)
    
    # Track metrics
    metrics_summary = {
        'snr_original': [], 'snr_denoised': [],
        'cnr_original': [], 'cnr_denoised': [],
        'enl_original': [], 'enl_denoised': [],
        'epi': [], 'ssim_self': []
    }
    
    # Create save directory if needed
    save_path = os.path.join(os.getcwd(), 'validation_results')
    if save_results and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Keep track of one example for visualization
    example_result = None
    
    for patient_id, data in val_dataset.items():
        for i in range(len(data) - 1):
            input_img = data[i][0]
            target_img = data[i+1][0]  # Next scan as reference
            
            # Denoise the input image
            denoised_img = denoise_image(model, input_img, device)
            
            # Calculate metrics
            metrics = evaluate_oct_denoising(input_img, denoised_img, target_img)
            
            # Add to summary
            for key in metrics_summary:
                if key in metrics:
                    metrics_summary[key].append(metrics[key])
            
            # Store the first result as our example
            if example_result is None:
                example_result = {
                    'patient_id': patient_id,
                    'image_id': i,
                    'input': input_img,
                    'target': target_img,
                    'denoised': denoised_img,
                    'metrics': {k: metrics[k] for k in metrics if not np.isnan(metrics[k]) and not np.isinf(metrics[k])}
                }
    
    # Calculate averages only for metrics that have values
    avg_metrics = {}
    for k, v in metrics_summary.items():
        if len(v) > 0:
            avg_metrics[k] = np.mean(v)
    
    # Print results
    print("\n===== Validation Results =====")
    
    # Print metrics if available
    if 'snr_original' in avg_metrics and 'snr_denoised' in avg_metrics:
        print(f"SNR: {avg_metrics['snr_original']:.2f} dB → {avg_metrics['snr_denoised']:.2f} dB")
    
    if 'cnr_original' in avg_metrics and 'cnr_denoised' in avg_metrics:
        print(f"CNR: {avg_metrics['cnr_original']:.2f} dB → {avg_metrics['cnr_denoised']:.2f} dB")
    
    if 'enl_original' in avg_metrics and 'enl_denoised' in avg_metrics:
        print(f"ENL: {avg_metrics['enl_original']:.2f} → {avg_metrics['enl_denoised']:.2f}")
    
    if 'epi' in avg_metrics:
        print(f"Edge Preservation Index: {avg_metrics['epi']:.4f}")
    
    if 'ssim_self' in avg_metrics:
        print(f"SSIM (self-reference): {avg_metrics['ssim_self']:.4f}")
    
    print("=============================\n")
    
    # Save results
    if save_results and example_result is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(save_path, f'results_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save summary metrics
        with open(os.path.join(results_dir, 'metrics_summary.txt'), 'w') as f:
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v}\n")
        
        # Save the single example image
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(example_result['input'], cmap='gray')
        plt.title('Input (Noisy)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(example_result['denoised'], cmap='gray')
        plt.title('Denoised')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(example_result['target'], cmap='gray')
        plt.title('Target')
        plt.axis('off')
        
        plt.savefig(os.path.join(results_dir, 'example_comparison.png'))
        plt.close()
        
        print(f"Validation results saved to {results_dir}")
    
    return avg_metrics

from IPython.display import display

def display_metrics(metrics):
    import pandas as pd
    df = pd.DataFrame(metrics)
    def highlight_max(s):
        is_max = s == s.max()
        return ['font-weight: bold' if v else '' for v in is_max]
    styled_df = df.style.apply(highlight_max, axis=1)
    display(styled_df)
    return df

def display_grouped_metrics(metrics):
        import pandas as pd
        from IPython.display import display, HTML
        
        # Define base schemas to group by
        base_schemas = ['n2n', 'n2v', 'n2s', 'pfn']
        
        schema_tables = {}
        
        # Process each metric key
        for metric_key in metrics.keys():
            # Find which base schema this metric belongs to
            matched_schema = None
            for schema in base_schemas:
                if schema in metric_key:
                    matched_schema = schema
                    break
            
            if matched_schema:
                if matched_schema not in schema_tables:
                    schema_tables[matched_schema] = {}
                
                schema_tables[matched_schema][metric_key] = metrics[metric_key]
        
        # Display tables for each schema group
        all_dfs = {}
        for schema, data in schema_tables.items():
            if data:  # Only process if we have data
                df = pd.DataFrame(data)
                
                # Apply styling to highlight maximum values in each row
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['font-weight: bold' if v else '' for v in is_max]
                
                styled_df = df.style.apply(highlight_max, axis=1)
                
                print(f"\n--- {schema.upper()} Models ---")
                display(styled_df)
                all_dfs[schema] = df
        
        return all_dfs