

def threshold_octa(octa, method='adaptive', threshold_percentage=0.1):
    """
    Threshold OCTA image to remove low-intensity noise
    
    Args:
    - octa: Input OCTA image tensor
    - method: Thresholding method ('adaptive' or 'percentile')
    - threshold_percentage: Percentage for percentile-based thresholding
    
    Returns:
    - Thresholded OCTA image
    """
    # Ensure tensor is numpy or converted to numpy
    if torch.is_tensor(octa):
        octa = octa.detach().squeeze().cpu().numpy()
    
    if method == 'adaptive':
        # Otsu's method for adaptive thresholding
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(octa)
    elif method == 'percentile':
        # Percentile-based thresholding
        threshold = np.percentile(octa, (1 - threshold_percentage) * 100)
    else:
        # Default to percentile method
        threshold = np.percentile(octa, (1 - threshold_percentage) * 100)
    
    # Create binary mask
    thresholded = (octa > threshold).astype(float)
    
    # Convert back to tensor if needed
    if not torch.is_tensor(octa):
        thresholded = torch.from_numpy(thresholded).float()
    
    return thresholded

def normalize_image(np_img):
    if np_img.max() > 0:
        # Create mask of non-background pixels
        foreground_mask = np_img > 0.01
        if foreground_mask.any():
            # Get min/max of only foreground pixels
            fg_min = np_img[foreground_mask].min()
            fg_max = np_img[foreground_mask].max()
            
            # Normalize only foreground pixels to [0,1] range
            if fg_max > fg_min:
                np_img[foreground_mask] = (np_img[foreground_mask] - fg_min) / (fg_max - fg_min)
    
    # Force background to be true black
    np_img[np_img < 0.01] = 0
    return np_img