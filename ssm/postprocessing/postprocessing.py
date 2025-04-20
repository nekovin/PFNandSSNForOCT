import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssm")

def normalize_image(np_img):
    np_img = np_img.copy()
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