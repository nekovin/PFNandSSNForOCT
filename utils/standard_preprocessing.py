import os
import glob
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.filters import threshold_local
import cv2
import numpy as np

def normalize_image(np_img):
    if np_img.max() > 0:
        foreground_mask = np_img > 0.01
        if foreground_mask.any():
            fg_min = np_img[foreground_mask].min()
            fg_max = np_img[foreground_mask].max()
            
            if fg_max > fg_min:
                np_img[foreground_mask] = (np_img[foreground_mask] - fg_min) / (fg_max - fg_min)
    
    np_img[np_img < 0.01] = 0
    return np_img


def standard_preprocessing(oct_volume):
    preprocessed = []

    for i, img in enumerate(oct_volume):
        
        if img.max() > 1.0:
            img = img / 255.0
        
        if len(img.shape) > 2:
            img = img[:, :, 0]
        
        resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        resized = resized[:, :, np.newaxis]
        
        preprocessed.append(resized)

    return np.array(preprocessed)


