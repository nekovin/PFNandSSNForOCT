import glob
import os
import numpy as np
from skimage import io

def load_patient_data(base_path, verbose=False):

    if verbose:
        print(f"Loading data from: {base_path}")
    
    files = sorted(glob.glob(os.path.join(base_path, "*.tiff")))
    if not files:
        files = sorted(glob.glob(os.path.join(base_path, "*.tif")))
    if not files:
        files = sorted(glob.glob(os.path.join(base_path, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(base_path, "*.jpg")))
    
    if verbose:
        print(f"Found {len(files)} files")

    oct_scans = []
    for file in files:
        try:
            img = io.imread(file)
            
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
                
            oct_scans.append(img)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return oct_scans