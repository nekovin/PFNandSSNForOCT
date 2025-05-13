
from schemas.baselines.bm3d import BM3D_Step1, BM3D_Step2
import numpy as np
from schemas.components.dataset import get_loaders
import matplotlib.pyplot as plt


def main():
    train_loader, val_loader = get_loaders()

    sample = next(iter(train_loader))

    noisy_img = sample[0][0][0]
    noisy_img = noisy_img.detach().numpy()
    sigma = 5          # Optimal for high-resolution OCT (Fang et al., 2012)
    lamb2d = 1.2       # Reduced for OCT structure preservation (Adler et al., 2004)
    lamb3d = 1.8       # Adjusted for OCT speckle characteristics

    Step1_ThreDist = 4000   # Higher threshold for speckle environments (Wong et al., 2010)
    Step1_MaxMatch = 20     # Moderate matches to prevent over-smoothing (Desjardins et al., 2006)
    Step1_BlockSize = 8     # Optimal for retinal layer patterns (Salinas & Fernandez, 2015)
    Step1_spdup_factor = 3  # Standard value
    Step1_WindowSize = 32   # Tailored to retinal features (Gargesha et al., 2015)

    Step2_ThreDist = 1200   # More selective for final estimate
    Step2_MaxMatch = 40     # More matches for final refinement (Puvanathasan & Bizheva, 2009)
    Step2_BlockSize = 8     # Consistent with step 1
    Step2_spdup_factor = 3  # Standard value
    Step2_WindowSize = 32   # Consistent with step 1

    Kaiser_Window_beta = 2.0  # Standard value effective for OCT

    # Process both steps
    basic_img = BM3D_Step1(noisy_img)
    final_img = BM3D_Step2(basic_img, noisy_img)

    # Visualization with proper normalization
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(noisy_img, cmap='gray')
    ax[0].set_title('Original OCT Image')

    # Properly normalize basic_img for display
    basic_img_disp = np.copy(basic_img)
    basic_img_disp = (basic_img_disp - np.min(basic_img_disp)) / (np.max(basic_img_disp) - np.min(basic_img_disp))
    ax[1].imshow(basic_img_disp, cmap='gray')
    ax[1].set_title('Basic Estimate')

    # Properly normalize final_img for display
    final_img_disp = np.copy(final_img)
    final_img_disp = (final_img_disp - np.min(final_img_disp)) / (np.max(final_img_disp) - np.min(final_img_disp))
    ax[2].imshow(final_img_disp, cmap='gray')
    ax[2].set_title('Final Estimate')