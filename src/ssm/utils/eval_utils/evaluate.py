import torch
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.transform import resize

from ssm.utils import normalize_image
from ssm.utils.eval_utils.metrics import evaluate_oct_denoising

def get_sample_image(dataloader, device):
    sample = next(iter(dataloader))
    image = sample[0].to(device)
    return image

def plot_sample(image, denoised, model, method):
    

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    fig.suptitle(f"Model: {model.__class__.__name__} - {method}", fontsize=16)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    normalised_denoised = normalize_image(denoised)

    ax[1].imshow(denoised, cmap='gray')
    ax[1].set_title('Denoised Image')
    ax[1].axis('off')

    ax[2].imshow(normalised_denoised, cmap='gray')
    ax[2].set_title('Normalized Denoised Image')
    ax[2].axis('off')

    plt.show()
    

def denoise_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        denoised_image = model(image)
    return denoised_image

load_dotenv()

device = os.getenv("DEVICE")
device

def evaluate(image, reference, model, method):

    denoised = denoise_image(model, image, device='cuda')[0][0]

    try:
        sample_image = image.cpu().numpy()[0][0]
    except:
        sample_image = image

    denoised = denoised.cpu().numpy()
    reference = reference.cpu().numpy()

    if len(denoised.shape) == 3: # this is because my pfn model retusn a 3d tensor
        denoised = denoised[0]
    metrics = evaluate_oct_denoising(sample_image, denoised, reference)

    return metrics, denoised


def load_sdoct_dataset(dataset_path, target_size=(256, 256)):
    """Load all images from the SDOCT dataset and resize them.
    
    Args:
        dataset_path: Path to the SDOCT dataset directory
        target_size: Size to resize images to
        
    Returns:
        Dictionary containing patient data with raw and averaged images
    """
    sdoct_data = {}
    patients = os.listdir(dataset_path)
    
    print(f"Loading SDOCT dataset from {dataset_path}")
    for patient in tqdm(patients, desc="Loading patients"):
        patient_path = os.path.join(dataset_path, patient)
        avg_path = os.path.join(patient_path, f"{patient}_Averaged Image.tif")
        raw_path = os.path.join(patient_path, f"{patient}_Raw Image.tif")
        
        try:
            # Check if both files exist
            if not os.path.exists(avg_path) or not os.path.exists(raw_path):
                print(f"Missing files for patient {patient}")
                continue
                
            # Load and resize images
            raw_img = io.imread(raw_path)
            avg_img = io.imread(avg_path)
            
            # Normalize images to 0-1 range if needed
            if raw_img.max() > 1.0:
                raw_img = raw_img / 255.0
            if avg_img.max() > 1.0:
                avg_img = avg_img / 255.0
                
            # Resize images
            raw_img = resize(raw_img, target_size, anti_aliasing=True)
            avg_img = resize(avg_img, target_size, anti_aliasing=True)
            
            # Convert to PyTorch tensors
            raw_tensor = torch.from_numpy(raw_img).float().unsqueeze(0).unsqueeze(0)
            avg_tensor = torch.from_numpy(avg_img).float().unsqueeze(0).unsqueeze(0)
            
            # Store in dictionary
            sdoct_data[patient] = {
                "raw": raw_tensor,
                "avg": avg_tensor,
                "raw_np": raw_img,
                "avg_np": avg_img
            }
            
        except Exception as e:
            print(f"Error processing patient {patient}: {e}")
    
    print(f"Successfully loaded {len(sdoct_data)} patients")
    return sdoct_data