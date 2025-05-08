import os
import re
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def extract_number(filename):
    """Extract number from filename pattern (number)"""
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else -1

def create_oct_mask(image, threshold_factor=0.3):
    """Create a binary mask for OCT image focusing on tissue regions"""
    # Convert to numpy if tensor
    if hasattr(image, 'cpu') and callable(getattr(image, 'cpu')):
        image = image.cpu().detach().numpy()
    
    # Ensure 2D
    if isinstance(image, np.ndarray) and image.ndim == 3:
        image = image.squeeze()
    
    # Normalize to 0-1
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Calculate adaptive threshold
    mean_val = np.mean(image)
    std_val = np.std(image)
    threshold = mean_val + threshold_factor * std_val
    
    # Create initial mask
    mask = (image > threshold).astype(np.float32)
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small objects and fill holes
    mask = cv2.medianBlur(mask.astype(np.uint8), 5)
    
    return mask

def compute_dice_coefficient(mask1, mask2):
    """Calculate Dice coefficient between two binary masks"""
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return 1.0  # Both masks are empty
    return (2.0 * intersection) / sum_masks

def compute_jaccard_index(mask1, mask2):
    """Calculate Jaccard index between two binary masks"""
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0  # Both masks are empty
    return intersection / union

def process_folder(input_folder):
    """Process all images in a folder to create masks"""
    masks = {}
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.tiff'))]
    
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = create_oct_mask(image)
        masks[filename] = mask
        
    return masks

def save_fused_images(fused_images, level, output_dir):
    """Save fused images to the specified directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, image_array in enumerate(fused_images):
        image = Image.fromarray(image_array.astype(np.uint8)).convert('L')
        image_filename = f'Fused_Image_Level_{level}_{idx}.tif'
        image.save(os.path.join(output_dir, image_filename))
    
    print(f"Saved {len(fused_images)} fused images at Level {level} in {output_dir}")

def fuse_images(image_list, diabetes, patient, output_dir, level=0):
    """Recursively fuse images to create fusion levels"""
    if len(image_list) <= 1:
        return
    
    fused_images = []

    for i in range(0, len(image_list), 2):
        if i + 1 < len(image_list):
            fused = np.mean([image_list[i], image_list[i + 1]], axis=0)
        else:
            fused = image_list[i]
        fused_images.append(fused)
    
    output_dir = output_dir + f"_{level}"
    save_fused_images(fused_images, level, output_dir)
    
    fuse_images(fused_images, diabetes, patient, output_dir, level + 1)

def fuse_images(image_list, diabetes, patient, base_output_dir, level=0):
    """Recursively fuse images to create fusion levels"""
    if len(image_list) <= 1:
        return
    
    fused_images = []

    for i in range(0, len(image_list), 2):
        if i + 1 < len(image_list):
            fused = np.mean([image_list[i], image_list[i + 1]], axis=0)
        else:
            fused = image_list[i]
        fused_images.append(fused)
    
    # Create a new output directory for this level instead of appending to the path
    current_output_dir = f"{base_output_dir}_{level}"
    save_fused_images(fused_images, level, current_output_dir)
    
    # Pass the original base_output_dir to the next recursive call
    fuse_images(fused_images, diabetes, patient, base_output_dir, level + 1)

def single(config_path=None):

    if not config_path:
        patient_int = 1
        patient = r'RawDataQA ({patient_int})'
        diabetes = 2

        if diabetes != 0:
            patient = patient.replace(' ', f'-{diabetes} ')
    else:
        config = get_config(config_path)

        fusion_config = config['fusion']

        base_dataset_path = fusion_config['base_dataset_path']

        single_patient = fusion_config.get('single_patient', True)

        if single_patient:
            print("Single patient mode")
            patient_int = fusion_config['patient']
            patient = f'RawDataQA ({patient_int})'
            
            diabetes = fusion_config['diabetes']
            if diabetes != 0:
                patient = patient.replace(' ', f'-{diabetes} ')
            dataset_path = base_dataset_path + f'/{diabetes}/{patient}'
            #output_dir = f'../data/FusedDataset/{diabetes}/{patient}/FusedImages_Level_{level}'
            base_output_dir = fusion_config['base_output_path']
            output_dir = base_output_dir + f'/{diabetes}/{patient}/FusedImages_Level'
            print(f"Output directory: {output_dir}")
    
    images = sorted(
        [os.path.join(dataset_path, image) for image in os.listdir(dataset_path)],
        key=lambda x: extract_number(os.path.basename(x))
    )
    
    loaded_images = [Image.open(image) for image in images]
    image_arrays = [np.array(image) for image in loaded_images]
    
    stacked_images = np.stack(image_arrays, axis=0)
    fused_image_array = np.mean(stacked_images, axis=0).astype(np.uint8)
    reference_mask = create_oct_mask(fused_image_array)
    
    masks = process_folder(dataset_path)

    dice_scores = []
    jaccard_scores = []
    image_paths = []
    
    for mask_name, mask in masks.items():
        mask = (mask > 0).astype(np.uint8)
        
        if reference_mask.shape != mask.shape:
            resized_reference = cv2.resize(
                reference_mask, 
                (mask.shape[1], mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            resized_reference = reference_mask
        
        resized_reference = (resized_reference > 0).astype(np.uint8)
        
        dice = compute_dice_coefficient(mask, resized_reference)
        jaccard = compute_jaccard_index(mask, resized_reference)
        
        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        image_paths.append(mask_name)
        
        print(f"Mask: {mask_name}, Dice: {dice:.4f}, Jaccard: {jaccard:.4f}")
    
    dice_threshold = 0.5
    good_images = [image_paths[i] for i, dice in enumerate(dice_scores) if dice >= dice_threshold]
    
    good_image_arrays = []
    for img in good_images:
        npimg = cv2.imread(os.path.join(dataset_path, img))
        good_image_arrays.append(npimg)
    
    print(f"Selected {len(good_image_arrays)} good quality images")

    print(f"Output directory: {output_dir}")
    
    fuse_images(good_image_arrays, diabetes, patient, output_dir, level=0)

from utils.config import get_config

def main(config_path=None, override=None):

    if not config_path:
        patient_int = 1
        patient = fr'RawDataQA ({patient_int})'
        diabetes = 2

        
    else:
        config = get_config(config_path, override)

        fusion_config = config['fusion']

        base_dataset_path = fusion_config['base_dataset_path']
        base_output_dir = fusion_config['base_output_path']

        diabetes = fusion_config['diabetes']

    start_patient = fusion_config['start_patient']
    end_patient = fusion_config['end_patient']

    for diabetes in range(0, 2+1):
        if diabetes == 0:
            end_patient = 42+1
        if diabetes == 1:
            end_patient = 30+1
        if diabetes == 2:
            end_patient = 28+1
        for patient_int in range(start_patient, end_patient):
            patient = f'RawDataQA ({patient_int})'

            if diabetes != 0:
                patient = patient.replace(' ', f'-{diabetes} ')
                
            dataset_path = os.path.join(base_dataset_path, str(diabetes), patient)
            print(f"Processing dataset: {dataset_path}")

            dataset_path = base_dataset_path + f'/{diabetes}/{patient}'
            
            output_dir = base_output_dir + f'/{diabetes}/{patient}/FusedImages_Level'

            #if not os.path.exists(output_dir):
                #os.makedirs(output_dir)

            print(os.listdir(dataset_path))
        
            images = sorted(
                [os.path.join(dataset_path, image) for image in os.listdir(dataset_path)],
                key=lambda x: extract_number(os.path.basename(x))
            )
            
            loaded_images = [Image.open(image) for image in images]
            image_arrays = [np.array(image) for image in loaded_images]
            
            stacked_images = np.stack(image_arrays, axis=0)
            fused_image_array = np.mean(stacked_images, axis=0).astype(np.uint8)
            reference_mask = create_oct_mask(fused_image_array)
            
            masks = process_folder(dataset_path)

            dice_scores = []
            jaccard_scores = []
            image_paths = []
            
            for mask_name, mask in masks.items():
                mask = (mask > 0).astype(np.uint8)
                
                if reference_mask.shape != mask.shape:
                    resized_reference = cv2.resize(
                        reference_mask, 
                        (mask.shape[1], mask.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    resized_reference = reference_mask
                
                resized_reference = (resized_reference > 0).astype(np.uint8)
                
                dice = compute_dice_coefficient(mask, resized_reference)
                jaccard = compute_jaccard_index(mask, resized_reference)
                
                dice_scores.append(dice)
                jaccard_scores.append(jaccard)
                image_paths.append(mask_name)
            
            dice_threshold = 0.5
            good_images = [image_paths[i] for i, dice in enumerate(dice_scores) if dice >= dice_threshold]
            
            good_image_arrays = []
            for img in good_images:
                npimg = cv2.imread(os.path.join(dataset_path, img))
                good_image_arrays.append(npimg)
            
            fuse_images(good_image_arrays, diabetes, patient, output_dir, level=0)
    

if __name__ == "__main__":
    main()