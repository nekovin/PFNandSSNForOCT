import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")

from torch.utils.data import Dataset, DataLoader

import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ssn2v.stage1.preprocessing import preprocessing
from ssn2v.stage1.preprocessing_v2 import preprocessing_v2

class Stage2(Dataset):
    def __init__(self, data_by_patient, transform=None):
        """
        Args:
            data_by_patient: Dictionary where keys are patient IDs and values are lists of samples
            transform: Optional transforms to apply
        """
        self.data_by_patient = data_by_patient
        self.transform = transform
        
        # Create an index mapping to locate samples
        self.index_mapping = []
        for patient_id, patient_data in self.data_by_patient.items():
            for i in range(len(patient_data) - 1):  # Ensure we can get pairs
                self.index_mapping.append((patient_id, i))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        # Get patient ID and sample index from our mapping
        patient_id, sample_idx = self.index_mapping[idx]
        
        # Get two consecutive samples from the same patient
        sample1 = self.data_by_patient[patient_id][sample_idx]
        sample2 = self.data_by_patient[patient_id][sample_idx + 1]
        
        # Extract the images from both samples
        preprocessed1 = sample1[0]  # First preprocessed image
        preprocessed2 = sample2[0]  # Second preprocessed image
        octa_calc = sample1[1]      # OCTA calculation 
        stage1_output = sample1[2]  # Stage 1 output
        
        # Convert to tensors if they aren't already
        if not torch.is_tensor(preprocessed1):
            preprocessed1 = torch.tensor(preprocessed1, dtype=torch.float32).unsqueeze(0)
        if not torch.is_tensor(preprocessed2):
            preprocessed2 = torch.tensor(preprocessed2, dtype=torch.float32).unsqueeze(0)
        if not torch.is_tensor(octa_calc):
            octa_calc = torch.tensor(octa_calc, dtype=torch.float32).unsqueeze(0)
        if not torch.is_tensor(stage1_output):
            stage1_output = torch.tensor(stage1_output, dtype=torch.float32).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            preprocessed1 = self.transform(preprocessed1)
            preprocessed2 = self.transform(preprocessed2)
            octa_calc = self.transform(octa_calc)
            stage1_output = self.transform(stage1_output)
        
        # Return both preprocessed images along with other data
        return preprocessed1, preprocessed2, octa_calc, stage1_output

def get_stage2_loaders(dataset, img_size, test_split=0.2, val_split=0.15):
    # Get list of patient IDs
    patients = list(dataset.keys())
    print(f"Available patients: {patients}")
    
    # Prepare data by split type while maintaining patient separation
    train_data_by_patient = {}
    val_data_by_patient = {}
    test_data_by_patient = {}
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    # Process each patient separately
    for patient in patients:
        # Get patient's data
        patient_data = dataset[patient]
        
        # Shuffle patient's data
        random.shuffle(patient_data)
        
        # Calculate split sizes
        total_samples = len(patient_data)
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * val_split)
        train_size = total_samples - test_size - val_size
        
        # Split patient's data
        train_data = patient_data[:train_size]
        val_data = patient_data[train_size:train_size + val_size]
        test_data = patient_data[train_size + val_size:]
        
        # Add to respective patient dictionaries
        if train_data:  # Only add if there's data
            train_data_by_patient[patient] = train_data
        if val_data:
            val_data_by_patient[patient] = val_data
        if test_data:
            test_data_by_patient[patient] = test_data
    
    # Create datasets with patient-specific organization
    train_dataset = Stage2(data_by_patient=train_data_by_patient, transform=transform)
    val_dataset = Stage2(data_by_patient=val_data_by_patient, transform=transform)
    test_dataset = Stage2(data_by_patient=test_data_by_patient, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train size: {len(train_loader)}, Validation size: {len(val_loader)}, Test size: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def load_data(n_patients = 4, background_thresh=0.01):
    stage1_data = r"C:\Datasets\OCTData\stage1_outputs"

    img_size = 256

    patients = range(1, n_patients)  # Adjust this to the number of patients you have

    def normalize_image(np_img):
        if np_img.max() > 0:
            # Create mask of non-background pixels
            foreground_mask = np_img > background_thresh
            if foreground_mask.any():
                # Get min/max of only foreground pixels
                fg_min = np_img[foreground_mask].min()
                fg_max = np_img[foreground_mask].max()
                
                # Normalize only foreground pixels to [0,1] range
                if fg_max > fg_min:
                    np_img[foreground_mask] = (np_img[foreground_mask] - fg_min) / (fg_max - fg_min)
        
        # Force background to be true black
        np_img[np_img < background_thresh] = 0
        return np_img

    stage1_dataset = {}

    for patient in patients:

        stage1_dataset[patient] = {}
        patient_path = os.path.join(stage1_data, f"{patient}")
        patient_data_len = len([img for img in os.listdir(patient_path) if 'raw' in img])

        for i in range(1, patient_data_len+1):

            # list images with raw in name
            raw_images = []

            stage1_dataset[patient][i] = {}

            raw_img = plt.imread(os.path.join(stage1_data, f"{patient}", f"raw{i}.png"))
            octa_img = plt.imread(os.path.join(stage1_data, f"{patient}", f"octa{i}.png"))

            # norm the octa
            octa_img = normalize_image(octa_img)

            stage1_dataset[patient][i]['raw'] = raw_img
            stage1_dataset[patient][i]['octa'] = octa_img[:,:,1]

    regular = False

    n_patients = len(patients)
    n = 50
    n_images_per_patient = max(10, n)


    if regular:
        dataset = preprocessing(n_patients, n_images_per_patient, n_neighbours = 2,  threshold=0) #n neighbours must be 2
        name = "regular"
    else:
        dataset = preprocessing_v2(n_patients, n_images_per_patient, n_neighbours = 5, threshold=60, sample=False, post_process_size=10)
        name = "v2"

    print(f"Dataset {name} created with {len(dataset)} images")

    for patient in dataset.keys():
        for i in range(len(dataset[patient])):
            dataset[patient][i].append(stage1_dataset[patient][i+1]['octa'])

    return get_stage2_loaders(dataset, img_size, test_split=0.2, val_split=0.15)