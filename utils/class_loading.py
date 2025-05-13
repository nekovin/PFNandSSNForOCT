import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
from utils.data import preprocessing_v2


class OCTDataset(Dataset):
    """
    PyTorch Dataset for OCT images with classification labels
    """
    def __init__(self, data_dict, classes=[0, 1, 2], transform=None, train=True, val_split=0.2, seed=42):
        """
        Initialize the dataset
        
        Args:
            data_dict: Dictionary from preprocessing_v2 function with patient data
            classes: List of class labels
            transform: PyTorch transforms to apply
            train: If True, use training split, else validation split
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
        """
        self.transform = transform
        self.classes = classes
        
        # Flatten the data structure from patient dictionary to a list of (image, label) pairs
        self.data = []
        for patient_id, patient_data in data_dict.items():
            # Assuming patient_data contains (input_image, target) pairs
            for input_img, target in patient_data:
                # Determine which class this image belongs to
                # This is where you need to define your classification logic
                # For example, if you're classifying based on noise level:
                
                # Example classification logic (modify as needed for your specific use case)
                # Class 0: Clean images - low noise
                # Class 1: Moderate noise
                # Class 2: Heavy noise
                label = self._determine_class(input_img, target)
                
                self.data.append((input_img, label))
        
        # Split data into train and validation sets
        random.seed(seed)
        train_indices, val_indices = train_test_split(
            range(len(self.data)), 
            test_size=val_split, 
            random_state=seed,
            stratify=[item[1] for item in self.data]  # Stratify by label
        )
        
        # Select appropriate indices based on train/val
        self.indices = train_indices if train else val_indices
    
    def _determine_class(self, input_img, target):
        """
        Determine class based on the input image and target
        Modify this logic based on your specific classification needs
        
        Args:
            input_img: Input OCT image
            target: Target/clean image or OCTA image
            
        Returns:
            class_idx (int): Class index (0, 1, or 2)
        """
        # Example classification logic based on noise level
        # You should replace this with your own logic
        
        # Calculate noise level (e.g., using standard deviation of difference)
        if isinstance(input_img, np.ndarray) and isinstance(target, np.ndarray):
            diff = input_img - target
            noise_level = np.std(diff)
            
            # Thresholds for classification - adjust these based on your data
            if noise_level < 0.1:
                return 0  # Clean
            elif noise_level < 0.2:
                return 1  # Moderate noise
            else:
                return 2  # Heavy noise
        
        # Alternative: if you already have predetermined class labels
        # return predefined_label
        
        # Default fallback - you should rarely reach here
        return random.choice(self.classes)
    
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Map the index to the shuffled index
        actual_idx = self.indices[idx]
        
        # Get the image and label
        image, label = self.data[actual_idx]
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Add channel dimension if missing
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension (C, H, W)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Example usage with your preprocessing function:
def create_data_loaders(n_patients, n_images_per_patient, batch_size=8, transform=None):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        n_patients: Number of patients for preprocessing_v2
        n_images_per_patient: Number of images per patient
        batch_size: Batch size for DataLoader
        transform: PyTorch transforms to apply
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    # Get preprocessed data
    dataset_dict = preprocessing_v2(n_patients, n_images_per_patient)
    
    # Create datasets
    train_dataset = OCTDataset(dataset_dict, transform=transform, train=True)
    val_dataset = OCTDataset(dataset_dict, transform=transform, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader