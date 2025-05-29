import torch
import matplotlib.pyplot as plt

def save_checkpoint(epoch, val_loss, model, optimizer, checkpoint_dir, img_size):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / f'{img_size}_checkpoint_epoch_{epoch}.pt')

def normalize_to_target(input_img, target_img):
    target_mean = target_img.mean()
    target_std = target_img.std()
    input_mean = input_img.mean()
    input_std = input_img.std()

    eps = 1e-8
    input_std = torch.clamp(input_std, min=eps)
    target_std = torch.clamp(target_std, min=eps)
    
    scale = torch.abs(target_std / input_std)
    
    normalized = ((input_img - input_mean) * scale) + target_mean
    
    return normalized

def compute_low_signal_mask(output, threshold_factor=0.5):
        """Compute mask for low-signal areas based on intensity."""
        mean_intensity = output.mean()
        threshold = threshold_factor * mean_intensity  # Define threshold as a fraction of the mean
        low_signal_mask = (output < threshold).float()
        return low_signal_mask


def visualize_batch(epoch, batch_idx, input_images, output_images, target_images, vis_dir):
        """Visualize a batch of images"""

        fig, axes = plt.subplots(3, min(4, input_images.shape[0]), 
                                figsize=(15, 10))
        
        if input_images.shape[0] == 1:
            axes = axes.reshape(-1, 1)
            
        for i in range(min(4, input_images.shape[0])):
            input_img = input_images[i].cpu().squeeze().numpy()
            output_img = output_images[i].detach().cpu().squeeze().numpy()
            target_img = target_images[i].cpu().squeeze().numpy()
            
            axes[0, i].imshow(input_img, cmap='gray')
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(output_img, cmap='gray')
            axes[1, i].set_title(f'Output {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(target_img, cmap='gray')
            axes[2, i].set_title(f'Target {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        #plt.savefig(vis_dir / f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm import tqdm

class FusionDataset:
    def __init__(self, basedir, size, transform=None, levels=None, n_patients=1):
        self.transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        self.image_groups = []
        self.size = size
        self.levels = levels

        n, m = 0, 0

        for p in os.listdir(basedir):

            n += 1
            if n > 1: # currently diabetes
                break

            level_dir = os.path.join(basedir, p)
            if not os.path.isdir(level_dir):
                continue

            for patient in os.listdir(level_dir):
                m += 1
                print(m)
                if m > n_patients: # currently diabetes
                    break
                patient_dir = os.path.join(level_dir, patient)
                level_0_dir = os.path.join(patient_dir, "FusedImages_Level_0")

                if not os.path.exists(level_0_dir):
                    continue

                if levels is None:
                    num_levels = sum(1 for d in os.listdir(patient_dir) if d.startswith("FusedImages_Level_"))
                else:
                    num_levels = levels

                print(f"Processing patient {patient} in {p} with {num_levels} levels")

                for base_idx in range(len(os.listdir(level_0_dir))):
                    paths = []
                    names = []
                    current_idx = base_idx
                    valid_group = True

                    # Level 0
                    name = f"Fused_Image_Level_0_{base_idx}.tif"
                    path = os.path.join(level_0_dir, name)

                    if not os.path.exists(path):
                        continue

                    paths.append(path)
                    names.append(name)

                    for level in range(1, num_levels):
                        current_idx = current_idx // 2
                        name = f"Fused_Image_Level_{level}_{current_idx}.tif"
                        path = os.path.join(patient_dir, f"FusedImages_Level_{level}", name)

                        if not os.path.exists(path):
                            valid_group = False
                            break

                        paths.append(path)
                        names.append(name)

                    if valid_group:
                        self.image_groups.append((paths, names))

    def _apply_transform(self, img):
        img = Image.fromarray(img) 
        img = self.transform(img)   
        return img
    
    def __len__(self):
        return len(self.image_groups)
    
    def __getitem__(self, idx):
        paths, names = self.image_groups[idx]
        images = []
        
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
                
            img = self._apply_transform(img)
            images.append(img)
        
        stacked_images = torch.stack(images)
        
        return [stacked_images, names] 

def get_dataset(basedir = "../FusedDataset", size=512, levels=None, n_patients=1):

    dataset = FusionDataset(basedir=basedir, size=size, levels=levels, n_patients=n_patients)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.90), int(len(dataset)*0.1)+1])
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(val_set)}")
    val_size = len(val_set)
    split_size = val_size // 2
    remainder = val_size % 2
    val_split = split_size + remainder  # Add remainder to one split
    test_split = split_size
    #val_set, test_set = torch.utils.data.random_split(val_set, [round(int(len(val_set)*0.50)), round(int(len(val_set)*0.50))+1])
    val_set, test_set = torch.utils.data.random_split(val_set, [val_split, test_split])


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader