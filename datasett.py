import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FalconOffRoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = self.images 
        self.mapping = {
            0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 600: 5, 
            700: 6, 900: 7, 800: 8, 7100: 9, 10000: 10
        }

    def mask_to_class(self, mask):
        new_mask = np.zeros_like(mask)
        for k, v in self.mapping.items():
            new_mask[mask == k] = v
        return new_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            image = np.array(image)
            mask = np.array(mask)
            mask = self.mask_to_class(mask)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long()
            else:
                mask = mask.long() 
            return image, mask
        except Exception as e:
            print(f"❌ Error at {idx}: {e}")
            return torch.zeros((3, 544, 960)), torch.zeros((544, 960)).long()
