from dataset import FalconOffRoadDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_PATH = os.path.join(BASE_DIR, "Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset", "train", "Color_Images")
MASK_PATH = os.path.join(BASE_DIR, "Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset", "train", "Segmentation")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_CLASSES = 11
EPOCHS = 35 
LEARNING_RATE = 5e-5

def get_rare_class_sampler(dataset):
    print("Scanning masks for rare classes...")
    weights = []
    for mask_name in dataset.masks:
        mask_full_path = os.path.join(dataset.mask_dir, mask_name)
        mask = cv2.imread(mask_full_path, 0)
        if mask is not None and np.any(np.isin(mask, [600, 700, 800, 900])):
            weights.append(10.0)
        else:
            weights.append(1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

train_transform = A.Compose([
    A.Resize(544, 960),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class RobustLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.tensor([1.0, 1.5, 3.0, 2.0, 2.5, 5.0, 15.0, 10.0, 5.0, 1.0, 0.1]).to(DEVICE)
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.focal = smp.losses.FocalLoss(mode='multiclass')
        self.ce = nn.CrossEntropyLoss(weight=weights)
    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true) + self.ce(y_pred, y_true)

def main():
    if not os.path.exists(IMG_PATH):
        print(f"ERROR: Path not found: {IMG_PATH}")
        return
    dataset = FalconOffRoadDataset(IMG_PATH, MASK_PATH, transform=train_transform)
    sampler = get_rare_class_sampler(dataset)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b5", encoder_weights="imagenet", classes=NUM_CLASSES).to(DEVICE)
    criterion = RobustLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    print(f"--- Starting Training on {DEVICE} ---")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        print(f" Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f} | LR: {curr_lr:.6f}")
        if (epoch + 1) % 7 == 0:
            torch.save(model.state_dict(), os.path.join(BASE_DIR, f"falcon_epoch_{epoch+1}.pth"))
    final_path = os.path.join(BASE_DIR, "falcon_donkey_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"--- Training Finished! Model saved to {final_path} ---")

if __name__ == "__main__":
    main()
