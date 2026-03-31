import torch
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import FalconOffRoadDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm 
import ttach as tta 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_IMG_PATH = os.path.join(BASE_DIR, "Offroad_Segmentation_testImages", "Offroad_Segmentation_testImages", "Color_Images")
TEST_MASK_PATH = os.path.join(BASE_DIR, "Offroad_Segmentation_testImages", "Offroad_Segmentation_testImages", "Segmentation")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "falcon_donkey_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 11

eval_transform = A.Compose([
    A.Resize(544, 960), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_model():
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b5", classes=NUM_CLASSES).to(DEVICE)
    
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"❌ Weights not found at {MODEL_WEIGHTS}")
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    model = tta.SegmentationTTAWrapper(
        model, 
        tta.aliases.flip_transform(), 
        merge_mode='mean'
    )
    
    return model

def calculate_iou(preds, labels, num_classes):
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        inter = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union == 0:
            ious.append(float('nan')) 
        else:
            ious.append(inter / union)
    return ious

def main():
    if not os.path.exists(TEST_IMG_PATH):
        print(f"ERROR: Test images not found at {TEST_IMG_PATH}")
        return

    model = load_model()
    test_dataset = FalconOffRoadDataset(TEST_IMG_PATH, TEST_MASK_PATH, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            ious = calculate_iou(outputs, masks, NUM_CLASSES)
            all_ious.append(ious)

    all_ious_np = np.array(all_ious)
    mean_ious = np.nanmean(all_ious_np, axis=0)

    print("\n" + "="*45)
    print(f"{'CLASS':<15} | {'IoU SCORE':<10} | {'PERCENTAGE'}")
    print("-" * 45)

    classes = [
        "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
        "Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]

    valid_scores = []
    for i, score in enumerate(mean_ious):
        class_name = classes[i]
        if np.isnan(score):
            print(f"{class_name:<15} | {'N/A':<10} | (Not in Test Set)")
        else:
            print(f"{class_name:<15} | {score:.4f}      | {score*100:.2f}%")
            valid_scores.append(score)

    print("="*45)
    if valid_scores:
        total_miou = np.mean(valid_scores)
        print(f"{'OVERALL mIoU':<15} | {total_miou:.4f}      | {total_miou*100:.2f}%")
    else:
        print("No valid classes were found to calculate mIoU.")
    print("="*45)

if __name__ == "__main__":
    main()
