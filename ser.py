import torch
import segmentation_models_pytorch as smp
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])



COLOR_MAP = {
    0: [0, 0, 0],         
    1: [0, 255, 0],      
    2: [128, 128, 128],  
    3: [34, 139, 34],     
    4: [189, 183, 107],   
    5: [244, 164, 96],   
    6: [255, 255, 255],  
    7: [139, 69, 19],    
    8: [255, 0, 0],       
    9: [70, 70, 70],      
    10: [135, 206, 235]   
}


model = smp.DeepLabV3Plus(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=3, 
    classes=11
)
model.load_state_dict(torch.load("falcon_donkey_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image'].split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        image_np = np.array(image)
        
        
        original_h, original_w = image_np.shape[:2]
        
        
        input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        
        mask_resized = cv2.resize(mask.astype('uint8'), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        
        mask_colored = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        for class_idx, color in COLOR_MAP.items():
            mask_colored[mask_resized == class_idx] = color

        
        mask_colored[np.where((mask_colored == [0,0,0]).all(axis=2) & (mask_resized > 0))] = [255, 0, 255]

        
        print(f"Detected class indices: {np.unique(mask)}")

        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "mask": f"data:image/png;base64,{mask_base64}"
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Falcon AI Server is active on http://127.0.0.1:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
