import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add src to path
sys.path.append(r'e:\project\aura-veracity-lab\model-service\src')

try:
    from models.multimodal_model import MultimodalModel, AudioCNN
    from models.frame_model import FrameModel
except ImportError as e:
    print(f"Import Error: {e}")
    pass

def main():
    image_path = r'C:\Users\heman\Downloads\images (2).jpg'
    checkpoint_path = r'e:\project\aura-veracity-lab\model-service\checkpoints\final.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Inspect Checkpoint
    print("Inspecting checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        keys = list(state_dict.keys())
        is_frame_model = any(k.startswith('backbone.') for k in keys)
        is_multimodal = any(k.startswith('video_backbone.') for k in keys)
        
        model = None
        
        if is_frame_model:
            print("Detected FrameModel (Video Only).")
            model = FrameModel(num_classes=2)
            model.load_state_dict(state_dict, strict=True)
            
        elif is_multimodal:
            print("Detected MultimodalModel.")
            # Determine audio usage by classifier shape
            classifier_weight = state_dict.get('classifier.0.weight')
            enable_audio = True
            if classifier_weight is not None:
                if classifier_weight.shape[1] == 1536:
                    enable_audio = False
            
            config = checkpoint.get('config', None)
            model = MultimodalModel(config=config, enable_audio=enable_audio, enable_video=True)
            model.load_state_dict(state_dict, strict=True)
            
        else:
            print("Unknown model type. Keys do not match expected 'backbone' or 'video_backbone'.")
            print(f"Sample keys: {keys[:5]}")
            return

        model.to(device)
        model.eval()

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare Image
    print("Preprocessing image...")
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error reading image: {e}")
        return
        
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Model expects [B, 3, H, W] for FrameModel, but [B, T, 3, H, W] for Multimodal
    img_tensor = preprocess(img).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # Inference
    print("Running inference...")
    try:
        with torch.no_grad():
            if isinstance(model, FrameModel):
                # FrameModel forward takes [B, 3, H, W]
                logits = model(img_tensor)
            else:
                # Multimodal takes [B, T, 3, H, W]
                video_tensor = img_tensor.unsqueeze(1) # [1, 1, 3, H, W]
                
                # Handle audio
                audio_tensor = None
                if model.enable_audio:
                    if isinstance(model.audio_encoder, AudioCNN):
                         n_mels = getattr(model, 'audio_n_mels', 64)
                         audio_tensor = torch.zeros(1, 1, n_mels, 100).to(device)
                    else:
                         audio_tensor = torch.zeros(1, 16000).to(device)
                
                logits = model(video=video_tensor, audio=audio_tensor)
                
            probs = torch.softmax(logits, dim=1)
            confidence_fake = probs[0, 1].item()
            prediction = probs[0].argmax().item()
            
        label = "FAKE" if prediction == 1 else "REAL"
        print(f"\n--- RESULTS ---")
        print(f"File: {os.path.basename(image_path)}")
        print(f"Prediction: {label}")
        print(f"Confidence (Fake): {confidence_fake:.4f}")
        
    except Exception as e:
        print(f"Runtime Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
