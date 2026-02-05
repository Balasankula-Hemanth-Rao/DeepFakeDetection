import argparse
import torch
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.frame_model import FrameModel

def load_model(model_path: str, device: str = 'auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model on {device}...")
    # Initialize model architecture
    model = FrameModel(num_classes=2)
    
    # Load weights
    try:
        model.load_for_inference(model_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        # Try loading state dict directly just in case
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model, device

def predict_image(model, image_path, device):
    # Transforms must match training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            
            # Map index to class name
            # 0 = fake, 1 = real
            is_fake = (predicted_class.item() == 0)
            label = 'FAKE' if is_fake else 'REAL'
            
            return {
                'label': label,
                'confidence': confidence.item(),
                'fake_probability': probs[0][0].item(),
                'real_probability': probs[0][1].item()
            }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('image_path', type=str, help='Path to image file to test')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to model checkpoint (default: model.pth)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        exit(1)
        
    try:
        model, device = load_model(args.model, args.device)
        result = predict_image(model, args.image_path, device)
        
        if result:
            print("\n" + "="*40)
            print(f"  RESULT: {result['label']}")
            print(f"  Confidence: {result['confidence']*100:.2f}%")
            print("="*40)
            print(f"  Fake Probability: {result['fake_probability']:.4f}")
            print(f"  Real Probability: {result['real_probability']:.4f}")
            print("="*40 + "\n")
            
    except Exception as e:
        print(f"An error occurred: {e}")
