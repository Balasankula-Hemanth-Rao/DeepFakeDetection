# FaceForensics Deepfake Detection Training on Google Colab
# Copy this entire file and paste into Google Colab
# Make sure to enable GPU: Runtime -> Change runtime type -> GPU

# ============================================================================
# STEP 1: Mount Google Drive
# ============================================================================
print("📁 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")

# ============================================================================
# STEP 2: Install Required Packages
# ============================================================================
print("\n📦 Installing required packages...")
!pip install -q torch torchvision timm opencv-python pillow tqdm loguru
print("✅ Packages installed!")

# ============================================================================
# STEP 3: Check GPU Availability
# ============================================================================
print("\n🔍 Checking GPU...")
import torch
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = "cuda"
else:
    print("⚠️ No GPU found! Training will be slow.")
    print("   Go to: Runtime -> Change runtime type -> Select GPU")
    device = "cpu"

# ============================================================================
# STEP 4: Extract Data from Google Drive
# ============================================================================
print("\n📂 Extracting data from Google Drive...")

import zipfile
import os
import shutil

# IMPORTANT: Update these paths to match your Google Drive folder structure
DRIVE_FOLDER = '/content/drive/MyDrive/FaceForensics_Training'

def extract_and_normalize_zip(zip_path, extract_to):
    """
    Extract zip file and normalize Windows paths to Unix paths.
    This handles zip files created on Windows that contain backslashes.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Normalize path separators (replace backslashes with forward slashes)
            normalized_path = member.replace('\\', '/')
            
            # Skip if it's just a directory entry
            if normalized_path.endswith('/'):
                continue
            
            # Create the full target path
            target_path = os.path.join(extract_to, normalized_path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Extract the file
            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

# Extract training data
print("Extracting training data...")
train_zip = f'{DRIVE_FOLDER}/train_data.zip'
if os.path.exists(train_zip):
    print("  Extracting and normalizing paths...")
    extract_and_normalize_zip(train_zip, '/content/data/processed/')
    print("✅ Training data extracted!")
else:
    print(f"❌ File not found: {train_zip}")
    print("   Please upload train_data.zip to Google Drive")

# Extract validation data (optional)
print("Extracting validation data...")
val_zip = f'{DRIVE_FOLDER}/val_data.zip'
if os.path.exists(val_zip):
    print("  Extracting and normalizing paths...")
    extract_and_normalize_zip(val_zip, '/content/data/processed/')
    print("✅ Validation data extracted!")
else:
    print("⚠️ Validation data not found (optional)")

# Extract source code
print("Extracting source code...")
src_zip = f'{DRIVE_FOLDER}/src_code.zip'
if os.path.exists(src_zip):
    print("  Extracting and normalizing paths...")
    extract_and_normalize_zip(src_zip, '/content/')
    print("✅ Source code extracted!")
else:
    print(f"❌ File not found: {src_zip}")
    print("   Please upload src_code.zip to Google Drive")

# ============================================================================
# STEP 5: Verify Data
# ============================================================================
print("\n🔍 Verifying extracted data...")
import os

# Define base paths (must match extraction paths from Step 4)
BASE_DATA_DIR = '/content/data/processed'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'val')

# Show directory structure for debugging
print("\nDirectory structure:")
print(f"Contents of {BASE_DATA_DIR}:")
if os.path.exists(BASE_DATA_DIR):
    for item in os.listdir(BASE_DATA_DIR):
        item_path = os.path.join(BASE_DATA_DIR, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    file_count = len([f for f in os.listdir(subitem_path) if f.endswith('.jpg')])
                    print(f"    📁 {subitem}/ ({file_count:,} images)")
                else:
                    print(f"    📄 {subitem}")
        else:
            print(f"  📄 {item}")
else:
    print(f"  ❌ Directory does not exist: {BASE_DATA_DIR}")

# Count frames using consistent path construction
train_fake_path = os.path.join(TRAIN_DIR, 'fake')
train_real_path = os.path.join(TRAIN_DIR, 'real')

train_fake = len([f for f in os.listdir(train_fake_path) if f.endswith('.jpg')]) if os.path.exists(train_fake_path) else 0
train_real = len([f for f in os.listdir(train_real_path) if f.endswith('.jpg')]) if os.path.exists(train_real_path) else 0

print(f"\n📊 Training data summary:")
print(f"  - Fake frames: {train_fake:,}")
print(f"  - Real frames: {train_real:,}")
print(f"  - Total: {train_fake + train_real:,}")

if train_fake == 0 or train_real == 0:
    print("\n❌ No training data found!")
    print(f"   Checked paths:")
    print(f"   - {train_fake_path} (exists: {os.path.exists(train_fake_path)})")
    print(f"   - {train_real_path} (exists: {os.path.exists(train_real_path)})")
    print("\n   Troubleshooting:")
    print("   1. Check that zip files uploaded correctly to Google Drive")
    print("   2. Verify DRIVE_FOLDER path in Step 4")
    print("   3. Re-run Step 4 extraction")
else:
    print("\n✅ Data looks good! Ready for training.")

# ============================================================================
# STEP 6: Set Python Path
# ============================================================================
import sys
sys.path.insert(0, '/content/src')
sys.path.insert(0, '/content')

# ============================================================================
# STEP 7: Start Training
# ============================================================================
print("\n🚀 Starting training...")
print("=" * 70)

# Training configuration
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OUTPUT_DIR = '/content/checkpoints'

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set PYTHONPATH and run training
# Using ! command for real-time output in Colab
import os
os.environ['PYTHONPATH'] = '/content'

print(f"\nStarting training with {EPOCHS} epochs, batch size {BATCH_SIZE}...\n")

!cd /content && python src/train.py --data-dir /content/data/processed/train --epochs 5 --batch-size 32 --lr 0.0001 --output /content/checkpoints

print("\n" + "=" * 70)
print("✅ Training completed!")
print("=" * 70)

# ============================================================================
# STEP 8: Save Checkpoints to Google Drive (Backup)
# ============================================================================
print("\n💾 Backing up checkpoints to Google Drive...")

import shutil

backup_dir = f'{DRIVE_FOLDER}/checkpoints'
os.makedirs(backup_dir, exist_ok=True)

# Copy all checkpoint files
checkpoint_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pth')]
for ckpt in checkpoint_files:
    src = f'{OUTPUT_DIR}/{ckpt}'
    dst = f'{backup_dir}/{ckpt}'
    shutil.copy(src, dst)
    print(f"  ✅ Backed up: {ckpt}")

print(f"✅ All checkpoints saved to Google Drive!")
print(f"   Location: {backup_dir}")

# ============================================================================
# STEP 9: Download Final Model to Your Computer
# ============================================================================
print("\n⬇️ Downloading final model to your computer...")

from google.colab import files

final_model = f'{OUTPUT_DIR}/final.pth'
if os.path.exists(final_model):
    files.download(final_model)
    print("✅ Model downloaded! Check your Downloads folder.")
else:
    print("⚠️ Final model not found. Check the checkpoint files.")

# ============================================================================
# STEP 10: Display Training Summary
# ============================================================================
print("\n" + "=" * 70)
print("🎉 ALL DONE!")
print("=" * 70)
print("\nWhat you have now:")
print("  ✅ Trained model (downloaded to your computer)")
print("  ✅ Checkpoints backed up to Google Drive")
print("  ✅ Ready to detect deepfakes!")
print("\nNext steps:")
print("  1. Use the downloaded model with the detection scripts")
print("  2. Test on your own images/videos")
print("  3. Enjoy detecting deepfakes!")
print("=" * 70)

# ============================================================================
# OPTIONAL: Quick Test on Sample Image
# ============================================================================
print("\n🧪 Want to test the model? Run the cell below:")
print("""
# Test the trained model on a sample image
import torch
from PIL import Image
from torchvision import transforms
from src.models.frame_model import FrameModel

# Load model
checkpoint = torch.load('/content/checkpoints/final.pth', map_location=device)
model = FrameModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Prepare transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test on a random training image
import random
test_img_path = random.choice(os.listdir('/content/data/processed/train/fake'))
test_img_path = f'/content/data/processed/train/fake/{test_img_path}'

image = Image.open(test_img_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0][pred].item()

result = 'REAL' if pred == 1 else 'FAKE'
print(f"Prediction: {result}")
print(f"Confidence: {conf*100:.2f}%")
print(f"Tested image: {test_img_path}")
""")
