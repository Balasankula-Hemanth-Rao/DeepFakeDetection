# FaceForensics Training on Kaggle
# Dataset structure: train/, val/, test/, src/

# ============================================================================
# STEP 1: Install Packages
# ============================================================================
print("📦 Installing required packages...")
!pip install -q loguru
print("✅ Packages installed!")

# ============================================================================
# STEP 2: Check GPU
# ============================================================================
print("\n🔍 Checking GPU...")
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU! Enable in Settings → Accelerator → GPU")

# ============================================================================
# STEP 3: Setup Paths
# ============================================================================
import os
import sys

# CHANGE THIS to your dataset name!
DATASET_NAME = 'your-dataset-name'  # ← UPDATE THIS!

INPUT_DIR = f'/kaggle/input/{DATASET_NAME}'
WORKING_DIR = '/kaggle/working'

# Verify dataset
print(f"\n📂 Dataset contents:")
if os.path.exists(INPUT_DIR):
    for item in os.listdir(INPUT_DIR):
        item_path = os.path.join(INPUT_DIR, item)
        if os.path.isdir(item_path):
            try:
                count = len(os.listdir(item_path))
                print(f"  📁 {item}/ ({count:,} items)")
            except:
                print(f"  📁 {item}/")
else:
    print(f"❌ Dataset not found at {INPUT_DIR}")
    print("   Make sure you added the dataset to this notebook!")

# ============================================================================
# STEP 4: Copy Source Code
# ============================================================================
print("\n📝 Setting up source code...")
import shutil

src_input = f'{INPUT_DIR}/src'
src_working = f'{WORKING_DIR}/src'

if os.path.exists(src_input):
    if os.path.exists(src_working):
        shutil.rmtree(src_working)
    shutil.copytree(src_input, src_working)
    print(f"✅ Source code copied to {src_working}")
else:
    print(f"❌ Source code not found at {src_input}")
    print("   Make sure you uploaded the 'src' folder to your dataset!")

# Set Python path
sys.path.insert(0, src_working)
sys.path.insert(0, WORKING_DIR)
os.environ['PYTHONPATH'] = WORKING_DIR

# ============================================================================
# STEP 5: Verify Training Data
# ============================================================================
print("\n🔍 Verifying training data...")

train_dir = f'{INPUT_DIR}/train'
train_fake = f'{train_dir}/fake'
train_real = f'{train_dir}/real'

if os.path.exists(train_fake) and os.path.exists(train_real):
    fake_count = len([f for f in os.listdir(train_fake) if f.endswith('.jpg')])
    real_count = len([f for f in os.listdir(train_real) if f.endswith('.jpg')])
    
    print(f"\n📊 Training data:")
    print(f"  - Fake: {fake_count:,}")
    print(f"  - Real: {real_count:,}")
    print(f"  - Total: {fake_count + real_count:,}")
    
    if fake_count > 0 and real_count > 0:
        print("\n✅ Data looks good! Ready for training.")
    else:
        print("\n❌ No images found in train folders!")
else:
    print(f"❌ Training folders not found!")
    print(f"   Expected: {train_fake} and {train_real}")

# ============================================================================
# STEP 6: Start Training
# ============================================================================
print("\n🚀 Starting training...")
print("=" * 70)

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OUTPUT_DIR = f'{WORKING_DIR}/checkpoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nConfiguration:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Data: {train_dir}")
print(f"  - Output: {OUTPUT_DIR}\n")

# Run training
!cd {WORKING_DIR} && python src/train.py --data-dir {INPUT_DIR}/train --epochs 5 --batch-size 32 --lr 0.0001 --output {OUTPUT_DIR}

print("\n" + "=" * 70)
print("✅ Training completed!")
print("=" * 70)

# ============================================================================
# STEP 7: Show Results
# ============================================================================
print("\n💾 Saved checkpoints:")
if os.path.exists(OUTPUT_DIR):
    checkpoint_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pth')]
    if checkpoint_files:
        for f in checkpoint_files:
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**2)
            print(f"  - {f} ({size:.2f} MB)")
    else:
        print("  No checkpoints found.")
else:
    print("  Checkpoint directory not found.")

print("\n✅ Download checkpoints from the 'Output' tab!")
print("   (Top right of the notebook)")
