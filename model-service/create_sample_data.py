"""Create sample training data for model service."""
from PIL import Image
import numpy as np
from pathlib import Path

# Create directories
Path('data/sample/train/real').mkdir(parents=True, exist_ok=True)
Path('data/sample/train/fake').mkdir(parents=True, exist_ok=True)

# Create 4 sample real images
for i in range(4):
    img_data = (np.random.rand(64, 64, 3) * 255).astype('uint8')
    Image.fromarray(img_data).save(f'data/sample/train/real/r{i}.jpg')
    print(f'✓ Created real image {i}')

# Create 4 sample fake images
for i in range(4):
    img_data = (np.random.rand(64, 64, 3) * 255).astype('uint8')
    Image.fromarray(img_data).save(f'data/sample/train/fake/f{i}.jpg')
    print(f'✓ Created fake image {i}')

print('\n✓ Sample data created successfully!')
