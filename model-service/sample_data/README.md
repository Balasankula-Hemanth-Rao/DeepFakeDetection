# Sample Data

This directory contains minimal sample data for testing and CI/CD purposes.

## Structure

```
sample_data/deepfake/
├── train/
│   ├── video_001/
│   │   ├── video.mp4       (dummy file for testing)
│   │   └── meta.json       (label: 0 - real)
│   └── video_002/
│       ├── video.mp4       (dummy file for testing)
│       └── meta.json       (label: 1 - fake)
└── val/
    └── video_003/
        ├── video.mp4       (dummy file for testing)
        └── meta.json       (label: 0 - real)
```

## Purpose

- **Testing**: Used in unit and integration tests
- **CI/CD**: Quick validation in GitHub Actions or other CI pipelines
- **Documentation**: Demonstrates expected data format

## Notes

- Video files are **dummy files** (not real MP4s)
- Tests use debug mode which limits to 4 samples max
- For real training, provide actual video files

## Using for Training

To run debug training with sample data:

```bash
python src/train/multimodal_train.py \
  --data-root sample_data/deepfake \
  --debug
```

## Creating Your Own Data

To prepare production data:

```
data/deepfake/
├── train/
│   ├── video_id_1/
│   │   ├── video.mp4
│   │   └── meta.json        # { "label": 0/1 }
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

See `README.md` for detailed dataset structure documentation.
