# FaceForensics Dataset Status Report

## ✅ Good News: Official Download Script Available!

You now have the **official FaceForensics++ download script** that can download datasets directly from public servers - **no credentials required!**

**Script Location:** `e:\project\aura-veracity-lab\FaceForensics-master\download-FaceForensics.py`

---

## 📊 Current State

### What You Already Have ✅

**Location:** `e:\project\aura-veracity-lab\FaceForensics-master\original_sequences\youtube\raw\videos\`

**Files Found:** 43 high-quality original videos (.mp4)
- Size: ~27 MB to 387 MB per video
- Total: ~4.2 GB
- Quality: Raw/uncompressed (best quality)
- **These are REAL videos** - perfect for testing!

### What's in model-service/data ⚠️

**Location:** `data/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos/`

**Files Found:** 3 temporary files (should be deleted)
1. `tmp4djliu1y` - 16 KB
2. `tmppqout1sl` - 0 KB
3. `tmprcm7tywq` - 64 KB

❌ **These are NOT valid video files** - likely from incomplete download

---

## 🎯 Recommended: Download Small Dataset

You don't need the full dataset! Here are **3 options** for downloading a small, suitable dataset:

### Option 1: Minimal Test Set (FASTEST) ⚡
- **50 fake + 50 real videos**
- **Size:** ~300 MB
- **Time:** 5-15 minutes
- **Perfect for:** Quick testing, proof-of-concept

**Commands:**
```powershell
cd e:\project\aura-veracity-lab\FaceForensics-master
python download-FaceForensics.py . -d Deepfakes -c c40 -t videos -n 50 --server EU2
python download-FaceForensics.py . -d original -c c40 -t videos -n 50 --server EU2
```

### Option 2: Small Training Set (RECOMMENDED) ✅
- **200 fake + 200 real videos**
- **Size:** ~1.5 GB
- **Time:** 20-45 minutes
- **Perfect for:** Initial training, experimentation

**Commands:**
```powershell
cd e:\project\aura-veracity-lab\FaceForensics-master
python download-FaceForensics.py . -d Deepfakes -c c40 -t videos -n 200 --server EU2
python download-FaceForensics.py . -d original -c c40 -t videos -n 200 --server EU2
```

### Option 3: Full Deepfakes Dataset
- **~1,000 fake + ~1,000 real videos**
- **Size:** ~2-3 GB (c40) or ~10 GB (c23)
- **Time:** 1-3 hours
- **Perfect for:** Serious training, better results

**Commands:**
```powershell
cd e:\project\aura-veracity-lab\FaceForensics-master
# c40 compression (smallest)
python download-FaceForensics.py . -d Deepfakes -c c40 -t videos --server EU2
python download-FaceForensics.py . -d original -c c40 -t videos --server EU2
```

---

## 📁 Expected Dataset Structure

After downloading, you'll have:

```
FaceForensics-master/
├── download-FaceForensics.py          ← Official script
├── manipulated_sequences/
│   └── Deepfakes/
│       └── c40/
│           └── videos/
│               ├── 000_003.mp4        ← FAKE videos
│               ├── 000_870.mp4
│               └── ...
└── original_sequences/
    └── youtube/
        ├── raw/
        │   └── videos/
        │       └── *.mp4              ← 43 videos (already have!)
        └── c40/
            └── videos/
                ├── 000.mp4            ← REAL videos (to download)
                ├── 003.mp4
                └── ...
```

---

## 🚀 Quick Start Guide

**I recommend Option 2** - it's the sweet spot:

```powershell
# 1. Navigate to directory
cd e:\project\aura-veracity-lab\FaceForensics-master

# 2. Download 200 fake videos (Deepfakes)
python download-FaceForensics.py . -d Deepfakes -c c40 -t videos -n 200 --server EU2

# 3. Download 200 real videos
python download-FaceForensics.py . -d original -c c40 -t videos -n 200 --server EU2

# 4. Verify download
Get-ChildItem -Path "manipulated_sequences\Deepfakes\c40\videos\*.mp4" | Measure-Object
Get-ChildItem -Path "original_sequences\youtube\c40\videos\*.mp4" | Measure-Object
```

---

## 🔧 Script Parameters Explained

- **`-d`** = Dataset type (`Deepfakes`, `original`, `Face2Face`, etc.)
- **`-c`** = Compression (`c40` = smallest, `c23` = medium, `raw` = largest)
- **`-t`** = File type (`videos`, `masks`, `models`)
- **`-n`** = Number of videos to download (omit for all)
- **`--server`** = Download server (`EU2` recommended, also `EU`, `CA`)

---

## ⚠️ Important Notes

1. **No credentials needed!** The script downloads from public servers
2. **Press any key when prompted** to confirm Terms of Service
3. **Resume capability** - Script skips already downloaded files
4. **You already have 43 raw videos!** - Can use these for testing

---

## 📋 Comparison Table

| Option | Videos | Size | Quality | Training Viability | Download Time |
|--------|--------|------|---------|-------------------|---------------|
| **Option 1** | 100 (50+50) | ~300 MB | Low (c40) | ⚠️ Testing only | 5-15 min |
| **Option 2** | 400 (200+200) | ~1.5 GB | Low (c40) | ✅ Good for initial training | 20-45 min |
| **Option 3a** | 2,000 (1k+1k) | ~3 GB | Low (c40) | ✅ Full training | 1-2 hours |
| **Option 3b** | 2,000 (1k+1k) | ~10 GB | Medium (c23) | ✅✅ Best results | 2-3 hours |

---

## 📖 Additional Resources

**Detailed Guide:** See `C:\Users\heman\.gemini\antigravity\brain\cfe4fee4-7919-4a0c-9992-185c376607a3\small_dataset_guide.md` for comprehensive instructions

**Official Repo:** https://github.com/ondyari/FaceForensics

---

## ✅ Next Steps

1. **Choose your option** (I recommend Option 2)
2. **Run the download commands** above
3. **Verify the download** by counting .mp4 files
4. **Start preprocessing and training!**
