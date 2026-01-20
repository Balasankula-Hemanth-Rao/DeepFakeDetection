## Dataset Usage & License Notice

This project uses the FaceForensics and FaceForensics++ datasets for
non-commercial research and educational purposes only.

**IMPORTANT:** The datasets are not included in this repository due to licensing restrictions.
Users must request access directly from the dataset providers and agree to their respective terms of use.

---

## Dataset Information

### What You Have Downloaded

- **Dataset:** FaceForensics++
- **Manipulation Method:** Deepfakes
- **Compression:** c40 (heavy compression, smallest file size)
- **Location:** `data/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos/`

### Dataset Structure

```
data/FaceForensics++/
└── manipulated_sequences/
    └── Deepfakes/
        └── c40/
            └── videos/
                └── [video files]
```

---

## Legal Requirements

### Terms of Use
✓ **Non-commercial research and educational purposes only**  
✓ **No redistribution** - Do not share or re-host the dataset files  
✓ **Proper attribution** - Must cite in any publications or research  
✓ **Individual access** - Each user must obtain their own access credentials

### Citation Requirements

**If you use this project or the FaceForensics datasets in academic work, you MUST cite:**

#### FaceForensics (Original)
```bibtex
@article{roessler2018faceforensics,
  title={FaceForensics: A Large-scale Video Dataset for Forgery Detection in Human Faces},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:1803.09179},
  year={2018}
}
```

#### FaceForensics++ (Enhanced Version)
```bibtex
@inproceedings{roessler2019faceforensicspp,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={1--11},
  year={2019}
}
```

---

## Dataset Details

### FaceForensics++

- **Official Website:** https://github.com/ondyari/FaceForensics
- **Paper:** https://arxiv.org/abs/1901.08971
- **Total Videos:** 5,000 (1,000 real + 4,000 fake)
- **Manipulation Methods:** 
  - Deepfakes (face swapping using autoencoders)
  - Face2Face (facial reenactment)
  - FaceSwap (face swapping, different method)
  - NeuralTextures (texture synthesis)

### Compression Levels

| Level | Quality | Size | Description |
|-------|---------|------|-------------|
| **c0** | Raw/Lossless | ~500GB | Best quality for research |
| **c23** | Light compression | ~38GB | Recommended for training |
| **c40** | Heavy compression | ~10GB | Quick testing, limited storage |

**You are using:** c40 (heavy compression)

---

## Support & Resources

- **FaceForensics GitHub:** https://github.com/ondyari/FaceForensics
- **Contact:** faceforensics@googlegroups.com
- **Issues:** https://github.com/ondyari/FaceForensics/issues

---

## Acknowledgments

We thank the FaceForensics team for making this dataset available to the research community.
