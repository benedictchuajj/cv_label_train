# Aerial Vehicle Detection: SAM-Assisted Labeling & DETR/YOLO Training

Train a DETR or YOLO model on aerial images with SAM 2-assisted labeling.

## Overview

This repository provides a complete pipeline for:
1. **Evaluating** pretrained models on your aerial dataset to decide if fine-tuning is needed
2. **Relabeling** with SAM 2 automatic mask generation for improved segmentation quality
3. **Training** RT-DETR or YOLO11 on the SAM-relabeled dataset

## Dataset

Your dataset is in `aerial_images/` (YOLO format, 8 vehicle classes):
- PMT, articulated-bus, bus, car, freight, motorbike, small bus, truck
- Train/valid/test splits included

## Setup

```bash
# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

## Workflow

### 1. Baseline Evaluation (`notebooks/01_baseline_eval.ipynb`)

Evaluate whether pretrained YOLO/RT-DETR models already work well on your aerial images.

**What it does**:
- Runs pretrained models (YOLO11n/m, RT-DETR-l) on sample images
- Visualizes predictions vs ground truth side-by-side
- Computes per-class precision, recall, F1
- Recommends whether fine-tuning is necessary

**When to use**: Before investing time in relabeling/training.

### 2. SAM-Assisted Relabeling (`notebooks/02_sam_relabel.ipynb`)

Use SAM 2 to generate high-quality segmentation masks and assign classes from existing labels.

**What it does**:
- Runs SAM 2 automatic mask generation on all images
- Matches masks to existing YOLO bounding boxes via IoU
- Exports a new dataset with improved masks in `data/sam_labels/`

**Output**: New YOLO-format dataset with segmentation-quality masks.

**Note**: This overwrites labels in a new directory. Original labels in `aerial_images/` are preserved.

### 3. Training (`notebooks/03_train.ipynb`)

Train RT-DETR (default) or YOLO11 on the SAM-relabeled dataset.

**What it does**:
- Loads pretrained model (RT-DETR-l or YOLO11)
- Trains on SAM-relabeled dataset with configurable hyperparameters
- Visualizes training curves and validation results
- Exports best model for inference

**Hardware**: Works on Apple Silicon (MPS) but is slow. For serious training, use GPU (Colab, cloud, or local NVIDIA).

## Project Structure

```
yolo_train/
├── aerial_images/           # Original dataset (YOLO format)
├── data/
│   └── sam_labels/          # SAM-relabeled dataset (created by notebook 02)
├── helpers/
│   └── ntbk_utils.py        # Shared utilities
├── notebooks/
│   ├── 01_baseline_eval.ipynb
│   ├── 02_sam_relabel.ipynb
│   └── 03_train.ipynb
├── runs/                    # Training outputs (created by notebook 03)
├── pyproject.toml
└── README.md
```

## Configuration

### Notebook 02 (SAM Relabeling)
- `SAM_CHECKPOINT`: SAM 2 model checkpoint (default: `sam2.1_hiera_large.pt`)
- `IOU_THRESHOLD`: Minimum IoU to assign class to mask (default: 0.3)
- `MIN_MASK_AREA`: Filter out tiny masks (default: 100 pixels)

### Notebook 03 (Training)
- `MODEL_TYPE`: `"rtdetr"` or `"yolo"`
- `MODEL_SIZE`: For RT-DETR: `"l"`, `"x"`; For YOLO: `"n"`, `"s"`, `"m"`, `"l"`, `"x"`
- `EPOCHS`: Training epochs (default: 50)
- `BATCH_SIZE`: Reduce if OOM (default: 8)
- `IMG_SIZE`: Input image size (default: 640)

## Tips

- **Start with notebook 01** to see if you even need to train
- **Use GPU** for training if possible (Colab, cloud, or local NVIDIA)
- **Adjust batch size** if you run out of memory
- **Increase epochs** if training curves haven't converged
- **Try different model sizes** (e.g., YOLO11m instead of YOLO11n) for better accuracy

## References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
