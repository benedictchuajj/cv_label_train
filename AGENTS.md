# AGENTS.md

## Project Overview

Repository for training DETR/YOLO models on aerial vehicle images with
SAM 2-assisted labeling. Dataset: `aerial_images/` (Roboflow, YOLO format,
8 vehicle classes: PMT, articulated-bus, bus, car, freight, motorbike,
small bus, truck; train/valid/test splits).

## Environment

- Python 3.11 (conda env `yolo`) — NOT 3.13 (SAM 2 compat issues)
- Apple Silicon (MPS): training works but slow; `amp=True`, small models
- Install: `pip install -e .` (deps in `pyproject.toml`: ultralytics, sam2,
  opencv, matplotlib, nbformat; dev: ruff, mypy)

## Layout

- `notebooks/01_baseline_eval.ipynb` — zero-shot eval of pretrained YOLO11n/m
  + RT-DETR-l (COCO) vs ground truth; per-class P/R/F1 with exact-only COCO
  mapping (car, bus, truck, motorcycle→motorbike). Unmapped classes (PMT,
  articulated-bus, freight, small bus) intentionally show recall=0 to signal
  "training needed".
- `notebooks/02_sam_relabel.ipynb` — full-dataset SAM relabeling (original
  version; superseded by the alternative notebook).
- `notebooks/02_sam_relabel_alternative.ipynb` — **the one to use**: samples
  500 train images (SEED=42), SAM 2 automatic mask gen, class assignment by
  IoU≥0.3 with existing boxes, 80/20 train/val split (400/100) written to
  `data/sam_labels/`; test points to original `aerial_images/test/`.
- `notebooks/03_train.ipynb` — finetune (not scratch) RT-DETR-l (default) or
  YOLO11 from COCO weights on `data/sam_labels/data.yaml`; exports best.pt.
- `helpers/ntbk_utils.py` — shared utils: data.yaml loading, YOLO label
  parsing, bbox drawing, IoU, GT/pred matching.
- `data/sam_labels/` — generated dataset (run nb 02 alternative to create).

## Conventions

- Notebooks are self-configuring: all knobs (paths, model size, thresholds,
  epochs, seeds) in the first/second config cell — keep it that way when
  editing.
- Training hyperparams in nb 03: EPOCHS=50, BATCH_SIZE=8, IMG_SIZE=640,
  PATIENCE=10. Switch architectures via MODEL_TYPE ("rtdetr"|"yolo").
- After code changes, validate notebooks parse with:
  `python -c "import nbformat; nbformat.read('notebooks/X.ipynb', as_version=4)"`
