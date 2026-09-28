"""Shared utilities for notebooks."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_data_yaml(yaml_path: str | Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_yolo_labels(label_path: str | Path, img_h: int, img_w: int) -> list[dict]:
    """Load YOLO format labels (class x_center y_center width height) normalized."""
    labels = []
    label_path = Path(label_path)
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            labels.append({"class": cls, "bbox": [x1, y1, x2, y2]})
    return labels


def draw_boxes(img: np.ndarray, boxes: list, class_names: list, colors: list | None = None) -> np.ndarray:
    """Draw bounding boxes on image."""
    img = img.copy()
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 10)))
        colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
    for box in boxes:
        cls = box["class"]
        x1, y1, x2, y2 = map(int, box["bbox"])
        color = colors[cls % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def show_comparison(img: np.ndarray, gt_boxes: list, pred_boxes: list, class_names: list):
    """Show side-by-side GT vs predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(draw_boxes(img, gt_boxes, class_names))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(draw_boxes(img, pred_boxes, class_names))
    axes[1].set_title("Predictions")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def find_best_gt_match(pred_box: list, gt_boxes: list) -> tuple[int, float]:
    """Find best matching GT box by IoU. Returns (gt_index, iou)."""
    best_idx = -1
    best_iou = 0.0
    for i, gt in enumerate(gt_boxes):
        iou = compute_iou(pred_box, gt["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou
