"""
Aqua Vision - Inference Engine (Ultralytics YOLO11)
Handles model loading, fallback logic, and object detection inference.
"""

import os
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


def find_model_path() -> Tuple[str, bool]:
    """
    Locates the best.pt weights file in potential directories.
    Falls back to yolo11n.pt if not found.

    Returns:
        Tuple[str, bool]: (resolved_path, is_fallback)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "model", "best.pt"),
        os.path.join(base_dir, "models", "best.pt"),
        os.path.join(base_dir, "best.pt"),
        os.path.join("model", "best.pt"),
        os.path.join("models", "best.pt"),
        "best.pt",
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return os.path.abspath(path), False

    # Fallback to standard pretrained YOLO11 nano model
    return "yolo11n.pt", True


@st.cache_resource
def load_yolo_model(model_path: str):
    """
    Loads YOLO model from the specified path with Streamlit caching.

    Args:
        model_path: Filepath or name of the YOLO weights.

    Returns:
        YOLO: Ultralytics YOLO model instance.
    """
    from ultralytics import YOLO

    return YOLO(model_path)


def run_inference(
    model,
    image_input: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Runs YOLO11 inference on the enhanced image.

    Args:
        model: Ultralytics YOLO model instance.
        image_input: Enhanced image (RGB np.ndarray).
        conf_threshold: Confidence score threshold (0.0 to 1.0).
        iou_threshold: Intersection over Union (IoU) threshold for NMS.

    Returns:
        Dict with keys:
            - 'annotated_image': np.ndarray in RGB format
            - 'detections_df': pd.DataFrame with detection details
            - 'total_count': int
            - 'class_counts': Dict[str, int]
            - 'highest_conf': float
    """
    results = model.predict(
        source=image_input,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    result = results[0]
    names = result.names

    # result.plot() returns a BGR image array; convert to RGB for proper display
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    highest_conf = 0.0

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, clss):
            cls_raw = names.get(cls_id, f"Class {cls_id}")
            cls_name = cls_raw.title() if isinstance(cls_raw, str) else str(cls_raw)
            score = float(conf)
            if score > highest_conf:
                highest_conf = score

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            detections.append(
                {
                    "Class": cls_name,
                    "Confidence Score": round(score, 4),
                    "Bounding Box (x1, y1, x2, y2)": f"[{x1}, {y1}, {x2}, {y2}]",
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                }
            )

    detections_df = pd.DataFrame(detections)
    total_count = len(detections)

    return {
        "annotated_image": annotated_rgb,
        "detections_df": detections_df,
        "total_count": total_count,
        "class_counts": class_counts,
        "highest_conf": highest_conf,
    }
