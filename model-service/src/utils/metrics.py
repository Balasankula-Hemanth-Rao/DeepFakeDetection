"""
Metric utilities for deepfake detection evaluation.

Includes AUC, AP, FPR@TPR, aggregation functions, and JSON serialization.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve.
    
    Args:
        labels: Ground truth labels [0/1]
        scores: Predicted scores [0-1]
    
    Returns:
        AUC score
    """
    if roc_auc_score is None:
        raise ImportError("scikit-learn required. Install: pip install scikit-learn")
    
    try:
        return float(roc_auc_score(labels, scores))
    except Exception as e:
        print(f"Failed to compute AUC: {e}")
        return 0.5


def compute_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Average Precision.
    
    Args:
        labels: Ground truth labels
        scores: Predicted scores
    
    Returns:
        AP score
    """
    if average_precision_score is None:
        raise ImportError("scikit-learn required. Install: pip install scikit-learn")
    
    try:
        return float(average_precision_score(labels, scores))
    except Exception as e:
        print(f"Failed to compute AP: {e}")
        return 0.5


def compute_fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    tpr: float = 0.95,
) -> float:
    """
    Compute False Positive Rate at specified True Positive Rate.
    
    Args:
        labels: Ground truth labels
        scores: Predicted scores
        tpr: Target TPR threshold
    
    Returns:
        FPR at given TPR
    """
    # Sort by descending scores
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    # Compute TPR and FPR at each threshold
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    tpr_vals = []
    fpr_vals = []
    
    tp = 0
    fp = 0
    
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        current_tpr = tp / n_pos if n_pos > 0 else 0
        current_fpr = fp / n_neg if n_neg > 0 else 0
        
        tpr_vals.append(current_tpr)
        fpr_vals.append(current_fpr)
    
    # Find FPR at closest TPR >= target
    tpr_vals = np.array(tpr_vals)
    fpr_vals = np.array(fpr_vals)
    
    idx = np.argmin(np.abs(tpr_vals - tpr))
    if tpr_vals[idx] >= tpr:
        return float(fpr_vals[idx])
    else:
        return 1.0


def aggregate_frame_scores(
    scores: np.ndarray,
    method: str = 'mean',
) -> np.ndarray:
    """
    Aggregate per-frame scores (placeholder for video-level aggregation).
    
    In a real scenario, this would group scores by video_id and aggregate.
    For now, we assume one score per sample.
    
    Args:
        scores: Per-sample scores
        method: Aggregation method ('mean', 'max', 'attention')
    
    Returns:
        Aggregated scores (same shape if already aggregated)
    """
    # Placeholder: return as-is since we're already at sample level
    return scores


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics.
    
    Args:
        labels: Ground truth labels
        scores: Predicted scores/probabilities
    
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # AUC and AP
    metrics['auc'] = compute_auc(labels, scores)
    metrics['ap'] = compute_ap(labels, scores)
    
    # Thresholded metrics at 0.5
    predictions = (scores > 0.5).astype(int)
    
    accuracy = np.mean(predictions == labels)
    metrics['accuracy'] = float(accuracy)
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1'] = float(f1)
    
    # FPR@95%TPR
    metrics['fpr_at_95_tpr'] = compute_fpr_at_tpr(labels, scores, tpr=0.95)
    
    # Confusion matrix components
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tn'] = int(tn)
    
    return metrics


def save_metrics_json(metrics: Dict, output_path: Path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dict
        output_path: Path to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
