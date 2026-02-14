"""
Evaluation script for multimodal deepfake detection.

Computes comprehensive metrics:
- AUC, AP, accuracy, precision, recall, F1
- FPR@95%TPR
- Confusion matrix
- ROC and PR curves (optional visualization)

Usage:
    python src/eval/multimodal_eval.py --checkpoint checkpoints/best.pth --split test
    python src/eval/multimodal_eval.py --checkpoint checkpoints/best.pth --split val --aggregation mean --save-csv results.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.logging_config import setup_logging, get_logger
from src.datasets.multimodal_dataset import MultimodalDeepfakeDataset
from src.models.multimodal_model import MultimodalModel
from src.utils.metrics import (
    compute_auc,
    compute_ap,
    compute_fpr_at_tpr,
    aggregate_frame_scores,
    save_metrics_json,
)


logger = get_logger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    aggregation: str = 'mean',
) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to evaluate on
        aggregation: How to aggregate per-frame scores ('mean', 'max', 'attention')
    
    Returns:
        metrics: Dict with computed metrics
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            metas = batch['meta']
            
            logits = model(video, audio)
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()  # Probability of fake
            
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())
            all_video_ids.extend([m['video_id'] for m in metas])
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # Aggregate per-frame scores if needed
    aggregated_scores = aggregate_frame_scores(all_scores, method=aggregation)
    
    # Compute metrics
    metrics = {
        'auc': compute_auc(all_labels, aggregated_scores),
        'ap': compute_ap(all_labels, aggregated_scores),
    }
    
    # Additional metrics at threshold 0.5
    predictions = (aggregated_scores > 0.5).astype(int)
    accuracy = np.mean(predictions == all_labels)
    metrics['accuracy'] = accuracy
    
    # Precision, recall, F1
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # FPR@95%TPR
    fpr_at_95_tpr = compute_fpr_at_tpr(all_labels, aggregated_scores, tpr=0.95)
    metrics['fpr_at_95_tpr'] = fpr_at_95_tpr
    
    # Confusion matrix
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tn'] = int(tn)
    
    logger.info("Evaluation metrics", extra=metrics)
    
    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Evaluate multimodal deepfake detection model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--split', type=str, default='test', help='Split (test/val)')
    parser.add_argument('--data-root', type=str, default='data/deepfake', help='Data root')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--aggregation', type=str, default='mean', help='Aggregation method')
    parser.add_argument('--save-csv', type=str, default=None, help='Save results CSV')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device)
    config = get_config()
    
    # Load model
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = MultimodalModel.load_for_inference(
        args.checkpoint,
        config=config,
        device=str(device),
    )
    
    # Load data
    dataset = MultimodalDataset(
        data_root=args.data_root,
        split=args.split,
        config=config,
        debug=False,
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    logger.info(f"Evaluating on {len(dataset)} samples")
    
    # Evaluate
    metrics = evaluate_model(
        model,
        data_loader,
        device,
        aggregation=args.aggregation,
    )
    
    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    results_json = results_dir / 'eval_results.json'
    save_metrics_json(metrics, results_json)
    logger.info(f"Saved results to {results_json}")
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = results_dir / args.save_csv
        with open(csv_path, 'w') as f:
            f.write('metric,value\n')
            for key, value in metrics.items():
                f.write(f'{key},{value}\n')
        logger.info(f"Saved CSV to {csv_path}")


if __name__ == '__main__':
    main()
