"""
LOMO Evaluator

Comprehensive evaluation for Leave-One-Method-Out (LOMO) protocol.
Tracks per-method metrics: AUC, Accuracy, F1-Score, Precision, Recall.

Usage:
    python lomo_evaluator.py \
        --checkpoint checkpoints/lomo_split_1/best.pth \
        --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \
        --output results/lomo_split_1/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.multimodal_lomo_dataset import create_lomo_dataloaders
from models.multimodal_model import MultimodalDeepfakeDetector


class LomoEvaluator:
    """
    Evaluator for LOMO protocol with comprehensive metrics.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(
        self,
        dataloader,
        split_config: Dict
    ) -> Dict:
        """
        Evaluate model on test set and compute metrics.
        
        Args:
            dataloader: Test dataloader
            split_config: LOMO split configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_methods = []
        all_video_ids = []
        
        print(f"Evaluating on {split_config['test_method']} (unseen method)...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                frames = batch['frames'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label']
                
                # Forward pass
                outputs = self.model(video=frames, audio=audio)
                
                # Get probabilities and predictions
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of fake class
                all_labels.extend(labels.numpy())
                all_methods.extend(batch['method'])
                all_video_ids.extend(batch['video_id'])
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, probabilities, labels)
        
        # Add method-specific information
        metrics['test_method'] = split_config['test_method']
        metrics['split_name'] = split_config['split_name']
        metrics['num_samples'] = len(labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Store predictions for analysis
        metrics['predictions'] = {
            'video_ids': all_video_ids,
            'methods': all_methods,
            'true_labels': labels.tolist(),
            'predicted_labels': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted labels
            probabilities: Prediction probabilities
            labels: True labels
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = float(accuracy_score(labels, predictions))
        
        # Precision, Recall, F1 (for binary classification)
        metrics['precision'] = float(precision_score(labels, predictions, average='binary'))
        metrics['recall'] = float(recall_score(labels, predictions, average='binary'))
        metrics['f1_score'] = float(f1_score(labels, predictions, average='binary'))
        
        # AUC-ROC
        try:
            metrics['auc_roc'] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            # Handle case where only one class is present
            metrics['auc_roc'] = 0.0
        
        # Specificity and Sensitivity
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        return metrics
    
    def identify_failure_cases(
        self,
        metrics: Dict,
        top_k: int = 20
    ) -> Dict:
        """
        Identify and analyze failure cases (misclassifications).
        
        Args:
            metrics: Evaluation metrics with predictions
            top_k: Number of worst cases to identify
            
        Returns:
            Dictionary with failure case analysis
        """
        preds = metrics['predictions']
        video_ids = preds['video_ids']
        true_labels = np.array(preds['true_labels'])
        predicted_labels = np.array(preds['predicted_labels'])
        probabilities = np.array(preds['probabilities'])
        
        # Find misclassified samples
        misclassified_idx = np.where(true_labels != predicted_labels)[0]
        
        if len(misclassified_idx) == 0:
            return {'num_failures': 0, 'failure_cases': []}
        
        # Calculate confidence for misclassified samples
        misclassified_confidence = np.abs(probabilities[misclassified_idx] - 0.5)
        
        # Get top-k highest confidence failures (worst mistakes)
        sorted_idx = np.argsort(misclassified_confidence)[::-1][:top_k]
        worst_failures_idx = misclassified_idx[sorted_idx]
        
        failure_cases = []
        for idx in worst_failures_idx:
            failure_cases.append({
                'video_id': video_ids[idx],
                'true_label': int(true_labels[idx]),
                'predicted_label': int(predicted_labels[idx]),
                'probability': float(probabilities[idx]),
                'confidence': float(misclassified_confidence[sorted_idx[len(failure_cases)]])
            })
        
        return {
            'num_failures': len(misclassified_idx),
            'failure_rate': len(misclassified_idx) / len(true_labels),
            'failure_cases': failure_cases
        }


def aggregate_lomo_results(results_dir: Path) -> Dict:
    """
    Aggregate results from all LOMO splits.
    
    Args:
        results_dir: Directory containing individual split results
        
    Returns:
        Aggregated metrics across all splits
    """
    split_results = []
    
    # Find all evaluation results
    for result_file in results_dir.glob("lomo_split_*/evaluation.json"):
        with open(result_file) as f:
            split_results.append(json.load(f))
    
    if not split_results:
        raise ValueError(f"No evaluation results found in {results_dir}")
    
    # Aggregate metrics
    aggregated = {
        'num_splits': len(split_results),
        'splits': []
    }
    
    # Metrics to aggregate
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'specificity', 'sensitivity']
    
    for metric in metric_names:
        values = [r[metric] for r in split_results]
        aggregated[f'{metric}_mean'] = float(np.mean(values))
        aggregated[f'{metric}_std'] = float(np.std(values))
        aggregated[f'{metric}_min'] = float(np.min(values))
        aggregated[f'{metric}_max'] = float(np.max(values))
    
    # Add per-split summaries
    for result in split_results:
        aggregated['splits'].append({
            'split_name': result['split_name'],
            'test_method': result['test_method'],
            'accuracy': result['accuracy'],
            'f1_score': result['f1_score'],
            'auc_roc': result['auc_roc']
        })
    
    return aggregated


def print_metrics_table(metrics: Dict):
    """Print metrics in a formatted table."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {metrics['split_name']}")
    print("="*60)
    print(f"\nTest Method (Unseen): {metrics['test_method']}")
    print(f"Number of Samples: {metrics['num_samples']}")
    print("\nPerformance Metrics:")
    print("-"*60)
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
    print("-"*60)
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                Real    Fake")
    print(f"  Actual Real  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"        Fake   [{cm[1,0]:5d}  {cm[1,1]:5d}]")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LOMO split and compute per-method metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single split
  python lomo_evaluator.py \\
      --checkpoint checkpoints/lomo_split_1/best.pth \\
      --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \\
      --output results/lomo_split_1/
  
  # Aggregate results from all splits
  python lomo_evaluator.py \\
      --aggregate \\
      --results-dir results/ \\
      --output results/lomo_summary.json
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--split-config',
        type=Path,
        help='Path to LOMO split configuration'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for results'
    )
    
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Aggregate results from multiple splits'
    )
    
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Directory containing split results (for aggregation)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    
    parser.add_argument(
        '--identify-failures',
        action='store_true',
        help='Identify and save failure cases'
    )
    
    args = parser.parse_args()
    
    # Aggregation mode
    if args.aggregate:
        if not args.results_dir:
            print("Error: --results-dir required for aggregation")
            return 1
        
        print("Aggregating LOMO results...")
        aggregated = aggregate_lomo_results(args.results_dir)
        
        # Save aggregated results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"✓ Aggregated results saved to: {args.output}")
        
        # Print summary table
        print("\n" + "="*70)
        print("LOMO EVALUATION SUMMARY (All Splits)")
        print("="*70)
        print(f"\nNumber of splits: {aggregated['num_splits']}")
        print("\nAverage Metrics Across All Splits:")
        print("-"*70)
        print(f"  Accuracy:   {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        print(f"  F1-Score:   {aggregated['f1_score_mean']:.4f} ± {aggregated['f1_score_std']:.4f}")
        print(f"  AUC-ROC:    {aggregated['auc_roc_mean']:.4f} ± {aggregated['auc_roc_std']:.4f}")
        print("-"*70)
        print("\nPer-Split Results:")
        for split in aggregated['splits']:
            print(f"  {split['split_name'][:40]:40s}: AUC={split['auc_roc']:.3f}, Acc={split['accuracy']:.3f}, F1={split['f1_score']:.3f}")
        print("="*70 + "\n")
        
        return 0
    
    # Single split evaluation mode
    if not args.checkpoint or not args.split_config:
        print("Error: --checkpoint and --split-config required for evaluation")
        return 1
    
    # Load split configuration
    with open(args.split_config) as f:
        split_config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = MultimodalDeepfakeDetector(
        enable_video=True,
        enable_audio=True
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test dataloader
    print("Creating test dataloader...")
    dataloaders = create_lomo_dataloaders(
        split_config_path=str(args.split_config),
        batch_size=args.batch_size
    )
    
    # Evaluate
    evaluator = LomoEvaluator(model, device)
    metrics = evaluator.evaluate(dataloaders['test'], split_config)
    
    # Identify failure cases if requested
    if args.identify_failures:
        print("\nIdentifying failure cases...")
        failure_analysis = evaluator.identify_failure_cases(metrics, top_k=20)
        metrics['failure_analysis'] = failure_analysis
        print(f"Found {failure_analysis['num_failures']} misclassifications ({failure_analysis['failure_rate']*100:.2f}%)")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_file = args.output / "evaluation.json" if args.output.is_dir() else args.output
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print metrics table
    print_metrics_table(metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
