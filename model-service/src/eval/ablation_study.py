#!/usr/bin/env python3
"""
Modality Ablation Study Script

Quantifies the independent contributions of audio and video to deepfake detection.

Runs three configurations:
1. Video-only: enable_video=True, enable_audio=False
2. Audio-only: enable_video=False, enable_audio=True
3. Multimodal: enable_video=True, enable_audio=True

Generates:
- Structured comparison report (JSON/CSV)
- Performance metrics per modality
- Contribution analysis

Usage:
    python src/eval/ablation_study.py --config config/config.yaml
    python src/eval/ablation_study.py --config config/config.yaml --output results/ablation.json
    python src/eval/ablation_study.py --checkpoint checkpoints/model.pth --splits val test
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.logging_config import setup_logging, get_logger
from src.data.multimodal_dataset import MultimodalDataset
from src.models.multimodal_model import MultimodalModel
from src.eval.multimodal_eval import evaluate_model
from src.utils.metrics import compute_metrics


logger = get_logger(__name__)


class AblationStudy:
    """
    Multimodal ablation study manager.
    
    Runs evaluation with different modality combinations and compares performance.
    """
    
    def __init__(
        self,
        config=None,
        checkpoint_path: Optional[str] = None,
        data_root: str = "data/deepfake",
        splits: List[str] = None,
        device: str = "auto",
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        """
        Initialize ablation study.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to pretrained model checkpoint
            data_root: Root directory of dataset
            splits: Which splits to evaluate (default: ["val", "test"])
            device: Device to use ("auto", "cuda", "cpu")
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
        """
        self.config = config or get_config()
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.splits = splits or ["val", "test"]
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Resolve device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Storage for results
        self.results = {}
    
    def run(self) -> Dict:
        """
        Run full ablation study.
        
        Returns:
            Dictionary with results and comparisons
        """
        logger.info("Starting modality ablation study")
        start_time = datetime.now()
        
        # Define modality configurations
        configs = [
            ("Multimodal", True, True),
            ("Video-only", True, False),
            ("Audio-only", False, True),
        ]
        
        for config_name, enable_video, enable_audio in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {config_name} (video={enable_video}, audio={enable_audio})")
            logger.info(f"{'='*60}")
            
            try:
                split_results = self._evaluate_config(
                    config_name,
                    enable_video,
                    enable_audio,
                )
                self.results[config_name] = split_results
            except Exception as e:
                logger.error(f"Failed to evaluate {config_name}: {e}")
                self.results[config_name] = {"error": str(e)}
        
        # Compile comparison report
        elapsed = (datetime.now() - start_time).total_seconds()
        report = self._compile_report(elapsed)
        
        logger.info(f"\nAblation study completed in {elapsed:.2f}s")
        return report
    
    def _evaluate_config(
        self,
        config_name: str,
        enable_video: bool,
        enable_audio: bool,
    ) -> Dict:
        """
        Evaluate a single modality configuration across all splits.
        
        Args:
            config_name: Name of configuration (for logging)
            enable_video: Whether to use video
            enable_audio: Whether to use audio
        
        Returns:
            Dictionary with per-split results
        """
        split_results = {}
        
        # Load model with modality configuration
        model = self._load_model(enable_video, enable_audio)
        model = model.to(self.device)
        model.eval()
        
        # Evaluate on each split
        for split in self.splits:
            logger.info(f"Evaluating split: {split}")
            
            try:
                dataset = MultimodalDataset(
                    data_root=self.data_root,
                    split=split,
                    config=self.config,
                    debug=False,
                )
                
                if len(dataset) == 0:
                    logger.warning(f"No samples found for split: {split}")
                    split_results[split] = None
                    continue
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                
                # Run evaluation
                metrics = self._evaluate_split(model, loader, enable_video, enable_audio)
                split_results[split] = metrics
                
                logger.info(f"  AUC: {metrics.get('auc', 0):.4f}")
                logger.info(f"  F1:  {metrics.get('f1', 0):.4f}")
                logger.info(f"  ACC: {metrics.get('accuracy', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating split {split}: {e}")
                split_results[split] = {"error": str(e)}
        
        return split_results
    
    def _load_model(self, enable_video: bool, enable_audio: bool) -> MultimodalModel:
        """
        Load model with specific modality configuration.
        
        Args:
            enable_video: Whether to use video
            enable_audio: Whether to use audio
        
        Returns:
            MultimodalModel instance
        """
        model = MultimodalModel(
            config=self.config,
            num_classes=2,
            enable_video=enable_video,
            enable_audio=enable_audio,
        )
        
        # Load pretrained weights if available
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            try:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
                
                # Handle both direct state_dict and checkpoint wrapper
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                
                # Try to load state dict, skipping incompatible keys
                incompatible = model.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    logger.warning(f"Missing keys when loading checkpoint: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")
                
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Using random initialization.")
        
        return model
    
    def _evaluate_split(
        self,
        model: MultimodalModel,
        loader: DataLoader,
        enable_video: bool,
        enable_audio: bool,
    ) -> Dict:
        """
        Evaluate model on a single split.
        
        Args:
            model: Model to evaluate
            loader: Data loader
            enable_video: Whether video is enabled
            enable_audio: Whether audio is enabled
        
        Returns:
            Dictionary with metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Extract modalities based on enabled configuration
                if enable_video and enable_audio:
                    video, audio, labels = batch['video'], batch['audio'], batch['label']
                    logits = model(video.to(self.device), audio.to(self.device))
                elif enable_video:
                    video, labels = batch['video'], batch['label']
                    logits = model(video=video.to(self.device), audio=None)
                else:  # audio-only
                    audio, labels = batch['audio'], batch['label']
                    logits = model(video=None, audio=audio.to(self.device))
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of fake class
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        
        return metrics
    
    def _compile_report(self, elapsed_seconds: float) -> Dict:
        """
        Compile final ablation report with comparisons.
        
        Args:
            elapsed_seconds: Time taken for ablation study
        
        Returns:
            Comprehensive report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "configuration": {
                "data_root": str(self.data_root),
                "checkpoint": str(self.checkpoint_path),
                "splits": self.splits,
                "batch_size": self.batch_size,
                "device": str(self.device),
            },
            "results": self.results,
            "comparison": self._generate_comparison(),
            "analysis": self._generate_analysis(),
        }
        
        return report
    
    def _generate_comparison(self) -> Dict:
        """
        Generate side-by-side performance comparison.
        
        Returns:
            Comparison dictionary
        """
        comparison = {}
        
        # Aggregate metrics across all splits
        for split in self.splits:
            split_comparison = {}
            
            for config_name in ["Multimodal", "Video-only", "Audio-only"]:
                if config_name not in self.results:
                    continue
                
                split_data = self.results[config_name].get(split)
                if split_data is None or "error" in split_data:
                    split_comparison[config_name] = None
                    continue
                
                split_comparison[config_name] = {
                    "auc": split_data.get("auc", 0),
                    "f1": split_data.get("f1", 0),
                    "accuracy": split_data.get("accuracy", 0),
                    "fpr": split_data.get("fpr", 0),
                    "tpr": split_data.get("tpr", 0),
                }
            
            comparison[split] = split_comparison
        
        return comparison
    
    def _generate_analysis(self) -> Dict:
        """
        Generate modality contribution analysis.
        
        Returns:
            Analysis dictionary with insights
        """
        analysis = {
            "modality_contributions": {},
            "insights": [],
        }
        
        # For each split, calculate how much audio/video contribute
        for split in self.splits:
            split_analysis = {}
            
            multimodal_metrics = self.results.get("Multimodal", {}).get(split)
            video_metrics = self.results.get("Video-only", {}).get(split)
            audio_metrics = self.results.get("Audio-only", {}).get(split)
            
            if not (multimodal_metrics and video_metrics and audio_metrics):
                continue
            
            # Skip if any have errors
            if any("error" in m for m in [multimodal_metrics, video_metrics, audio_metrics]):
                continue
            
            mm_auc = multimodal_metrics.get("auc", 0)
            video_auc = video_metrics.get("auc", 0)
            audio_auc = audio_metrics.get("auc", 0)
            
            # Calculate contribution
            # This is a simplified metric: how much does adding one modality improve over the other
            if video_auc > 0:
                audio_contribution = ((mm_auc - video_auc) / video_auc * 100) if video_auc > 0 else 0
            else:
                audio_contribution = (mm_auc - video_auc) * 100
            
            if audio_auc > 0:
                video_contribution = ((mm_auc - audio_auc) / audio_auc * 100) if audio_auc > 0 else 0
            else:
                video_contribution = (mm_auc - audio_auc) * 100
            
            split_analysis = {
                "multimodal_auc": mm_auc,
                "video_auc": video_auc,
                "audio_auc": audio_auc,
                "audio_contribution_pct": round(audio_contribution, 2),
                "video_contribution_pct": round(video_contribution, 2),
                "best_single_modality": "video" if video_auc >= audio_auc else "audio",
                "best_single_modality_auc": max(video_auc, audio_auc),
                "multimodal_improvement": round((mm_auc - max(video_auc, audio_auc)) * 100, 2),
            }
            
            analysis["modality_contributions"][split] = split_analysis
        
        # Generate insights
        analysis["insights"] = self._generate_insights(analysis["modality_contributions"])
        
        return analysis
    
    def _generate_insights(self, contributions: Dict) -> List[str]:
        """
        Generate human-readable insights from ablation results.
        
        Args:
            contributions: Modality contributions per split
        
        Returns:
            List of insight strings
        """
        insights = []
        
        if not contributions:
            return ["No data available for analysis"]
        
        # Aggregate across splits
        all_audio_contrib = []
        all_video_contrib = []
        all_improvements = []
        
        for split_data in contributions.values():
            all_audio_contrib.append(split_data.get("audio_contribution_pct", 0))
            all_video_contrib.append(split_data.get("video_contribution_pct", 0))
            all_improvements.append(split_data.get("multimodal_improvement", 0))
        
        if all_audio_contrib:
            avg_audio = np.mean(all_audio_contrib)
            avg_video = np.mean(all_video_contrib)
            avg_improvement = np.mean(all_improvements)
            
            # Dominant modality
            if avg_video > avg_audio * 1.5:
                insights.append(f"Video is dominant: {avg_video:.1f}% contribution vs audio {avg_audio:.1f}%")
            elif avg_audio > avg_video * 1.5:
                insights.append(f"Audio is dominant: {avg_audio:.1f}% contribution vs video {avg_video:.1f}%")
            else:
                insights.append(f"Balanced contribution: video {avg_video:.1f}%, audio {avg_audio:.1f}%")
            
            # Fusion benefit
            if avg_improvement > 1:
                insights.append(f"Multimodal fusion improves performance by {avg_improvement:.2f}%")
            else:
                insights.append("Multimodal fusion provides minimal improvement")
        
        return insights


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run modality ablation study for deepfake detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eval/ablation_study.py --config config/config.yaml
  python src/eval/ablation_study.py --checkpoint checkpoints/model.pth --splits val test
  python src/eval/ablation_study.py --config config/config.yaml --output results/ablation.json
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/deepfake",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Also save results as CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    for handler in logger.handlers:
        handler.setLevel(log_level)
    logger.setLevel(log_level)
    
    # Load config
    config = get_config()
    
    # Run ablation study
    ablation = AblationStudy(
        config=config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        splits=args.splits,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    report = ablation.run()
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {output_path}")
    
    # Save CSV if requested
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _save_csv_report(report, csv_path)
        logger.info(f"CSV report saved to {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}\n")
    
    if "comparison" in report:
        for split, comparisons in report["comparison"].items():
            print(f"Split: {split}")
            print(f"  {'Config':<20} {'AUC':<10} {'F1':<10} {'ACC':<10}")
            print(f"  {'-'*50}")
            for config_name, metrics in comparisons.items():
                if metrics:
                    auc = metrics.get('auc', 0)
                    f1 = metrics.get('f1', 0)
                    acc = metrics.get('accuracy', 0)
                    print(f"  {config_name:<20} {auc:<10.4f} {f1:<10.4f} {acc:<10.4f}")
            print()
    
    if "analysis" in report and "insights" in report["analysis"]:
        print("Insights:")
        for insight in report["analysis"]["insights"]:
            print(f"  • {insight}")
    
    print(f"{'='*60}\n")


def _save_csv_report(report: Dict, csv_path: Path):
    """Save comparison results as CSV."""
    rows = []
    
    for split, comparisons in report.get("comparison", {}).items():
        for config_name, metrics in comparisons.items():
            if metrics:
                row = {
                    "split": split,
                    "configuration": config_name,
                    "auc": metrics.get('auc', 0),
                    "f1": metrics.get('f1', 0),
                    "accuracy": metrics.get('accuracy', 0),
                    "fpr": metrics.get('fpr', 0),
                    "tpr": metrics.get('tpr', 0),
                }
                rows.append(row)
    
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
