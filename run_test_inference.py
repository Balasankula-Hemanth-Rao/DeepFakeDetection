"""
run_test_inference.py
=====================
Batch-run predict_video.py over every video in test_data/real/ and
test_data/fake/, then print a precision/recall summary table.

Usage
------
  python run_test_inference.py
  python run_test_inference.py --test-dir test_data --checkpoint model-service/checkpoints/best_model.pth
  python run_test_inference.py --test-dir test_data --report results/test_report.txt

Arguments
---------
  --test-dir    Root test directory with real/ and fake/ subfolders (default: test_data)
  --checkpoint  Path to the model checkpoint (default: model-service/checkpoints/best_model.pth)
  --report      Path to save the summary report  (default: results/test_report.txt)
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

DEFAULT_CHECKPOINT = str(SCRIPT_DIR / "model-service" / "checkpoints" / "best_model.pth")
DEFAULT_TEST_DIR   = str(SCRIPT_DIR / "test_data")
DEFAULT_REPORT     = str(SCRIPT_DIR / "results" / "test_report.txt")


def collect_videos(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def run_single(video: Path, checkpoint: str) -> str | None:
    """
    Call predict_video.py for a single video.
    Returns the 'Prediction: X | Confidence: Y%' line, or None on error.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "predict_video.py"),
         "--video", str(video),
         "--checkpoint", checkpoint],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if line.strip().startswith("Prediction:"):
            return line.strip()
    return None


def parse_prediction(line: str) -> tuple[str, int]:
    """
    Parse 'Prediction: Fake | Confidence: 92%'
    → ('Fake', 92)
    """
    pred, conf = "Unknown", 0
    for part in line.split("|"):
        part = part.strip()
        if part.startswith("Prediction:"):
            pred = part.split(":", 1)[1].strip()
        elif part.startswith("Confidence:"):
            try:
                conf = int(part.split(":", 1)[1].strip().rstrip("%"))
            except ValueError:
                pass
    return pred, conf


def print_table(rows: list[dict], file=None) -> None:
    cols = ["#", "Class", "File", "Prediction", "Conf%", "Correct"]
    widths = {
        "#":          3,
        "Class":      6,
        "File":       50,
        "Prediction": 10,
        "Conf%":       6,
        "Correct":     7,
    }
    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    hdr = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"

    def p(*args): print(*args, file=file)

    p(sep)
    p(hdr)
    p(sep)
    for r in rows:
        correct_mark = "✓" if r["correct"] else "✗"
        row = "|" + "|".join([
            f" {str(r['idx']):<{widths['#']}} ",
            f" {r['cls']:<{widths['Class']}} ",
            f" {r['file'][:widths['File']]:<{widths['File']}} ",
            f" {r['pred']:<{widths['Prediction']}} ",
            f" {str(r['conf']):<{widths['Conf%']}} ",
            f" {correct_mark:<{widths['Correct']}} ",
        ]) + "|"
        p(row)
    p(sep)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch deepfake inference over test_data/ and print accuracy summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--test-dir",   default=DEFAULT_TEST_DIR,   help="Root of test data (contains real/ and fake/)")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    ap.add_argument("--report",     default=DEFAULT_REPORT,     help="Output report path")
    args = ap.parse_args()

    test_root  = Path(args.test_dir).resolve()
    checkpoint = args.checkpoint
    report_path = Path(args.report)

    if not test_root.is_dir():
        print(f"[ERROR] test-dir not found: {test_root}")
        return 1

    real_dir = test_root / "real"
    fake_dir = test_root / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        print(f"[ERROR] Expected real/ and fake/ inside: {test_root}")
        return 1

    real_videos = collect_videos(real_dir)
    fake_videos  = collect_videos(fake_dir)

    if not real_videos and not fake_videos:
        print("[ERROR] No videos found in test_data/real or test_data/fake")
        return 1

    print("=" * 70)
    print("  DeepFake Detection — Batch Test Inference")
    print("=" * 70)
    print(f"  Test dir   : {test_root}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Real videos: {len(real_videos)}")
    print(f"  Fake videos: {len(fake_videos)}")
    print("=" * 70)

    rows: list[dict] = []
    idx = 1

    for cls, videos, expected in [("Real", real_videos, "Real"), ("Fake", fake_videos, "Fake")]:
        for video in videos:
            print(f"[{idx:>2}] {cls:4s}  {video.name[:55]:<55}", end=" ", flush=True)
            line = run_single(video, checkpoint)
            if line is None:
                pred, conf = "Error", 0
                print("→  ERROR")
            else:
                pred, conf = parse_prediction(line)
                tick = "✓" if pred == expected else "✗"
                print(f"→  {pred} ({conf}%)  {tick}")
            rows.append({
                "idx":     idx,
                "cls":     cls,
                "file":    video.name,
                "pred":    pred,
                "conf":    conf,
                "correct": pred == expected,
            })
            idx += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    total   = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    accuracy = correct / total * 100 if total else 0

    real_rows = [r for r in rows if r["cls"] == "Real"]
    fake_rows  = [r for r in rows if r["cls"] == "Fake"]

    tp = sum(1 for r in fake_rows  if r["pred"] == "Fake")   # true  positives (fake correctly → Fake)
    tn = sum(1 for r in real_rows  if r["pred"] == "Real")   # true  negatives (real correctly → Real)
    fp = sum(1 for r in real_rows  if r["pred"] == "Fake")   # false positives
    fn = sum(1 for r in fake_rows  if r["pred"] == "Real")   # false negatives

    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    avg_conf  = sum(r["conf"] for r in rows) / total if total else 0

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n")
    print_table(rows)

    summary_lines = [
        "",
        "=" * 70,
        "  RESULTS SUMMARY",
        "=" * 70,
        f"  Total videos  : {total}",
        f"  Correct       : {correct}  /  {total}",
        f"  Accuracy      : {accuracy:.1f}%",
        f"  Precision     : {precision:.1f}%   (fake class)",
        f"  Recall        : {recall:.1f}%   (fake class)",
        f"  F1-Score      : {f1:.1f}%",
        f"  Avg Confidence: {avg_conf:.1f}%",
        f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        "=" * 70,
    ]

    for line in summary_lines:
        print(line)

    # ── Save report ────────────────────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"DeepFake Detection — Batch Test Report\n")
        f.write(f"Generated : {datetime.now().isoformat()}\n")
        f.write(f"Test dir  : {test_root}\n")
        f.write(f"Checkpoint: {checkpoint}\n\n")
        print_table(rows, file=f)
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n  Report saved to: {report_path.resolve()}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
