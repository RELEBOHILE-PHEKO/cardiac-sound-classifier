"""Small helper to prepare uploaded audio files and run the training pipeline.

Usage:
    python tools/retrain_from_uploads.py --uploads-dir data/uploads --target-data data --validation-split 0.2

What it does:
 - Validates `uploads_dir` contains subfolders for class labels (e.g., `normal`, `abnormal`).
 - Copies files into `target_data/train/<class>` and `target_data/validation/<class>` with a simple split.
 - Optionally runs `python -m src.train` to start training.

This script is intentionally conservative (does not delete uploaded files).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from random import shuffle


def prepare_dataset(uploads_dir: Path, target_dir: Path, val_split: float = 0.2, dry_run: bool = False):
    uploads_dir = Path(uploads_dir)
    target_dir = Path(target_dir)

    if not uploads_dir.exists():
        raise FileNotFoundError(f"Uploads directory not found: {uploads_dir}")

    class_dirs = [p for p in uploads_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subfolders found in uploads dir: {uploads_dir}")

    for class_dir in class_dirs:
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg'}]
        if not files:
            print(f"Warning: no audio files in {class_dir}")
            continue

        shuffle(files)
        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        train_target = target_dir / 'train' / class_dir.name
        val_target = target_dir / 'validation' / class_dir.name
        if not dry_run:
            train_target.mkdir(parents=True, exist_ok=True)
            val_target.mkdir(parents=True, exist_ok=True)

        for src in train_files:
            dest = train_target / src.name
            if dry_run:
                print(f"DRY: copy {src} -> {dest}")
            else:
                shutil.copy2(src, dest)

        for src in val_files:
            dest = val_target / src.name
            if dry_run:
                print(f"DRY: copy {src} -> {dest}")
            else:
                shutil.copy2(src, dest)

        print(f"Prepared class '{class_dir.name}': {len(train_files)} train, {len(val_files)} validation")


def run_training(target_dir: Path, epochs: int = 20, batch_size: int = 32):
    # Run training module from repository root. Use sys.executable to ensure venv.
    import subprocess
    import shlex

    cmd = [sys.executable, '-m', 'src.train', str(target_dir), '--epochs', str(epochs), '--batch-size', str(batch_size)]
    print("Starting training:", ' '.join(cmd))
    proc = subprocess.Popen(cmd)
    return proc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare uploads for retraining and optionally launch training.")
    parser.add_argument('--uploads-dir', default='data/uploads', help='Directory with uploaded files (class subfolders)')
    parser.add_argument('--target-data', default='data', help='Target data directory to populate (train/validation)')
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-train', action='store_true', help='Prepare data but do not launch training')

    args = parser.parse_args(argv)

    uploads_dir = Path(args.uploads_dir)
    target_dir = Path(args.target_data)

    try:
        prepare_dataset(uploads_dir, target_dir, val_split=args.validation_split, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return 2

    if args.dry_run or args.no_train:
        print("Done (dry run or no-train).")
        return 0

    proc = run_training(target_dir, epochs=args.epochs, batch_size=args.batch_size)
    print(f"Training process started (pid={proc.pid}).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
