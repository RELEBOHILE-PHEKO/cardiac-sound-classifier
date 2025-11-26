"""Run batch predictions locally using src.prediction.HeartbeatPredictor.

Usage:
    python tools/run_batch_local.py --folder data/validation --limit 100
"""
import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path so `src` imports resolve when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prediction import HeartbeatPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder with audio files")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process")
    parser.add_argument("--model", default="models/cardiac_cnn_model.h5", help="Path to model file")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    predictor = HeartbeatPredictor(args.model)
    predictor.load()
    # Force spectrogram-only inference as a robust fallback in local batch runs.
    # Some ensemble models may expect different waveform input shapes that
    # cause runtime errors on short/odd files; spectrogram-only path is more
    # tolerant and suitable for local smoke tests.
    predictor.is_ensemble = False

    files = list(folder.glob("**/*.wav")) + list(folder.glob("**/*.mp3"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    results = predictor.batch_predict(files)
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
