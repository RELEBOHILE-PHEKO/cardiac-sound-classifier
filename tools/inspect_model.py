"""Inspect saved Keras model & sample preprocessing shapes.

Prints:
- predictor.is_ensemble
- model inputs and summary
- sample waveform and mel shapes for the first WAV found under the repo

Run with the project's venv Python: `.venv\Scripts\python.exe tools\inspect_model.py`
"""
from pathlib import Path
import sys
import traceback

# Ensure repo root on sys.path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

print("Repo root:", HERE)

try:
    from src.prediction import HeartbeatPredictor
    from src.preprocessing import AudioPreprocessor
except Exception as e:
    print("Failed to import project modules:", e)
    traceback.print_exc()
    raise

model_path = HERE / "models" / "cardiac_cnn_model.h5"
print("Model path:", model_path)

try:
    predictor = HeartbeatPredictor(model_path.as_posix())
    predictor.load()
    is_ens = getattr(predictor, "is_ensemble", None)
    print("predictor.is_ensemble:", is_ens)

    model = getattr(predictor, "model", None)
    print("Loaded model object type:", type(model))
    if model is not None:
        try:
            print("Model inputs:")
            for inp in model.inputs:
                try:
                    print(" -", inp.name, "shape=", getattr(inp, 'shape', None))
                except Exception:
                    print(" - (could not read input) ", inp)
        except Exception as e:
            print("Error printing model.inputs:", e)

        try:
            print('\nModel summary (first 60 lines):')
            model.summary(print_fn=lambda s: print(s))
        except Exception as e:
            print("Error printing model.summary():", e)

except Exception:
    print("Error loading predictor/model:")
    traceback.print_exc()

# Find a WAV to inspect preprocessing
wav_files = list(HERE.rglob("*.wav"))
if not wav_files:
    print("No .wav files found in repo to test preprocessing.")
    sys.exit(0)

wav = wav_files[0]
print("Using WAV for preprocessing sample:", wav)

try:
    pre = AudioPreprocessor()
    waveform = pre.load_waveform(wav)
    mel = pre.to_mel_spectrogram(waveform)
    print("waveform.shape:", waveform.shape, "dtype:", waveform.dtype)
    print("mel.shape:", mel.shape, "dtype:", mel.dtype)
except Exception:
    print("Error during preprocessing sample:")
    traceback.print_exc()

print("Done.")
