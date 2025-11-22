from pathlib import Path
import sys
import traceback

p = Path('models/cardiac_cnn_model.h5')
print('MODEL PATH:', p.resolve())
print('EXISTS:', p.exists())
if p.exists():
    print('SIZE (bytes):', p.stat().st_size)
try:
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
    m = tf.keras.models.load_model(p)
    print('Loaded model successfully')
    try:
        m.summary()
    except Exception:
        print('Model summary failed (OK)')
except Exception as e:
    print('ERROR LOADING MODEL:')
    traceback.print_exc()
    sys.exit(1)
print('Done')
