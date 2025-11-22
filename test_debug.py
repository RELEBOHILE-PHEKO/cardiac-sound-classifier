import traceback
try:
    import librosa
    import numpy as np
    
    # Load audio
    wav, sr = librosa.load('data/test/validation/a0001.wav', sr=4000, mono=True)
    print(f'Audio loaded: shape={wav.shape}, sr={sr}')
    
    # Pad/truncate to 5 seconds
    target_len = int(sr * 5)
    if wav.shape[0] < target_len:
        wav = np.pad(wav, (0, target_len - wav.shape[0]), mode='constant')
    else:
        wav = wav[:target_len]
    print(f'After padding: shape={wav.shape}')
    
    # Create mel spectrogram
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, hop_length=256, fmin=20, fmax=2000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f'Mel spectrogram shape: {mel_db.shape}')
    
    # Add dimensions for model
    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.expand_dims(mel_db, axis=0)
    print(f'Final input shape: {mel_db.shape}')
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model('models/cardiac_cnn_model.h5')
    print(f'Model input shape: {model.input_shape}')
    print(f'Model output shape: {model.output_shape}')
    
    # Predict
    result = model.predict(mel_db, verbose=0)
    print(f'Prediction result: {result}')
    print('SUCCESS!')
    
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
