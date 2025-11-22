import numpy as np
import wave
import requests
from io import BytesIO

sr = 4000
t = np.linspace(0,1,int(sr), False)
note = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
# convert to 16-bit PCM
pcm = (note * 32767).astype('<i2').tobytes()
buf = BytesIO()
with wave.open(buf, 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    f.writeframes(pcm)
buf.seek(0)
files = {'file': ('test.wav', buf, 'audio/wav')}
resp = requests.post('http://localhost:8000/predict', files=files)
print('STATUS', resp.status_code)
try:
    print(resp.json())
except Exception:
    print(resp.text)
