import requests
from pathlib import Path

API = "http://127.0.0.1:8000"

# Choose a few small test files from the repo
files = [
    Path("data/test/validation/a0080.wav"),
    Path("data/test/validation/a0071.wav"),
    Path("data/test/validation/a0050.wav"),
]

existing = [p for p in files if p.exists()]
if not existing:
    print("No test WAV files found at expected paths; please adjust the script.")
    raise SystemExit(1)

# Try bulk endpoint first
try:
    multipart = [("files", (p.name, open(p, "rb"), "audio/wav")) for p in existing]
    print(f"Posting {len(multipart)} files to {API}/batch-predict")
    resp = requests.post(f"{API}/batch-predict", files=multipart, timeout=120)
    # close opened file objects
    for _, filetuple in multipart:
        try:
            fobj = filetuple[1]
            fobj.close()
        except Exception:
            pass
    print("Status:", resp.status_code)
    try:
        print(resp.json())
    except Exception as e:
        print("Non-json response:", resp.text)
    if resp.status_code == 404:
        raise requests.HTTPError("404")
except Exception as e:
    print("Bulk call failed, falling back to per-file /predict — reason:", e)
    results = []
    for p in existing:
        try:
            with open(p, "rb") as fh:
                filesingle = {"file": (p.name, fh, "audio/wav")}
                print(f"Posting {p} to {API}/predict")
                r = requests.post(f"{API}/predict", files=filesingle, timeout=60)
                print("->", r.status_code)
                try:
                    results.append(r.json())
                except Exception:
                    results.append({"error": r.text})
        except Exception as ex:
            results.append({"file": str(p), "error": str(ex)})

    print("Per-file results:")
    for r in results:
        print(r)
