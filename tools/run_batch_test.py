import argparse
import requests
from pathlib import Path


def find_files(folder: Path, limit: int = None):
    if not folder.exists():
        return []
    files = sorted(folder.rglob('*.wav'))
    if limit:
        return files[:limit]
    return files


def main():
    parser = argparse.ArgumentParser(description='Run a small batch predict test against the API')
    parser.add_argument('--url', default='http://127.0.0.1:8000', help='Base API URL')
    parser.add_argument('--folder', default='data/validation', help='Folder containing WAV files')
    parser.add_argument('--limit', type=int, default=5, help='Limit number of files')
    args = parser.parse_args()

    API = args.url.rstrip('/')
    folder = Path(args.folder)

    # Legacy fallback
    if not folder.exists():
        legacy = Path('data/test/validation')
        if legacy.exists():
            print(f"Folder {folder} not found, falling back to {legacy}")
            folder = legacy

    files = find_files(folder, limit=args.limit)
    if not files:
        print(f"No WAV files found in {folder}. Adjust --folder or add files.")
        raise SystemExit(1)

    # Try bulk endpoint first
    try:
        multipart = [("files", (p.name, open(p, "rb"), "audio/wav")) for p in files]
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
        except Exception:
            print("Non-json response:", resp.text)
        if resp.status_code == 404:
            raise requests.HTTPError("404")
    except Exception as e:
        print("Bulk call failed, falling back to per-file /predict — reason:", e)
        results = []
        for p in files:
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


if __name__ == '__main__':
    main()
