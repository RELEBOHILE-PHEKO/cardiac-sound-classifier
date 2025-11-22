import json
import urllib.request

urls = ["http://localhost:8001/health", "http://localhost:8001/metrics"]
for u in urls:
    print('\n---', u, '---')
    try:
        resp = urllib.request.urlopen(u, timeout=5)
        data = resp.read()
        try:
            obj = json.loads(data)
            print(json.dumps(obj, indent=2))
        except Exception:
            print(data.decode('utf-8', errors='replace'))
    except Exception as e:
        print('ERROR:', e)
