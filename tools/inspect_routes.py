import json
import importlib
import src.api as api
importlib.reload(api)

routes = []
for r in api.app.routes:
    try:
        methods = sorted(list(getattr(r, 'methods', []) or []))
    except Exception:
        methods = []
    routes.append({'path': getattr(r, 'path', None), 'name': getattr(r, 'name', None), 'methods': methods})

print(json.dumps(routes, indent=2))
