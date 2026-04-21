# Deployment

## Local (dev)

```bash
conda activate signlang
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -f docker/Dockerfile -t signlang:latest .
docker run --rm -p 8000:8000 \
  -v $(pwd)/models/weights:/app/models/weights:ro \
  -v $(pwd)/database:/app/database \
  signlang:latest
```

The image is CPU-only (`python:3.10-slim` base). Weights and the SQLite DB are
volume-mounted so they persist across container restarts.

## Behind a reverse proxy

WebSocket support requires `proxy_read_timeout` and the standard `Upgrade`
headers:

```nginx
location / {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
}
```

## Production hardening (when you actually ship)

- Replace `allow_origins=["*"]` in `api/main.py` with a real allowlist.
- Move SQLite to Postgres for multi-worker uvicorn (`--workers > 1`).
- Add request-size limits / rate limiting (e.g. `slowapi`) on `/predict`.
- Mount `models/weights/` read-only and run the container as non-root.
