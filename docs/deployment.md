# PolyGen Deployment Guide

## Production Architecture
- **Backend:** FastAPI running under Uvicorn with multiple worker processes.
- **Frontend:** Pre-built static assets served via NGINX or Caddy reverse proxy.
- **Hardware Integration:** Pass NVIDIA GPUs to container runtime via `--gpus all`.

## Environment Configuration
```env
POLYGEN_HOST=0.0.0.0
POLYGEN_PORT=8001
POLYGEN_ENGINE=auto
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
VAD_THRESHOLD=0.5
VAD_MIN_SILENCE_MS=600
```
