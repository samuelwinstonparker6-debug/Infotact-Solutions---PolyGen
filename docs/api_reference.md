# PolyGen REST API Reference

## Core Endpoints
- `GET /api/health` — Hardware detection, CUDA status, and VRAM availability.
- `POST /api/generate` — Generates a raw `.obj` 3D mesh from a text prompt.
- `GET /api/history` — Lists generated 3D meshes with vertex, face, and timing telemetry.
- `GET /api/models/{filename}` — Downloads raw Wavefront `.obj` mesh files.
- `POST /api/voice/vad` — Ingests audio samples through the Silero VAD state machine.
- `POST /api/voice/chat` — Transcribed text to Llama 3 Conversational Character.
- `POST /api/voice/pipeline` — End-to-end voice-to-3D asset pipeline.
- `GET /api/midproject/full-audit` — Unified Mid-Project Review audit report.
