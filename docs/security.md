# Security & Input Validation Architecture

PolyGen applies multi-layer security protections across all public endpoints:
- **Path Traversal Guard:** The `/api/models/{filename}` endpoint enforces strict basename validation, stripping directory traversal tokens (`../`, `..\`) and restricting downloads to the verified output directory.
- **Prompt Sanitization:** Incoming prompts are bounded between 2 and 300 characters, preventing prompt injection and memory exhaustion.
- **Audio Buffer Guards:** Incoming PCM samples are clamped to float32 bounds [-1.0, 1.0], preventing floating point overflow.
