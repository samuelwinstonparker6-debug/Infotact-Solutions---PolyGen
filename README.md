# PolyGen — Generative Text-to-3D Asset Pipeline (Weeks 1 & 2)

**Domain:** Spatial Computing, Gaming & 3D Generative AI  
**Milestones Completed:**
- **Week 1:** Mesh Generation (`.obj`) & React Three Fiber 3D Canvas
- **Week 2:** LLM Integration (Llama 3 via Ollama/vLLM) & Voice Activity Detection (Silero VAD)

---

## Architecture & Milestones Overview

### Week 1 — 3D Mesh Generation & Viewport
- **GPU-Ready Python Environment (`backend/services/mesh_generator.py`)**:
  - Automatically detects NVIDIA CUDA / PyTorch environments.
  - Implements fast Text-to-3D pipeline generating raw `.obj` Wavefront meshes using TripoSR / Shap-E hooks on CUDA GPUs, and prompt-aware Trimesh procedural geometric synthesis on CPU machines.
- **Interactive Three.js Canvas (`frontend/src/components/ObjViewer.jsx`)**:
  - React Three Fiber viewer with `OBJLoader`, 3-point lighting, orbit controls, wireframe mode, auto-rotation, and direct `.obj` file export.

### Week 2 — Conversational LLM Character & Silero VAD
- **Voice Activity Detection (Silero VAD — `backend/services/vad_service.py`)**:
  - Detects continuous speech probability, speech onset, and speech offset.
  - Turn-taking silence threshold (`600ms`) automatically triggers the AI's reply when the user finishes speaking.
  - Full-duplex barge-in interruption detection: instantly halts AI generation or audio playback if the user begins speaking while the AI is responding.
- **Local Llama 3 LLM Integration (`backend/services/llm_character.py`)**:
  - Connects to local Llama 3 model via Ollama (`http://localhost:11434/api/chat`) or vLLM.
  - Pre-prompted as conversational character **"Artificer Nova"** (master 3D asset blacksmith).
  - Converses with the user about 3D topology and aesthetics, and synthesizes refined `[3D_PROMPT: ...]` tags for the 3D generation engine.
  - Real-time performance tracking: Time-to-First-Token (TTFT) and total latency metrics.
- **Voice Copilot UI (`frontend/src/components/VoiceCopilot.jsx`)**:
  - Microphone capture with Web Audio API level metering.
  - Live Silero VAD telemetry: state badge (`Listening`, `Speech Active`, `Silence Detected`, `Interrupted`), probability bar, and silence countdown.
  - Character dialogue bubbles with audio speech synthesis and one-click 3D viewport generation.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Hardware detection (CUDA, VRAM, device) and system status |
| `GET` | `/api/week2/status` | Silero VAD parameters and Ollama / Llama 3 connection status |
| `POST` | `/api/generate` | Generates a raw `.obj` 3D mesh from a text prompt |
| `POST` | `/api/voice/vad` | Processes audio chunks through the Silero VAD state machine |
| `POST` | `/api/voice/chat` | Transcribed text to Llama 3 Conversational Character ("Artificer Nova") |
| `POST` | `/api/voice/pipeline` | Full Voice-to-3D pipeline (Transcript -> Llama 3 Character -> 3D Mesh) |
| `GET` | `/api/history` | Lists recently generated `.obj` 3D assets |
| `GET` | `/api/models/{file}` | Downloads the generated `.obj` 3D model |

---

## Quick Start & Verification

### 1. Run Unit Tests (13/13 Passed)
```bash
cd PolyGen
python -m pytest -v
```

### 2. Start Backend Server (Port `8001`)
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

### 3. Start Frontend Dev Server (Port `5174`)
```bash
cd PolyGen/frontend
npm run dev
```
Open **http://127.0.0.1:5174/** in your browser to experience the 3D Canvas, Silero VAD, and Llama 3 Conversational Character.
