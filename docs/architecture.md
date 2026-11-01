# PolyGen System Architecture

PolyGen is an autonomous Generative Text-to-3D pipeline designed for game developers and 3D artists.
It orchestrates multi-modal AI systems combining local LLMs (Llama 3), low-latency Voice Activity Detection (Silero VAD), and neural/procedural mesh synthesis rendered directly inside a WebGL Three.js canvas.

## System Organ Diagram
- **Audio & Voice Layer:** Web Audio API, Silero VAD state machine (speech onset, turn completion, barge-in).
- **Conversational Intelligence:** Llama 3 via Ollama / vLLM pre-prompted as Artificer Nova.
- **Mesh Synthesis Engine:** GPU-accelerated TripoSR / Shap-E with Trimesh procedural fallback.
- **Interactive Viewport:** React Three Fiber, OrbitControls, 3-point lighting, and OBJLoader.
