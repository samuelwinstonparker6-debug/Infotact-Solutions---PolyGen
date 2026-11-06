# PolyGen Frontend — Spatial WebGL Viewport

## Tech Stack
- **Framework:** React 19 + Vite 8
- **3D Engine:** Three.js + @react-three/fiber + @react-three/drei
- **UI Components:** Lucide React icons, Web Audio API analyzers, and responsive dark theme CSS.

## Key Components
- `ObjViewer.jsx`: Loads raw Wavefront `.obj` meshes with Three.js `OBJLoader`, ground grid, and 3-point lighting.
- `VoiceCopilot.jsx`: Live audio capture, Silero VAD state badge, and Artificer Nova dialogue card.
- `MidProjectAudit.jsx`: Live benchmark table displaying WER and TTFT waterfall latency graphs.
