# Three.js & React Three Fiber Integration Guide

## Viewport Architecture
PolyGen implements an optimized Three.js scene graph in `frontend/src/components/ObjViewer.jsx`:
- **Lighting Model:**
  - Ambient Light: Soft ambient fill preventing pitch-black shadows.
  - Hemisphere Light: Simulates sky/ground ambient gradients.
  - Directional Lights: Dual key and rim lights accentuating geometric silhouettes.
- **Controls:** OrbitControls with damping, auto-rotation toggle, and bounds centering.
- **Material Modes:** Toggle between PBR standard material and wireframe topology mode.
