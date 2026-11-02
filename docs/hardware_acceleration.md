# Hardware Acceleration & GPU Architecture

PolyGen automatically probes system hardware at boot:
- **CUDA Device Detection:** Checks `torch.cuda.is_available()`, device index, and total VRAM memory.
- **Engine Selection:**
  - If CUDA VRAM >= 6GB: Automatically invokes TripoSR / Shap-E neural radiance field models.
  - If CPU / Low-VRAM: Seamlessly routes prompts through prompt-aware procedural Trimesh synthesis.
- **Zero-Crash Resilience:** Fallback mechanism guarantees 100% API uptime regardless of GPU driver availability.
