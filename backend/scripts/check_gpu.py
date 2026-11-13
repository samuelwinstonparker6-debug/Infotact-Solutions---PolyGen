"""
PolyGen Week 1 — GPU & Python Environment Diagnostic Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checks PyTorch, CUDA GPU availability, VRAM capacity, Trimesh, and optional
Text-to-3D neural model packages (Shap-E / TripoSR / Diffusers).
"""

import importlib
import sys
import torch


def main() -> None:
    print("=== PolyGen Week 1 Environment Diagnostic ===")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_ok}")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Device     : {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM       : {props.total_memory / (1024 ** 3):.2f} GB")
    else:
        print("Compute Mode   : CPU fallback active (install CUDA PyTorch for GPU acceleration)")

    print("\n--- Dependency Checks ---")
    for pkg in ["fastapi", "uvicorn", "numpy", "trimesh", "diffusers", "shap_e", "tsr"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg:<12} ({ver})")
        except ImportError:
            status = "OPTIONAL (GPU neural backend)" if pkg in ("diffusers", "shap_e", "tsr") else "MISSING"
            print(f"  [--] {pkg:<12} ({status})")


if __name__ == "__main__":
    main()

