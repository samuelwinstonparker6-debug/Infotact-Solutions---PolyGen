"""
PolyGen — Week 1 & 2 Configuration & Hardware Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects CUDA GPU availability for TripoSR / Shap-E acceleration,
Ollama / vLLM local Llama 3 endpoints, and Silero VAD parameters.
"""

import os
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class PolyGenConfig:
    host: str = os.getenv("POLYGEN_HOST", "0.0.0.0")
    port: int = int(os.getenv("POLYGEN_PORT", "8001"))
    engine_mode: str = os.getenv("POLYGEN_ENGINE", "auto").lower()
    requested_device: str = os.getenv("POLYGEN_DEVICE", "auto").lower()
    base_dir: Path = Path(__file__).resolve().parent
    output_dir: Path = Path(__file__).resolve().parent / os.getenv("POLYGEN_OUTPUT_DIR", "outputs")

    # Week 2: LLM Integration (Llama 3 via Ollama or vLLM)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    vllm_url: str = os.getenv("VLLM_URL", "")

    # Week 2: Silero VAD Configuration
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    vad_min_speech_ms: int = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
    vad_min_silence_ms: int = int(os.getenv("VAD_MIN_SILENCE_MS", "600"))
    sample_rate: int = int(os.getenv("VAD_SAMPLE_RATE", "16000"))

    @property
    def cuda_available(self) -> bool:
        return torch.cuda.is_available()

    @property
    def device(self) -> str:
        if self.requested_device in ("cuda", "cpu"):
            if self.requested_device == "cuda" and not self.cuda_available:
                return "cpu"
            return self.requested_device
        return "cuda" if self.cuda_available else "cpu"

    @property
    def gpu_name(self) -> str:
        if self.cuda_available:
            return torch.cuda.get_device_name(0)
        return "CPU Fallback (No CUDA GPU detected)"

    @property
    def vram_gb(self) -> float:
        if self.cuda_available:
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 ** 3), 2)
        return 0.0


def get_config() -> PolyGenConfig:
    cfg = PolyGenConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg
