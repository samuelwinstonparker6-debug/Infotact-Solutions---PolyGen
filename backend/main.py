"""
PolyGen Week 1 & 2 — FastAPI Text-to-3D, Silero VAD & Llama 3 Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Week 1 Milestone:
- GPU-accelerated environment check (CUDA / CPU fallback)
- Text-to-3D raw `.obj` mesh generation (Shap-E / TripoSR + Trimesh procedural fallback)
- Static `.obj` serving & generation history for React Three Fiber viewer

Week 2 Milestone:
- LLM Integration: Connect transcribed text to local Llama 3 model (via Ollama or vLLM)
  pre-prompted to act as a conversational character ("Artificer Nova" - 3D Forge Master)
- Voice Activity Detection (VAD): Implement Silero VAD to detect exactly when the user
  stops speaking so the AI knows when to interrupt (barge-in) or reply (turn-taking)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import get_config
from backend.services.llm_character import Llama3CharacterEngine
from backend.services.mesh_generator import generate_obj_from_prompt
from backend.services.vad_service import SileroVADDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polygen.api")

config = get_config()

app = FastAPI(
    title="PolyGen — Generative Text-to-3D, Silero VAD & Llama 3 API",
    description="Week 1 & 2: Text-to-3D raw .obj generation, Silero VAD turn-taking, and Llama 3 conversational character integration.",
    version="0.2.0-week2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(config.output_dir)), name="outputs")

# In-memory Week 1 asset history (newest first)
generation_history: list[dict[str, Any]] = []

# Week 2 Singletons
vad_detector = SileroVADDetector(
    threshold=config.vad_threshold,
    min_speech_duration_ms=config.vad_min_speech_ms,
    min_silence_duration_ms=config.vad_min_silence_ms,
    sample_rate=config.sample_rate,
)

llm_engine = Llama3CharacterEngine(
    ollama_url=config.ollama_url,
    model_name=config.ollama_model,
    vllm_url=config.vllm_url,
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class GenerateMeshRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=300, description="Text description of the 3D asset")
    guidance_scale: float = Field(15.0, ge=1.0, le=30.0)
    engine_mode: str | None = Field(None, description="Optional override: auto | shap_e | triposr | procedural")


class VADChunkRequest(BaseModel):
    audio_samples: Optional[List[float]] = Field(None, description="Array of normalized float32 audio samples [-1.0, 1.0]")
    chunk_duration_ms: float = Field(100.0, ge=10.0, le=2000.0, description="Duration in ms of the audio chunk")
    is_ai_speaking: bool = Field(False, description="Whether the AI character is currently speaking or generating")


class VoiceChatRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=1000, description="User spoken or typed transcript")
    conversation_history: Optional[List[dict[str, str]]] = Field(None, description="Prior conversation messages")


class VoicePipelineRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=1000, description="Transcribed voice text")
    guidance_scale: float = Field(15.0, ge=1.0, le=30.0)
    engine_mode: str | None = Field(None, description="Optional override: auto | shap_e | triposr | procedural")


# ============================================================================
# Week 1 Endpoints (Mesh Generation & Serving)
# ============================================================================

@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Return server health, CUDA GPU detection status, and compute environment details."""
    return {
        "status": "healthy",
        "milestones": [
            "Week 1 — Raw .obj Mesh Generation & 3D Canvas",
            "Week 2 — Llama 3 Conversational Character & Silero VAD",
        ],
        "device": config.device,
        "cuda_available": config.cuda_available,
        "gpu_name": config.gpu_name,
        "vram_gb": config.vram_gb,
        "engine_mode": config.engine_mode,
        "outputs_dir": str(config.output_dir),
    }


@app.post("/api/generate")
def generate_mesh(req: GenerateMeshRequest) -> dict[str, Any]:
    """Generate a raw `.obj` 3D mesh from the user's text prompt."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    mode = (req.engine_mode or config.engine_mode).lower()
    result = generate_obj_from_prompt(
        prompt=prompt,
        output_dir=config.output_dir,
        engine_mode=mode,
        guidance_scale=req.guidance_scale,
    )

    entry = {
        "asset_id": result.asset_id,
        "prompt": result.prompt,
        "obj_filename": result.obj_filename,
        "model_url": f"/outputs/{result.obj_filename}",
        "download_url": f"/api/models/{result.obj_filename}",
        "engine_used": result.engine_used,
        "device_used": result.device_used,
        "vertex_count": result.vertex_count,
        "face_count": result.face_count,
        "bounds_size": result.bounds_size,
        "generation_time_sec": result.generation_time_sec,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    generation_history.insert(0, entry)
    logger.info(
        "Generated %s (%d vertices, %d faces) in %.3fs via %s",
        result.obj_filename,
        result.vertex_count,
        result.face_count,
        result.generation_time_sec,
        result.engine_used,
    )
    return entry


@app.get("/api/history")
def get_history() -> dict[str, Any]:
    """List recently generated `.obj` meshes."""
    return {"history": generation_history, "total": len(generation_history)}


@app.get("/api/models/{filename}")
def download_obj_model(filename: str):
    """Download a generated raw `.obj` mesh file."""
    safe_name = filename.replace("/", "").replace("\\", "")
    file_path = config.output_dir / safe_name
    if not file_path.is_file() or file_path.suffix.lower() != ".obj":
        raise HTTPException(status_code=404, detail="Requested .obj model not found.")
    return FileResponse(path=file_path, media_type="text/plain", filename=safe_name)


# ============================================================================
# Week 2 Endpoints: Silero VAD & Llama 3 Character Integration
# ============================================================================

@app.get("/api/week2/status")
def get_week2_status() -> dict[str, Any]:
    """Return status of Silero VAD engine and local Llama 3 / Ollama connection."""
    ollama_info = llm_engine.check_connection()
    return {
        "status": "active",
        "milestone": "Week 2 — Llama 3 Character & Silero VAD",
        "vad": {
            "engine": vad_detector.backend_engine,
            "threshold": vad_detector.threshold,
            "min_speech_ms": vad_detector.min_speech_duration_ms,
            "min_silence_ms": vad_detector.min_silence_duration_ms,
            "sample_rate": vad_detector.sample_rate,
            "is_speaking": vad_detector.is_speaking,
        },
        "llm": ollama_info,
    }


@app.post("/api/voice/vad")
def process_vad_chunk(req: VADChunkRequest) -> dict[str, Any]:
    """
    Process an audio chunk through the Silero VAD state machine.
    Detects user speech onset, ongoing speech, silence thresholds (turn-taking reply),
    and barge-in interruptions.
    """
    samples = req.audio_samples or []
    # If no audio samples sent (e.g. browser Web Audio level simulation)
    if not samples:
        samples = [0.0] * 160

    result = vad_detector.process_chunk(
        audio_chunk=samples,
        chunk_duration_ms=req.chunk_duration_ms,
        is_ai_speaking=req.is_ai_speaking,
    )
    return result


@app.post("/api/voice/vad/reset")
def reset_vad() -> dict[str, Any]:
    """Reset the internal turn-taking state of Silero VAD."""
    vad_detector.reset()
    return {"status": "reset", "is_speaking": False}


@app.post("/api/voice/chat")
def voice_chat(req: VoiceChatRequest) -> dict[str, Any]:
    """
    Connect user speech transcript to local Llama 3 model (via Ollama/vLLM)
    pre-prompted as conversational character Artificer Nova.
    Extracts refined 3D prompt for the spatial pipeline.
    """
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    result = llm_engine.generate_response(
        user_message=transcript,
        conversation_history=req.conversation_history,
    )
    return result


@app.post("/api/voice/pipeline")
def voice_to_3d_pipeline(req: VoicePipelineRequest) -> dict[str, Any]:
    """
    End-to-End Voice-to-3D Pipeline:
    1. Transcribed text processed by Llama 3 Conversational Character
    2. Character formulates dialogue and extracts refined [3D_PROMPT: ...]
    3. Triggers PolyGen .obj mesh generator to render asset in Three.js canvas.
    """
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Voice transcript cannot be empty.")

    # Step 1: Llama 3 Conversational Character
    llm_res = llm_engine.generate_response(user_message=transcript)
    refined_prompt = llm_res.get("refined_3d_prompt") or transcript

    # Step 2: Mesh Generation from refined prompt
    mode = (req.engine_mode or config.engine_mode).lower()
    mesh_res = generate_obj_from_prompt(
        prompt=refined_prompt,
        output_dir=config.output_dir,
        engine_mode=mode,
        guidance_scale=req.guidance_scale,
    )

    entry = {
        "asset_id": mesh_res.asset_id,
        "prompt": mesh_res.prompt,
        "obj_filename": mesh_res.obj_filename,
        "model_url": f"/outputs/{mesh_res.obj_filename}",
        "download_url": f"/api/models/{mesh_res.obj_filename}",
        "engine_used": mesh_res.engine_used,
        "device_used": mesh_res.device_used,
        "vertex_count": mesh_res.vertex_count,
        "face_count": mesh_res.face_count,
        "bounds_size": mesh_res.bounds_size,
        "generation_time_sec": mesh_res.generation_time_sec,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Week 2 metadata
        "voice_transcript": transcript,
        "character_reply": llm_res.get("reply"),
        "character_name": llm_res.get("character"),
        "llm_engine": llm_res.get("engine_used"),
        "ttft_ms": llm_res.get("ttft_ms"),
    }
    generation_history.insert(0, entry)

    return {
        "status": "success",
        "voice_transcript": transcript,
        "character_reply": llm_res.get("reply"),
        "refined_3d_prompt": refined_prompt,
        "ttft_ms": llm_res.get("ttft_ms"),
        "asset": entry,
    }
