"""
PolyGen Week 2 — Unit & Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests:
1. Silero VAD speech probability, silence thresholding (turn-taking reply),
   and barge-in interruption detection.
2. Llama 3 Conversational Character persona prompt & structured [3D_PROMPT: ...] extraction.
3. FastAPI /api/week2/status, /api/voice/vad, /api/voice/chat, and /api/voice/pipeline.
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_character import Llama3CharacterEngine
from backend.services.vad_service import SileroVADDetector


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def vad():
    return SileroVADDetector(
        threshold=0.5,
        min_speech_duration_ms=200,
        min_silence_duration_ms=500,
        sample_rate=16000,
    )


def generate_audio_tone(freq: float = 440.0, amplitude: float = 0.3, num_samples: int = 1600) -> list[float]:
    """Generates synthetic voiced audio samples simulating human speech."""
    t = np.linspace(0, num_samples / 16000.0, num_samples, endpoint=False)
    # Fundamental frequency + harmonics to simulate voiced vowel
    wave = amplitude * (np.sin(2 * math.pi * freq * t) + 0.5 * np.sin(2 * math.pi * 2 * freq * t))
    return wave.tolist()


def generate_silence(num_samples: int = 1600) -> list[float]:
    """Generates ambient silence samples."""
    return [0.0] * num_samples


# ============================================================================
# 1. Silero VAD Tests
# ============================================================================

def test_silero_vad_silence(vad):
    silence = generate_silence()
    res = vad.process_chunk(silence, chunk_duration_ms=100.0)
    assert res["is_speech"] is False
    assert res["speech_prob"] < 0.3
    assert res["state"] == "SILENCE"
    assert res["should_reply"] is False
    assert res["should_interrupt"] is False


def test_silero_vad_speech_detection(vad):
    speech = generate_audio_tone(freq=300.0, amplitude=0.35)
    res = vad.process_chunk(speech, chunk_duration_ms=100.0)
    assert res["is_speech"] is True
    assert res["speech_prob"] >= 0.5
    assert res["state"] == "SPEECH_STARTED"


def test_silero_vad_turn_taking_silence_threshold(vad):
    speech = generate_audio_tone(freq=350.0, amplitude=0.4)
    silence = generate_silence()

    # User speaks for 300ms (3 chunks of 100ms)
    vad.process_chunk(speech, chunk_duration_ms=100.0)
    vad.process_chunk(speech, chunk_duration_ms=100.0)
    vad.process_chunk(speech, chunk_duration_ms=100.0)
    assert vad.is_speaking is True

    # User stops speaking; silence accumulates
    res1 = vad.process_chunk(silence, chunk_duration_ms=200.0)
    assert res1["should_reply"] is False

    res2 = vad.process_chunk(silence, chunk_duration_ms=200.0)
    assert res2["should_reply"] is False

    # 500ms silence reached (threshold was 500ms)
    res3 = vad.process_chunk(silence, chunk_duration_ms=150.0)
    assert res3["should_reply"] is True
    assert res3["state"] == "SILENCE_DETECTED"
    assert vad.is_speaking is False


def test_silero_vad_barge_in_interruption(vad):
    vad.reset()
    speech = generate_audio_tone(freq=280.0, amplitude=0.4)

    # When AI is currently speaking, user starts talking -> interrupt immediately!
    res = vad.process_chunk(speech, chunk_duration_ms=100.0, is_ai_speaking=True)
    assert res["is_speech"] is True
    assert res["should_interrupt"] is True
    assert res["state"] == "INTERRUPTED"


# ============================================================================
# 2. Llama 3 Conversational Character Tests
# ============================================================================

def test_llama3_character_engine():
    engine = Llama3CharacterEngine()
    info = engine.check_connection()
    assert "ollama_online" in info
    assert "character_persona" in info
    assert "Artificer Nova" in info["character_persona"]

    # Test conversational character response & 3D prompt extraction
    res = engine.generate_response("Can you forge a broadsword with glowing blue sapphire?")
    assert res["character"] == "Artificer Nova"
    assert "reply" in res
    assert len(res["reply"]) > 10
    assert "[3D_PROMPT:" in res["reply"]
    assert "refined_3d_prompt" in res
    assert "broadsword" in res["refined_3d_prompt"].lower() or "sword" in res["refined_3d_prompt"].lower()
    assert res["ttft_ms"] > 0


def test_llama3_motorcycle_prompt():
    engine = Llama3CharacterEngine()
    res = engine.generate_response("Make a 1990s Yamaha RX 100 motorcycle")
    assert "rx 100" in res["refined_3d_prompt"].lower() or "yamaha" in res["refined_3d_prompt"].lower()


# ============================================================================
# 3. FastAPI Week 2 Endpoints Tests
# ============================================================================

def test_api_week2_status(client):
    res = client.get("/api/week2/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "active"
    assert "vad" in data
    assert "llm" in data
    assert data["vad"]["threshold"] == 0.5


def test_api_voice_vad_endpoint(client):
    silence = [0.0] * 800
    res = client.post("/api/voice/vad", json={"audio_samples": silence, "chunk_duration_ms": 50.0})
    assert res.status_code == 200
    data = res.json()
    assert data["is_speech"] is False
    assert data["state"] == "SILENCE"


def test_api_voice_chat_endpoint(client):
    res = client.post(
        "/api/voice/chat",
        json={"transcript": "Create a heavy iron-banded medieval treasure chest"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "refined_3d_prompt" in data
    assert "treasure chest" in data["refined_3d_prompt"].lower() or "chest" in data["refined_3d_prompt"].lower()


def test_api_voice_pipeline_end_to_end(client):
    res = client.post(
        "/api/voice/pipeline",
        json={"transcript": "A rusted medieval broadsword with sapphire in the hilt"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "character_reply" in data
    assert "refined_3d_prompt" in data
    assert "asset" in data
    assert data["asset"]["obj_filename"].endswith(".obj")
    assert data["asset"]["vertex_count"] > 0
    assert data["asset"]["face_count"] > 0
