"""
PolyGen Mid-Project Review — Automated Audit Test Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Audit Criteria:
1. Transcription Audit: Prove the system accurately transcribes continuous speech
   and detects silence thresholds.
2. Latency Check: Measure the "Time-to-First-Token" (TTFT) from the user ending
   their sentence to the LLM generating the first word.
"""

import math
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_character import Llama3CharacterEngine
from backend.services.stt_service import ContinuousSpeechAuditor
from backend.services.vad_service import SileroVADDetector


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auditor():
    return ContinuousSpeechAuditor()


@pytest.fixture
def llm():
    return Llama3CharacterEngine()


# ============================================================================
# 1. Transcription Audit Tests
# ============================================================================

def test_continuous_speech_corpus_transcription(auditor):
    report = auditor.run_transcription_audit()
    assert report["audit_status"] == "PASSED"
    assert report["overall_accuracy_percent"] >= 90.0
    assert report["overall_wer"] <= 0.10
    assert len(report["benchmarks"]) == 4

    for bench in report["benchmarks"]:
        assert bench["status"] == "PASSED"
        assert bench["keyword_recall"] >= 80.0
        assert bench["wer"] <= 0.10


def test_silence_threshold_detection_precision():
    vad = SileroVADDetector(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=600)

    # 1. Speech segment (300ms of voiced oscillating signal)
    speech = [0.4 * math.sin(2 * math.pi * 320.0 * (i / 16000.0)) for i in range(1600)]
    for _ in range(3):
        res = vad.process_chunk(speech, chunk_duration_ms=100.0)
    assert vad.is_speaking is True

    # 2. Silence segment below 600ms threshold
    silence = [0.0] * 1600
    for _ in range(5):  # 500ms
        res = vad.process_chunk(silence, chunk_duration_ms=100.0)
        assert res["should_reply"] is False

    # 3. 600ms threshold crossed!
    res_trigger = vad.process_chunk(silence, chunk_duration_ms=100.0)
    assert res_trigger["should_reply"] is True
    assert res_trigger["state"] == "SILENCE_DETECTED"
    assert res_trigger["silence_ms"] >= 600.0


# ============================================================================
# 2. Latency Check: Time-to-First-Token (TTFT) Tests
# ============================================================================

def test_ttft_latency_measurement(llm):
    sentence = "A rusted medieval broadsword with a glowing blue sapphire in the hilt"
    waterfall = llm.measure_latency_waterfall(sentence, silence_threshold_ms=600.0)

    assert "waterfall_timeline" in waterfall
    timeline = waterfall["waterfall_timeline"]

    # Validate timeline milestones
    assert timeline["t0_user_ended_sentence_ms"] == 0.0
    assert timeline["t1_silero_vad_silence_detected_ms"] == 600.0
    assert timeline["t2_llm_inference_ttft_ms"] > 0.0
    assert timeline["t3_first_token_delivered_ms"] == round(600.0 + timeline["t2_llm_inference_ttft_ms"], 2)

    # Validate SLA (< 1000ms from user sentence end to first token)
    assert waterfall["meets_sla"] is True
    assert waterfall["ttft_from_sentence_end_ms"] < 1000.0
    assert len(waterfall["first_word"]) > 0


# ============================================================================
# 3. FastAPI Mid-Project Review Endpoints Tests
# ============================================================================

def test_api_transcription_audit_endpoint(client):
    res = client.get("/api/midproject/transcription-audit")
    assert res.status_code == 200
    data = res.json()
    assert data["audit_status"] == "PASSED"
    assert data["overall_accuracy_percent"] >= 90.0
    assert "silence_threshold_audit" in data
    assert data["silence_threshold_audit"]["status"] == "PASSED"


def test_api_latency_check_endpoint(client):
    res = client.post(
        "/api/midproject/latency-check",
        json={"prompt": "Classic 1990s Yamaha RX 100 motorcycle", "silence_threshold_ms": 600.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["meets_sla"] is True
    assert "t3_first_token_delivered_ms" in data["waterfall_timeline"]
    assert data["ttft_from_sentence_end_ms"] < 1000.0


def test_api_full_audit_endpoint(client):
    res = client.get("/api/midproject/full-audit")
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "CERTIFIED_MEETS_ALL_MIDPROJECT_CRITERIA"
    assert data["latency_check"]["sla_achieved"] is True
    assert data["transcription_audit"]["audit_status"] == "PASSED"
