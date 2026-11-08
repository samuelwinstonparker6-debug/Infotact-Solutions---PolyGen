import pytest
from backend.services.vad_service import SileroVADDetector

def test_vad_short_click_filtering():
    vad = SileroVADDetector(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=600)
    click_audio = [0.4] * 800  # 50ms click
    res = vad.process_chunk(click_audio, chunk_duration_ms=50.0)
    assert res["is_speech"] is True
    # Click followed by silence should not trigger AI reply because speech was under 250ms
    silence = [0.0] * 1600
    for _ in range(7):
        res_silence = vad.process_chunk(silence, chunk_duration_ms=100.0)
    assert res_silence["should_reply"] is False

def test_vad_reset_state():
    vad = SileroVADDetector()
    vad.is_speaking = True
    vad.silence_accumulated_ms = 450.0
    vad.reset()
    assert vad.is_speaking is False
    assert vad.silence_accumulated_ms == 0.0
