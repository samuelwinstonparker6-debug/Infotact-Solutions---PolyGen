# Changelog — PolyGen

## [v0.2.5] - Mid-Project Review (2026-11-13)
- Added `ContinuousSpeechAuditor` for multi-corpus transcription evaluation.
- Added Time-to-First-Token (TTFT) latency waterfall profiling engine.
- Built interactive `MidProjectAudit.jsx` panel in the frontend.
- Added comprehensive test suite `test_midproject_review.py` (6/6 passed).

## [v0.2.0] - Week 2 Milestone (2026-11-11)
- Implemented Silero VAD state machine (`vad_service.py`) with turn-taking and barge-in.
- Integrated local Llama 3 conversational engine (`llm_character.py`) pre-prompted as Artificer Nova.
- Built `VoiceCopilot.jsx` component with Web Audio API real-time visualizer.

## [v0.1.0] - Week 1 Milestone (2026-11-09)
- Implemented Wavefront OBJ geometry synthesis (`mesh_generator.py`).
- Built React Three Fiber 3D viewport canvas (`ObjViewer.jsx`).
- Added FastAPI REST backend on port 8001.
