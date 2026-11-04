# Voice Activity Detection (Silero VAD) & Turn-Taking

## State Machine
The detector processes 16kHz audio chunks and transitions through the following states:
1. `IDLE` / `SILENCE`: Ambient noise below speech threshold (threshold = 0.50).
2. `SPEECH_STARTED`: User begins speaking (minimum speech duration = 250ms).
3. `SPEECH_ACTIVE`: Continuous user speech accumulation.
4. `SILENCE_DETECTED`: Energy drops below threshold for 600ms, signaling turn completion.
5. `INTERRUPTED`: User speaks while AI is speaking (full-duplex barge-in).
