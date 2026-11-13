import { useState, useEffect, useRef } from 'react'
import {
  Mic,
  MicOff,
  Sparkles,
  Volume2,
  VolumeX,
  Radio,
  Zap,
  Bot,
  AlertCircle,
  PlayCircle,
  Loader2,
  Send,
} from 'lucide-react'

const VOICE_PRESETS = [
  'Can you forge a rusted medieval broadsword with a glowing blue sapphire in the hilt?',
  'Build a classic 1990s Yamaha RX 100 two-stroke motorcycle with chrome exhaust and spoke wheels.',
  'Create a glowing cluster of hexagonal sapphire crystals on volcanic dark stone.',
  'Make a heavy iron-banded medieval treasure chest with aged brass fittings.',
  'Design an agile futuristic sci-fi reconnaissance drone with glowing neon sensor visor.',
]

export default function VoiceCopilot({ onAssetGenerated, onSelectPrompt, isGeneratingMesh }) {
  const [micActive, setMicActive] = useState(false)
  const [vadState, setVadState] = useState('IDLE') // IDLE, LISTENING, SPEECH_ACTIVE, SILENCE_DETECTED, INTERRUPTED
  const [speechProb, setSpeechProb] = useState(0)
  const [silenceMs, setSilenceMs] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [characterReply, setCharacterReply] = useState(
    "Greetings, creator! I am Artificer Nova, your 3D smith for PolyGen. Speak your vision into the microphone or test a prompt below—I will refine your spatial ideas and forge them in the 3D viewport!"
  )
  const [refinedPrompt, setRefinedPrompt] = useState(
    'A rusted, medieval broadsword with a glowing blue sapphire in the hilt'
  )
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [isAiSpeaking, setIsAiSpeaking] = useState(false)
  const [engineInfo, setEngineInfo] = useState(null)
  const [ttftMs, setTtftMs] = useState(148)
  const [latencyMs, setLatencyMs] = useState(210)
  const [speechVolume, setSpeechVolume] = useState(0)

  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const streamRef = useRef(null)
  const recognitionRef = useRef(null)
  const vadIntervalRef = useRef(null)

  // Fetch Week 2 status on load
  useEffect(() => {
    fetch('/api/week2/status')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => data && setEngineInfo(data))
      .catch(() => {})
  }, [])

  // Web Speech API initialization
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onresult = (event) => {
        let currentTranscript = ''
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript
        }
        if (currentTranscript.trim()) {
          setTranscript(currentTranscript)
        }
      }

      recognition.onerror = (err) => {
        console.warn('Speech recognition error:', err)
      }

      recognitionRef.current = recognition
    }
  }, [])

  // Start / Stop Microphone & Audio Stream
  const toggleMicrophone = async () => {
    if (micActive) {
      stopMicrophone()
    } else {
      await startMicrophone()
    }
  }

  const startMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 512
      const source = audioCtx.createMediaStreamSource(stream)
      source.connect(analyser)

      audioCtxRef.current = audioCtx
      analyserRef.current = analyser
      setMicActive(true)
      setVadState('LISTENING')

      if (recognitionRef.current) {
        try {
          recognitionRef.current.start()
        } catch (_) {}
      }

      // Start Silero VAD analysis loop (every 100ms)
      const dataArray = new Uint8Array(analyser.frequencyBinCount)
      vadIntervalRef.current = setInterval(async () => {
        analyser.getByteTimeDomainData(dataArray)
        // Convert to normalized float samples [-1.0, 1.0]
        const floatSamples = []
        let sumSquares = 0
        for (let i = 0; i < dataArray.length; i++) {
          const norm = (dataArray[i] - 128) / 128.0
          floatSamples.push(norm)
          sumSquares += norm * norm
        }
        const rms = Math.sqrt(sumSquares / dataArray.length)
        setSpeechVolume(Math.min(100, Math.round(rms * 400)))

        // Query backend Silero VAD endpoint
        try {
          const res = await fetch('/api/voice/vad', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_samples: floatSamples.slice(0, 256),
              chunk_duration_ms: 100.0,
              is_ai_speaking: isAiSpeaking,
            }),
          })
          if (res.ok) {
            const vad = await res.json()
            setSpeechProb(Math.round(vad.speech_prob * 100))
            setSilenceMs(vad.silence_ms)

            if (vad.should_interrupt) {
              setVadState('INTERRUPTED')
              setIsAiSpeaking(false)
              window.speechSynthesis?.cancel()
            } else if (vad.should_reply) {
              setVadState('SILENCE_DETECTED')
              // Silence threshold passed -> automatically trigger conversational character reply!
              handleSileroTurnTrigger()
            } else if (vad.is_speech) {
              setVadState('SPEECH_ACTIVE')
            } else if (vadState !== 'SILENCE_DETECTED') {
              setVadState('LISTENING')
            }
          }
        } catch (e) {
          console.debug('VAD ping error:', e)
        }
      }, 120)
    } catch (err) {
      console.warn('Microphone access denied or unavailable:', err)
      // Fallback: Enable simulated voice mode
      setMicActive(true)
      setVadState('LISTENING')
    }
  }

  const stopMicrophone = () => {
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current)
      vadIntervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch (_) {}
    }
    setMicActive(false)
    setVadState('IDLE')
    setSpeechVolume(0)
    setSpeechProb(0)
  }

  // Triggered when Silero VAD detects user has finished speaking (silence threshold passed)
  const handleSileroTurnTrigger = () => {
    const textToSend = transcript.trim() || 'A rusted medieval broadsword with a glowing blue sapphire in the hilt'
    dispatchToLlama3Character(textToSend)
  }

  // Dispatch text to Llama 3 Conversational Character
  const dispatchToLlama3Character = async (text) => {
    if (!text.trim() || isAiThinking) return
    setIsAiThinking(true)
    try {
      const res = await fetch('/api/voice/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: text }),
      })
      const data = await res.json()
      if (res.ok) {
        setCharacterReply(data.reply)
        setRefinedPrompt(data.refined_3d_prompt)
        setTtftMs(data.ttft_ms)
        setLatencyMs(data.total_latency_ms)
        onSelectPrompt?.(data.refined_3d_prompt)

        // Speak aloud via browser speech synthesizer
        speakAiReply(data.reply.split('[3D_PROMPT:')[0])
      }
    } catch (err) {
      console.warn('Llama 3 Character error:', err)
    } finally {
      setIsAiThinking(false)
    }
  }

  const speakAiReply = (text) => {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 180))
    utterance.rate = 1.05
    utterance.pitch = 1.0
    utterance.onstart = () => setIsAiSpeaking(true)
    utterance.onend = () => setIsAiSpeaking(false)
    utterance.onerror = () => setIsAiSpeaking(false)
    window.speechSynthesis.speak(utterance)
  }

  // End-to-End: Generate 3D asset from Voice
  const handleGenerate3D = async () => {
    if (!refinedPrompt.trim() || isGeneratingMesh) return
    onAssetGenerated?.(refinedPrompt)
  }

  return (
    <div className="panel voice-panel">
      {/* Header & Status */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          <Radio size={16} color="var(--cyan)" />
          <span style={{ fontSize: '0.78rem', fontWeight: 700, letterSpacing: '0.04em', color: 'var(--text)' }}>
            VOICE COPILOT & SILERO VAD
          </span>
        </div>
        <span
          className="pill"
          style={{
            fontSize: '0.68rem',
            padding: '0.2rem 0.5rem',
            borderColor: engineInfo?.llm?.ollama_online ? 'var(--green)' : 'rgba(99, 102, 241, 0.4)',
            color: engineInfo?.llm?.ollama_online ? 'var(--green)' : 'var(--cyan)',
          }}
        >
          {engineInfo?.llm?.ollama_online ? 'Ollama: Llama-3' : 'Llama 3 Character'}
        </span>
      </div>

      {/* Mic Control Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '0.75rem',
          borderRadius: '10px',
          background: 'rgba(11, 15, 25, 0.65)',
          border: '1px solid var(--border)',
          marginBottom: '0.85rem',
        }}
      >
        <button
          type="button"
          onClick={toggleMicrophone}
          style={{
            width: '42px',
            height: '42px',
            borderRadius: '50%',
            border: micActive ? '2px solid #ef4444' : '1px solid var(--cyan)',
            background: micActive ? 'rgba(239, 68, 68, 0.18)' : 'rgba(34, 211, 238, 0.12)',
            color: micActive ? '#ef4444' : 'var(--cyan)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            flexShrink: 0,
          }}
          title={micActive ? 'Mute Microphone' : 'Start Voice Listening with Silero VAD'}
        >
          {micActive ? <Mic size={20} /> : <MicOff size={20} />}
        </button>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text)' }}>
              {vadState === 'IDLE' && 'Mic Inactive — Click to Speak'}
              {vadState === 'LISTENING' && 'Listening for Voice...'}
              {vadState === 'SPEECH_ACTIVE' && `User Speaking (${speechProb}%)`}
              {vadState === 'SILENCE_DETECTED' && 'Silence Detected (AI Turn Triggered)'}
              {vadState === 'INTERRUPTED' && 'Barge-In Interruption Detected!'}
            </span>
            <span style={{ fontSize: '0.68rem', color: 'var(--muted)', fontFamily: 'monospace' }}>
              Prob: {speechProb}%
            </span>
          </div>

          {/* VAD Probability / Audio Level Bar */}
          <div style={{ height: '5px', borderRadius: '4px', background: 'rgba(255, 255, 255, 0.08)', overflow: 'hidden' }}>
            <div
              style={{
                height: '100%',
                width: `${Math.max(speechProb, speechVolume)}%`,
                background:
                  vadState === 'INTERRUPTED'
                    ? '#f59e0b'
                    : vadState === 'SPEECH_ACTIVE'
                    ? '#10b981'
                    : 'var(--cyan)',
                transition: 'width 0.1s ease',
              }}
            />
          </div>
        </div>
      </div>

      {/* Silero VAD Telemetry Badges */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.85rem', flexWrap: 'wrap' }}>
        <span
          className="pill"
          style={{
            fontSize: '0.68rem',
            padding: '0.2rem 0.5rem',
            background: vadState === 'SPEECH_ACTIVE' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.03)',
            color: vadState === 'SPEECH_ACTIVE' ? '#10b981' : 'var(--muted)',
            borderColor: vadState === 'SPEECH_ACTIVE' ? '#10b981' : 'var(--border)',
          }}
        >
          Silero VAD: {vadState}
        </span>
        <span
          className="pill"
          style={{
            fontSize: '0.68rem',
            padding: '0.2rem 0.5rem',
            color: 'var(--muted)',
          }}
        >
          Silence: {silenceMs}ms / 600ms
        </span>
        <span
          className="pill"
          style={{
            fontSize: '0.68rem',
            padding: '0.2rem 0.5rem',
            color: 'var(--cyan)',
          }}
        >
          <Zap size={10} /> TTFT: {ttftMs}ms
        </span>
      </div>

      {/* Conversational Character Bubble ("Artificer Nova") */}
      <div
        style={{
          background: 'rgba(99, 102, 241, 0.07)',
          border: '1px solid rgba(99, 102, 241, 0.25)',
          borderRadius: '10px',
          padding: '0.85rem',
          marginBottom: '0.85rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.45rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Bot size={15} color="#818cf8" />
            <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#c7d2fe' }}>
              Artificer Nova (Llama 3 Character)
            </span>
          </div>
          {isAiSpeaking ? (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: '#10b981' }}>
              <Volume2 size={12} /> Speaking
            </span>
          ) : (
            <span style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>
              Turn-Taking Ready
            </span>
          )}
        </div>

        <p style={{ fontSize: '0.8rem', lineHeight: '1.4', color: 'var(--text)', marginBottom: '0.6rem' }}>
          {isAiThinking ? (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: 'var(--muted)' }}>
              <Loader2 size={14} className="animate-spin" /> Nova is formulating geometry...
            </span>
          ) : (
            characterReply.split('[3D_PROMPT:')[0]
          )}
        </p>

        {/* Refined 3D Asset Prompt */}
        <div
          style={{
            padding: '0.5rem 0.65rem',
            borderRadius: '6px',
            background: 'rgba(11, 15, 25, 0.8)',
            border: '1px solid var(--border)',
            fontSize: '0.74rem',
            color: 'var(--cyan)',
            fontFamily: 'monospace',
            marginBottom: '0.65rem',
            wordBreak: 'break-word',
          }}
        >
          <strong>3D Asset:</strong> {refinedPrompt}
        </div>

        <button
          type="button"
          onClick={handleGenerate3D}
          disabled={isGeneratingMesh}
          className="btn-primary"
          style={{
            padding: '0.5rem 0.8rem',
            fontSize: '0.8rem',
            background: 'linear-gradient(135deg, #10b981, #059669)',
          }}
        >
          {isGeneratingMesh ? <Loader2 size={14} /> : <Sparkles size={14} />}
          {isGeneratingMesh ? 'Generating 3D Model...' : 'Forge 3D Asset in Canvas'}
        </button>
      </div>

      {/* Preset Voice Prompt Test Chips */}
      <div>
        <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--muted)', marginBottom: '0.45rem' }}>
          CLICK TO TEST SPOKEN PROMPTS (SILERO VAD & LLAMA 3)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          {VOICE_PRESETS.map((p, idx) => (
            <div
              key={idx}
              onClick={() => {
                setTranscript(p)
                dispatchToLlama3Character(p)
              }}
              style={{
                fontSize: '0.72rem',
                padding: '0.4rem 0.6rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid var(--border)',
                color: 'var(--muted)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--cyan)'
                e.currentTarget.style.color = 'var(--text)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border)'
                e.currentTarget.style.color = 'var(--muted)'
              }}
            >
              <PlayCircle size={12} color="var(--cyan)" />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {p}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

