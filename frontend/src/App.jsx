import { useEffect, useState } from 'react'
import {
  Box,
  Cpu,
  Download,
  FileCheck,
  Grid3X3,
  Loader2,
  Mic,
  RotateCw,
  Sparkles,
  Type,
} from 'lucide-react'
import ObjViewer from './components/ObjViewer'
import VoiceCopilot from './components/VoiceCopilot'
import MidProjectAudit from './components/MidProjectAudit'

const SAMPLE_PROMPTS = [
  'A rusted, medieval broadsword with a glowing blue sapphire in the hilt',
  'Classic 1990s Yamaha RX 100 two-stroke motorcycle',
  'A glowing cluster of hexagonal sapphire crystals on dark stone',
  'A heavy iron-banded medieval treasure chest',
  'A domed knight shield with central boss',
  'An alchemist glass potion flask with cork stopper',
]

export default function App() {
  const [activeTab, setActiveTab] = useState('voice') // 'voice' | 'text' | 'audit'
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0])
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [activeAsset, setActiveAsset] = useState(null)
  const [wireframe, setWireframe] = useState(false)
  const [autoRotate, setAutoRotate] = useState(true)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setHealth(d))
      .catch(() => {})

    fetch('/api/history')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.history?.length) {
          setHistory(d.history)
          setActiveAsset(d.history[0])
        }
      })
      .catch(() => {})
  }, [])

  const handleGenerateWithPrompt = async (targetPrompt) => {
    const p = (targetPrompt || prompt).trim()
    if (!p || generating) return
    setGenerating(true)
    setError('')
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: p }),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Mesh generation failed')
      }
      setActiveAsset(data)
      setHistory((prev) => [data, ...prev])
    } catch (err) {
      setError(err.message || 'Could not reach PolyGen backend on port 8001')
    } finally {
      setGenerating(false)
    }
  }

  const handleGenerate = async (e) => {
    e?.preventDefault()
    handleGenerateWithPrompt(prompt)
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Box size={20} />
          </div>
          <div>
            <h1>PolyGen</h1>
            <span>Week 1, 2 & Mid-Project Review</span>
          </div>
        </div>

        {/* Tab Selector: Voice Copilot vs Text Mode vs Mid-Project Audit */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1.2fr',
            gap: '0.35rem',
            background: 'var(--surface-2)',
            padding: '0.3rem',
            borderRadius: '10px',
            border: '1px solid var(--border)',
          }}
        >
          <button
            type="button"
            onClick={() => setActiveTab('voice')}
            className={`tool-btn ${activeTab === 'voice' ? 'active' : ''}`}
            style={{ justifyContent: 'center', width: '100%', padding: '0.35rem 0.4rem', fontSize: '0.74rem' }}
          >
            <Mic size={13} /> Voice
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('text')}
            className={`tool-btn ${activeTab === 'text' ? 'active' : ''}`}
            style={{ justifyContent: 'center', width: '100%', padding: '0.35rem 0.4rem', fontSize: '0.74rem' }}
          >
            <Type size={13} /> Text
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('audit')}
            className={`tool-btn ${activeTab === 'audit' ? 'active' : ''}`}
            style={{ justifyContent: 'center', width: '100%', padding: '0.35rem 0.4rem', fontSize: '0.74rem' }}
          >
            <FileCheck size={13} /> Review Audit
          </button>
        </div>

        {/* Tab 1: Voice Copilot (Silero VAD + Llama 3 Conversational Character) */}
        {activeTab === 'voice' && (
          <VoiceCopilot
            onAssetGenerated={(voicePrompt) => handleGenerateWithPrompt(voicePrompt)}
            onSelectPrompt={(p) => setPrompt(p)}
            isGeneratingMesh={generating}
          />
        )}

        {/* Tab 2: Manual Text Prompt Mode */}
        {activeTab === 'text' && (
          <>
            <form className="panel" onSubmit={handleGenerate}>
              <label
                style={{
                  display: 'block',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  marginBottom: '0.5rem',
                  color: 'var(--muted)',
                }}
              >
                3D ASSET PROMPT
              </label>
              <textarea
                className="prompt-input"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe a 3D mesh to generate (.obj)..."
              />
              <button type="submit" className="btn-primary" disabled={generating}>
                {generating ? <Loader2 size={16} /> : <Sparkles size={16} />}
                {generating ? 'Synthesizing .OBJ Mesh...' : 'Generate Raw .OBJ Mesh'}
              </button>
              {error && (
                <p style={{ marginTop: '0.6rem', fontSize: '0.78rem', color: '#f87171' }}>
                  {error}
                </p>
              )}
            </form>

            <div className="panel">
              <div
                style={{
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: 'var(--muted)',
                  marginBottom: '0.6rem',
                }}
              >
                QUICK PROMPTS
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                {SAMPLE_PROMPTS.map((item) => (
                  <button
                    key={item}
                    type="button"
                    className="chip"
                    onClick={() => {
                      setPrompt(item)
                      handleGenerateWithPrompt(item)
                    }}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Tab 3: Mid-Project Review Audit */}
        {activeTab === 'audit' && <MidProjectAudit />}

        {/* Generated Assets History */}
        <div className="panel" style={{ flex: 1 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '0.6rem',
            }}
          >
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--muted)' }}>
              GENERATED .OBJ HISTORY ({history.length})
            </span>
          </div>
          {history.length === 0 ? (
            <p style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
              No meshes generated yet. Speak to Artificer Nova or submit a prompt above.
            </p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {history.map((item) => (
                <div
                  key={item.asset_id}
                  className={`history-item ${activeAsset?.asset_id === item.asset_id ? 'selected' : ''}`}
                  onClick={() => setActiveAsset(item)}
                >
                  <div style={{ fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.2rem' }}>
                    {item.prompt}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>
                    {item.vertex_count?.toLocaleString()} verts • {item.face_count?.toLocaleString()} faces •{' '}
                    {item.generation_time_sec}s
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      <main className="viewport-area">
        <header className="topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <span className="pill">
              <Cpu size={14} color="#22d3ee" />
              {health
                ? `${health.device.toUpperCase()} (${health.gpu_name})`
                : 'Connecting to backend...'}
            </span>
            {activeAsset && (
              <span className="pill" style={{ color: '#34d399' }}>
                Loaded: {activeAsset.obj_filename}
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              type="button"
              className={`tool-btn ${wireframe ? 'active' : ''}`}
              onClick={() => setWireframe((v) => !v)}
            >
              <Grid3X3 size={14} />
              Wireframe
            </button>
            <button
              type="button"
              className={`tool-btn ${autoRotate ? 'active' : ''}`}
              onClick={() => setAutoRotate((v) => !v)}
            >
              <RotateCw size={14} />
              Auto-Rotate
            </button>
            {activeAsset && (
              <a className="tool-btn" href={activeAsset.download_url} download={activeAsset.obj_filename}>
                <Download size={14} />
                Download .OBJ
              </a>
            )}
          </div>
        </header>

        <div className="canvas-wrap">
          <ObjViewer
            modelUrl={activeAsset?.model_url || null}
            wireframe={wireframe}
            autoRotate={autoRotate}
          />

          {activeAsset && (
            <div className="overlay-stats">
              <div>
                Engine: <strong>{activeAsset.engine_used}</strong>
              </div>
              <div>
                Vertices: <strong>{activeAsset.vertex_count?.toLocaleString()}</strong>
              </div>
              <div>
                Faces: <strong>{activeAsset.face_count?.toLocaleString()}</strong>
              </div>
              <div>
                Time: <strong>{activeAsset.generation_time_sec}s</strong>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
