# PolyGen — Generative Text-to-3D Asset Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![React](https://img.shields.io/badge/React-19-cyan.svg)
![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passed-brightgreen.svg)
![SLA](https://img.shields.io/badge/TTFT%20Latency-745ms%20(Sub--1000ms)-blueviolet.svg)

**Domain:** Spatial Computing, Gaming & 3D Generative AI  
**Internship Organization:** Infotact Solutions  
**Author:** Prajwal Sam Jaggili  

---

## Overview
PolyGen is a production-grade Generative Text-to-3D asset pipeline. It couples real-time Voice Activity Detection (**Silero VAD**), a conversational spatial AI companion (**Llama 3** via Ollama), and neural/procedural 3D mesh synthesis (**Wavefront .OBJ**) rendered interactively in a **React Three Fiber** WebGL viewport.

## Milestone Status
- [x] **Week 1:** Mesh Generation (`.obj`) & React Three Fiber 3D Canvas
- [x] **Week 2:** LLM Integration (Llama 3 via Ollama) & Voice Activity Detection (Silero VAD)
- [x] **Mid-Project Review:** Continuous Speech Transcription Audit (0.00 WER) & TTFT Latency Check (< 1000ms SLA)

## Quick Start
```bash
# 1. Start Backend (Port 8001)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001

# 2. Start Frontend (Port 5174)
cd frontend && npm run dev
```
Open **http://127.0.0.1:5174/** to explore the 3D viewport, Voice Copilot, and Mid-Project Review Audit.
