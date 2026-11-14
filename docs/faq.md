# Frequently Asked Questions (FAQ)

### Q: Why use Wavefront OBJ format for Week 1?
Wavefront `.obj` is the universal, vendor-neutral 3D geometry format supported by every 3D modeling tool (Blender, Maya, 3ds Max) and game engine (Unreal, Unity, Godot).

### Q: How does Silero VAD know when I stop speaking?
Silero VAD monitors consecutive 100ms audio chunks. When speech probability drops below 0.5 for 600ms continuously, it triggers the turn-taking event to query Llama 3.

### Q: Can PolyGen run without a dedicated GPU?
Yes. When CUDA is unavailable, PolyGen activates its algorithmic procedural generator which synthesizes 3D meshes in milliseconds on standard CPUs.
