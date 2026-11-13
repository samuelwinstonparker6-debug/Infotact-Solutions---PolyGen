"""
PolyGen Week 1 — Text-to-3D Raw Mesh Generator (.obj)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Orchestrates Text-to-3D raw `.obj` mesh generation:
1. GPU Neural Backends (Shap-E / TripoSR) when CUDA + model packages are available.
2. Prompt-aware Procedural 3D Synthesizer (Trimesh + NumPy) as an instant, deterministic
   CPU fallback so `.obj` assets can be generated and inspected in React Three Fiber
   even on CPU-only machines.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

logger = logging.getLogger("polygen.mesh_generator")


@dataclass
class MeshGenerationResult:
    asset_id: str
    prompt: str
    obj_filename: str
    obj_path: Path
    engine_used: str
    device_used: str
    vertex_count: int
    face_count: int
    bounds_size: list[float]
    generation_time_sec: float


def _detect_neural_backend() -> str | None:
    """Return 'shap_e' or 'triposr' if installed with CUDA GPU, else None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        try:
            import shap_e  # noqa: F401
            return "shap_e"
        except ImportError:
            pass
        try:
            import tsr  # noqa: F401
            return "triposr"
        except ImportError:
            pass
    except Exception:
        pass
    return None


def _run_shape_e_obj(prompt: str, out_path: Path, guidance_scale: float = 15.0) -> trimesh.Trimesh:
    """Generate a raw OBJ mesh using OpenAI Shap-E on CUDA GPU."""
    import torch
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt]),
        progress=False,
        clip_denoised=True,
        use_fp16=(device.type == "cuda"),
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    tri_mesh_obj = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(out_path, "w", encoding="utf-8") as f:
        tri_mesh_obj.write_obj(f)
    return trimesh.load(out_path, force="mesh")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-Aware Procedural 3D Asset Builders (Fast CPU / Fallback Engine)
# ─────────────────────────────────────────────────────────────────────────────

def _build_broadsword(seed: int) -> trimesh.Trimesh:
    """Build a multi-part medieval broadsword with crossguard, grip, pommel & sapphire gem."""
    parts: list[trimesh.Trimesh] = []

    # 1. Diamond-section tapered blade
    blade_verts = np.array([
        [0.0,  2.15,  0.0],     # 0: tip
        [-0.14, 0.35,  0.0],    # 1: left shoulder
        [0.14,  0.35,  0.0],    # 2: right shoulder
        [0.0,   0.35,  0.035],  # 3: front spine base
        [0.0,   0.35, -0.035],  # 4: back spine base
        [-0.09, 1.55,  0.0],    # 5: left upper taper
        [0.09,  1.55,  0.0],    # 6: right upper taper
        [0.0,   1.55,  0.024],  # 7: front spine upper
        [0.0,   1.55, -0.024],  # 8: back spine upper
    ], dtype=float)
    blade_faces = np.array([
        [1, 3, 7], [1, 7, 5],
        [3, 2, 6], [3, 6, 7],
        [2, 4, 8], [2, 8, 6],
        [4, 1, 5], [4, 5, 8],
        [5, 7, 0], [7, 6, 0], [6, 8, 0], [8, 5, 0],
        [1, 4, 3], [2, 3, 4],
    ], dtype=int)
    blade = trimesh.Trimesh(vertices=blade_verts, faces=blade_faces, process=True)
    parts.append(blade)

    # 2. Crossguard bar + flared quillons
    guard = trimesh.creation.box(extents=[0.95, 0.11, 0.14])
    guard.apply_translation([0.0, 0.31, 0.0])
    parts.append(guard)

    for side in (-1.0, 1.0):
        cap = trimesh.creation.icosphere(subdivisions=1, radius=0.085)
        cap.apply_translation([side * 0.49, 0.31, 0.0])
        parts.append(cap)

    # 3. Central hilt housing + faceted sapphire jewel on both sides
    # Note: trimesh cylinders align along Z by default (pointing front/back through the guard)
    hilt_socket = trimesh.creation.cylinder(radius=0.11, height=0.16, sections=8)
    hilt_socket.apply_translation([0.0, 0.31, 0.0])
    parts.append(hilt_socket)

    for z_side in (-1.0, 1.0):
        gem = trimesh.creation.icosphere(subdivisions=1, radius=0.068)
        gem.apply_scale([1.0, 1.15, 0.55])
        gem.apply_translation([0.0, 0.31, z_side * 0.082])
        parts.append(gem)

    # 4. Ribbed grip handle (rotate Z-aligned cylinder onto Y axis)
    grip = trimesh.creation.cylinder(radius=0.055, height=0.62, sections=14)
    grip.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    grip.apply_translation([0.0, -0.02, 0.0])
    parts.append(grip)

    for y_ring in np.linspace(-0.22, 0.18, 5):
        ring = trimesh.creation.torus(major_radius=0.058, minor_radius=0.012, major_sections=16, minor_sections=8)
        ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        ring.apply_translation([0.0, float(y_ring), 0.0])
        parts.append(ring)

    # 5. Faceted pommel counterweight
    pommel = trimesh.creation.icosphere(subdivisions=1, radius=0.125)
    pommel.apply_scale([1.1, 0.9, 0.75])
    pommel.apply_translation([0.0, -0.37, 0.0])
    parts.append(pommel)

    return trimesh.util.concatenate(parts)


def _y_cylinder(radius: float, height: float, sections: int = 16) -> trimesh.Trimesh:
    """Create a cylinder aligned vertically along the Y axis."""
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return m


def _y_cone(radius: float, height: float, sections: int = 16) -> trimesh.Trimesh:
    """Create a cone pointing up along the +Y axis."""
    m = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return m


def _build_crystal_cluster(seed: int) -> trimesh.Trimesh:
    """Build a crystalline cluster on a stone base."""
    rng = np.random.default_rng(seed)
    parts: list[trimesh.Trimesh] = []

    base = _y_cylinder(radius=0.85, height=0.22, sections=9)
    base.apply_translation([0.0, -0.75, 0.0])
    parts.append(base)

    for i in range(9):
        radius = rng.uniform(0.11, 0.22)
        height = rng.uniform(0.75, 1.65) if i > 0 else 1.85
        shaft = _y_cylinder(radius=radius, height=height, sections=6)
        tip = _y_cone(radius=radius, height=radius * 1.8, sections=6)
        tip.apply_translation([0.0, height / 2.0, 0.0])
        crystal = trimesh.util.concatenate([shaft, tip])

        tilt_x = rng.uniform(-0.38, 0.38) if i > 0 else 0.04
        tilt_z = rng.uniform(-0.38, 0.38) if i > 0 else -0.04
        crystal.apply_transform(trimesh.transformations.rotation_matrix(tilt_x, [1, 0, 0]))
        crystal.apply_transform(trimesh.transformations.rotation_matrix(tilt_z, [0, 0, 1]))

        angle = (i / 8.0) * 2.0 * np.pi if i > 0 else 0.0
        dist = rng.uniform(0.20, 0.55) if i > 0 else 0.0
        crystal.apply_translation([np.cos(angle) * dist, -0.65 + height * 0.42, np.sin(angle) * dist])
        parts.append(crystal)

    return trimesh.util.concatenate(parts)


def _build_shield(seed: int) -> trimesh.Trimesh:
    """Build a domed round shield facing +Z with central boss and rim."""
    parts: list[trimesh.Trimesh] = []
    disc = trimesh.creation.cylinder(radius=1.05, height=0.09, sections=28)
    parts.append(disc)

    rim = trimesh.creation.torus(major_radius=1.02, minor_radius=0.065, major_sections=32, minor_sections=12)
    parts.append(rim)

    boss = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
    boss.apply_scale([1.0, 1.0, 0.65])
    boss.apply_translation([0.0, 0.0, 0.07])
    parts.append(boss)

    return trimesh.util.concatenate(parts)


def _build_potion_flask(seed: int) -> trimesh.Trimesh:
    """Build an alchemist potion bottle with collar and cork stopper."""
    parts: list[trimesh.Trimesh] = []
    body = trimesh.creation.icosphere(subdivisions=2, radius=0.68)
    body.apply_scale([1.0, 0.92, 1.0])
    body.apply_translation([0.0, -0.25, 0.0])
    parts.append(body)

    neck = _y_cylinder(radius=0.19, height=0.55, sections=16)
    neck.apply_translation([0.0, 0.50, 0.0])
    parts.append(neck)

    lip = trimesh.creation.torus(major_radius=0.22, minor_radius=0.05, major_sections=18, minor_sections=10)
    lip.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    lip.apply_translation([0.0, 0.78, 0.0])
    parts.append(lip)

    cork = _y_cylinder(radius=0.15, height=0.24, sections=12)
    cork.apply_translation([0.0, 0.92, 0.0])
    parts.append(cork)

    return trimesh.util.concatenate(parts)


def _build_treasure_chest(seed: int) -> trimesh.Trimesh:
    """Build a stylized chest with arched barrel lid and trim straps."""
    parts: list[trimesh.Trimesh] = []
    base = trimesh.creation.box(extents=[1.45, 0.72, 0.92])
    base.apply_translation([0.0, -0.25, 0.0])
    parts.append(base)

    lid = trimesh.creation.cylinder(radius=0.46, height=1.45, sections=18)
    lid.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    lid.apply_translation([0.0, 0.11, 0.0])
    parts.append(lid)

    for x_band in (-0.52, 0.0, 0.52):
        band = trimesh.creation.box(extents=[0.11, 0.78, 0.98])
        band.apply_translation([x_band, -0.24, 0.0])
        parts.append(band)

    lock = trimesh.creation.box(extents=[0.20, 0.24, 0.10])
    lock.apply_translation([0.0, 0.02, 0.49])
    parts.append(lock)

    return trimesh.util.concatenate(parts)


def _beam_between(p1: list[float], p2: list[float], radius: float, sections: int = 10) -> trimesh.Trimesh:
    """Create a cylinder connecting 3D points p1 and p2."""
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return trimesh.creation.icosphere(subdivisions=1, radius=radius)
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    direction = vec / length
    rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    cyl.apply_transform(rot)
    cyl.apply_translation((a + b) / 2.0)
    return cyl


def _build_motorcycle(seed: int) -> trimesh.Trimesh:
    """Build a 1980s/90s Yamaha RX-100 inspired retro 2-stroke motorcycle."""
    parts: list[trimesh.Trimesh] = []

    # 1. Front (+0.88) and Rear (-0.88) spoked wheels (lying in XY plane -> axis Z)
    for wx in (-0.88, 0.88):
        wy = -0.38
        tire = trimesh.creation.torus(major_radius=0.31, minor_radius=0.072, major_sections=32, minor_sections=12)
        tire.apply_translation([wx, wy, 0.0])
        parts.append(tire)

        hub = trimesh.creation.cylinder(radius=0.085, height=0.14, sections=14)
        hub.apply_translation([wx, wy, 0.0])
        parts.append(hub)

        for i in range(6):
            ang = (i / 6.0) * np.pi
            dx, dy = np.cos(ang) * 0.27, np.sin(ang) * 0.27
            spoke = _beam_between([wx - dx, wy - dy, 0.0], [wx + dx, wy + dy, 0.0], radius=0.011, sections=6)
            parts.append(spoke)

    # 2. Cradle frame, backbone & swingarm
    parts.append(_beam_between([0.56, 0.28, 0.0], [-0.14, 0.14, 0.0], radius=0.032))       # top backbone
    parts.append(_beam_between([0.54, 0.24, 0.0], [0.16, -0.30, 0.0], radius=0.026))       # downtube
    parts.append(_beam_between([0.16, -0.30, 0.0], [-0.14, -0.32, 0.0], radius=0.026))     # lower cradle
    parts.append(_beam_between([-0.14, 0.14, 0.0], [-0.14, -0.32, 0.0], radius=0.028))     # seat post
    parts.append(_beam_between([-0.14, 0.14, 0.0], [-0.88, 0.10, 0.0], radius=0.024))      # rear subframe

    for z_side in (-0.11, 0.11):
        parts.append(_beam_between([-0.12, -0.32, z_side], [-0.88, -0.38, z_side], radius=0.022))  # swingarm
        parts.append(_beam_between([-0.82, -0.34, z_side * 1.15], [-0.56, 0.13, z_side], radius=0.028))  # rear shocks

    # 3. 2-Stroke air-cooled engine block & cooling fins
    crankcase = trimesh.creation.cylinder(radius=0.14, height=0.28, sections=16)
    crankcase.apply_translation([0.02, -0.26, 0.0])
    parts.append(crankcase)

    cyl_block = _beam_between([0.04, -0.18, 0.0], [0.10, 0.05, 0.0], radius=0.095, sections=12)
    parts.append(cyl_block)

    for fy in np.linspace(-0.12, 0.04, 5):
        fin = trimesh.creation.box(extents=[0.24, 0.018, 0.22])
        fin.apply_translation([0.07, float(fy), 0.0])
        parts.append(fin)

    # 4. Signature RX-100 teardrop fuel tank & filler cap
    tank = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    tank.apply_scale([0.33, 0.14, 0.17])
    tank.apply_translation([0.23, 0.27, 0.0])
    parts.append(tank)

    cap = _y_cylinder(radius=0.045, height=0.04, sections=12)
    cap.apply_translation([0.29, 0.40, 0.0])
    parts.append(cap)

    # 5. Classic flat ribbed retro bench seat + rear grab rail
    seat = trimesh.creation.box(extents=[0.74, 0.12, 0.25])
    seat.apply_translation([-0.41, 0.19, 0.0])
    parts.append(seat)

    grab_rail = _beam_between([-0.74, 0.20, -0.11], [-0.86, 0.22, 0.0], radius=0.015)
    parts.append(grab_rail)
    grab_rail2 = _beam_between([-0.74, 0.20, 0.11], [-0.86, 0.22, 0.0], radius=0.015)
    parts.append(grab_rail2)

    # 6. Front telescopic forks, wide handlebars, round headlamp & twin gauges
    for z_side in (-0.085, 0.085):
        parts.append(_beam_between([0.56, 0.34, z_side], [0.88, -0.38, z_side], radius=0.024))

    parts.append(_beam_between([0.52, 0.42, -0.36], [0.52, 0.42, 0.36], radius=0.018))  # handlebar
    for z_grip in (-0.33, 0.33):
        grip = trimesh.creation.cylinder(radius=0.026, height=0.13, sections=12)
        grip.apply_translation([0.52, 0.42, z_grip])
        parts.append(grip)

    headlamp = _beam_between([0.62, 0.24, 0.0], [0.76, 0.24, 0.0], radius=0.105, sections=18)
    parts.append(headlamp)
    lens = trimesh.creation.icosphere(subdivisions=2, radius=0.095)
    lens.apply_scale([0.45, 1.0, 1.0])
    lens.apply_translation([0.76, 0.24, 0.0])
    parts.append(lens)

    # Twin RX-100 instrument dials
    for z_dial in (-0.055, 0.055):
        dial = _y_cylinder(radius=0.042, height=0.055, sections=14)
        dial.apply_translation([0.60, 0.38, z_dial])
        parts.append(dial)

    # 7. Fenders & Tail light
    f_fender = trimesh.creation.box(extents=[0.42, 0.04, 0.16])
    f_fender.apply_translation([0.86, -0.02, 0.0])
    parts.append(f_fender)

    r_fender = trimesh.creation.box(extents=[0.46, 0.04, 0.17])
    r_fender.apply_transform(trimesh.transformations.rotation_matrix(-0.28, [0, 0, 1]))
    r_fender.apply_translation([-0.92, 0.04, 0.0])
    parts.append(r_fender)

    tail_lamp = trimesh.creation.box(extents=[0.08, 0.07, 0.10])
    tail_lamp.apply_translation([-1.06, 0.10, 0.0])
    parts.append(tail_lamp)

    # 8. Signature long chrome RX-100 exhaust pipe (right side +Z)
    parts.append(_beam_between([0.12, -0.14, 0.08], [0.22, -0.38, 0.16], radius=0.024))
    parts.append(_beam_between([0.22, -0.38, 0.16], [-0.25, -0.40, 0.17], radius=0.028))
    parts.append(_beam_between([-0.25, -0.40, 0.17], [-0.96, -0.32, 0.18], radius=0.046, sections=16))

    return trimesh.util.concatenate(parts)


def _build_procedural_totem(seed: int) -> trimesh.Trimesh:
    """Build an architectural sci-fi / fantasy totem for general prompts."""
    rng = np.random.default_rng(seed)
    parts: list[trimesh.Trimesh] = []

    plinth = _y_cylinder(radius=0.82, height=0.20, sections=8)
    plinth.apply_translation([0.0, -0.82, 0.0])
    parts.append(plinth)

    core = trimesh.creation.icosphere(subdivisions=1, radius=0.58)
    core.apply_scale([1.0, rng.uniform(1.05, 1.35), 1.0])
    core.apply_translation([0.0, 0.05, 0.0])
    parts.append(core)

    halo = trimesh.creation.torus(major_radius=0.82, minor_radius=0.055, major_sections=24, minor_sections=10)
    halo.apply_transform(trimesh.transformations.rotation_matrix(0.55, [1, 0.4, 0]))
    halo.apply_translation([0.0, 0.10, 0.0])
    parts.append(halo)

    return trimesh.util.concatenate(parts)


_PROMPT_BUILDERS: list[tuple[tuple[str, ...], Callable[[int], trimesh.Trimesh]]] = [
    (("yamaha", "rx100", "rx 100", "rx-100", "motorcycle", "motorbike", "bike"), _build_motorcycle),
    (("sword", "broadsword", "blade", "katana", "dagger", "sabre", "hilt"), _build_broadsword),
    (("crystal", "gem", "sapphire", "quartz", "shard", "rock"), _build_crystal_cluster),
    (("shield", "buckler", "aegis"), _build_shield),
    (("potion", "flask", "bottle", "elixir", "jar"), _build_potion_flask),
    (("chest", "crate", "box", "treasure"), _build_treasure_chest),
]


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center the mesh at the origin and scale longest dimension to ~2.2 units."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation(-center)
    max_extent = float(np.max(mesh.extents))
    if max_extent > 0:
        mesh.apply_scale(2.2 / max_extent)
    return mesh


def generate_obj_from_prompt(
    prompt: str,
    output_dir: Path,
    engine_mode: str = "auto",
    guidance_scale: float = 15.0,
) -> MeshGenerationResult:
    """
    Generate a raw `.obj` mesh from a natural-language prompt.

    Selects neural Text-to-3D (Shap-E / TripoSR) when running on CUDA GPU
    with neural packages installed, or falls back to fast prompt-guided
    procedural 3D geometry synthesis via Trimesh.
    """
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_id = str(uuid.uuid4())[:8]
    slug = "".join(c if c.isalnum() else "_" for c in prompt.lower()).strip("_")[:28] or "mesh"
    obj_filename = f"polygen_{slug}_{asset_id}.obj"
    obj_path = output_dir / obj_filename

    neural_backend = _detect_neural_backend() if engine_mode in ("auto", "shap_e", "triposr") else None

    if neural_backend == "shap_e" and engine_mode in ("auto", "shap_e"):
        logger.info("Generating OBJ via OpenAI Shap-E (CUDA GPU)...")
        mesh = _run_shape_e_obj(prompt, obj_path, guidance_scale=guidance_scale)
        engine_used = "shap_e_cuda"
        device_used = "cuda"
    else:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        lower = prompt.lower()
        builder = _build_procedural_totem
        for keywords, fn in _PROMPT_BUILDERS:
            if any(kw in lower for kw in keywords):
                builder = fn
                break

        mesh = _normalize_mesh(builder(seed))
        mesh.export(str(obj_path), file_type="obj")
        engine_used = "procedural_trimesh_cpu"
        device_used = "cpu"

    elapsed = round(time.perf_counter() - t0, 3)
    extents = [round(float(v), 3) for v in mesh.extents]

    return MeshGenerationResult(
        asset_id=asset_id,
        prompt=prompt,
        obj_filename=obj_filename,
        obj_path=obj_path,
        engine_used=engine_used,
        device_used=device_used,
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        bounds_size=extents,
        generation_time_sec=elapsed,
    )
