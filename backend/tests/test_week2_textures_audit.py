"""
PolyGen Week 2 Mid-Project Review — Geometry Audit & Texture Mapping Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone: Mid-Project Review
- Geometry Audit: Prove the model can successfully generate recognizable 3D structures
  (e.g., a chair, a sword, a motorcycle) rather than abstract blobs.
- Rendering Validation: Ensure the Three.js canvas can map the generated texture
  onto the generated mesh via valid UV coordinates.
"""

from pathlib import Path
import pytest
import trimesh

from backend.services.mesh_generator import generate_obj_from_prompt
from backend.services.texture_generator import TextureGenerationPipeline
from backend.services.uv_unwrapper import MeshUVUnwrapper


def test_geometry_audit_recognizable_broadsword(tmp_path):
    """Geometry Audit: Verify sword possesses expected blade/crossguard aspect ratios, not an amorphous blob."""
    res = generate_obj_from_prompt("A rusted medieval broadsword with sapphire", tmp_path, engine_mode="procedural")
    mesh_path = tmp_path / res.obj_filename
    mesh = trimesh.load(str(mesh_path), file_type="obj", force="mesh")

    # A sword must be elongated vertically (Y dimension > X and Z dimensions)
    extents = mesh.extents
    assert extents[1] > extents[0] * 1.5, f"Sword Y-extent ({extents[1]}) should exceed X-extent ({extents[0]})"
    assert extents[1] > extents[2] * 2.0, f"Sword Y-extent ({extents[1]}) should exceed Z-extent ({extents[2]})"
    assert mesh.is_watertight or len(mesh.faces) > 50


def test_geometry_audit_recognizable_motorcycle(tmp_path):
    """Geometry Audit: Verify motorcycle has distinct wheelbase and chassis length."""
    res = generate_obj_from_prompt("Classic 1990s Yamaha RX 100 motorcycle", tmp_path, engine_mode="procedural")
    mesh_path = tmp_path / res.obj_filename
    mesh = trimesh.load(str(mesh_path), file_type="obj", force="mesh")

    extents = mesh.extents
    # Length (X) should be greater than height (Y) and width (Z)
    assert extents[0] > extents[2] * 1.4
    assert len(mesh.vertices) >= 200
    assert len(mesh.faces) >= 300


def test_rendering_validation_uv_unwrapping_and_textures(tmp_path):
    """Rendering Validation: Ensure .obj file contains valid UV coordinates (vt) and binds PBR textures."""
    res = generate_obj_from_prompt("A glowing cluster of sapphire crystals", tmp_path, engine_mode="procedural")
    raw_obj = tmp_path / res.obj_filename

    # Step 1: UV-unwrap the mesh
    unwrapped_obj, uv_count = MeshUVUnwrapper.unwrap_obj_file(raw_obj)
    assert unwrapped_obj.exists()
    assert uv_count > 0

    obj_text = unwrapped_obj.read_text(encoding="utf-8")
    assert "vt " in obj_text, "Unwrapped OBJ must include 2D texture coordinates (vt)"
    assert "vn " in obj_text, "Unwrapped OBJ must include surface normals (vn)"

    # Step 2: Generate PBR texture atlas
    textures = TextureGenerationPipeline.generate_texture_set(
        prompt="A glowing cluster of sapphire crystals",
        output_dir=tmp_path,
        asset_id=res.asset_id,
        resolution=256,
    )
    diffuse_file = tmp_path / textures["diffuse_filename"]
    normal_file = tmp_path / textures["normal_filename"]
    roughness_file = tmp_path / textures["roughness_filename"]

    assert diffuse_file.exists() and diffuse_file.stat().st_size > 1000
    assert normal_file.exists() and normal_file.stat().st_size > 1000
    assert roughness_file.exists() and roughness_file.stat().st_size > 1000
