import pytest
from pathlib import Path
from backend.services.mesh_generator import generate_obj_from_prompt

def test_mesh_bounding_box_and_faces(tmp_path):
    result = generate_obj_from_prompt("A medieval shield with iron boss", tmp_path, engine_mode="procedural")
    assert result.vertex_count >= 8
    assert result.face_count >= 6
    assert len(result.bounds_size) == 3
    assert all(dim > 0 for dim in result.bounds_size)

def test_mesh_obj_file_contents(tmp_path):
    result = generate_obj_from_prompt("A glass potion flask", tmp_path, engine_mode="procedural")
    obj_file = tmp_path / result.obj_filename
    assert obj_file.exists()
    content = obj_file.read_text(encoding="utf-8")
    assert "v " in content
    assert "f " in content
