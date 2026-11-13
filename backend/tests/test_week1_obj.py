"""
PolyGen Week 1 — Unit & API Verification Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verifies:
1. /api/health reports compute device & Week 1 milestone metadata.
2. /api/generate creates a valid Wavefront .obj mesh on disk from a text prompt.
3. Generated .obj loads cleanly back into Trimesh with valid vertices & faces.
4. /outputs/{filename} serves the generated .obj over HTTP.
"""

from fastapi.testclient import TestClient
import trimesh

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["device"] in ("cuda", "cpu")


def test_generate_broadsword_obj_and_fetch():
    prompt = "A rusted, medieval broadsword with a glowing blue sapphire in the hilt"
    res = client.post("/api/generate", json={"prompt": prompt})
    assert res.status_code == 200
    data = res.json()

    assert data["obj_filename"].endswith(".obj")
    assert data["vertex_count"] > 20
    assert data["face_count"] > 20

    # Verify static serving returns real Wavefront OBJ content
    obj_res = client.get(data["model_url"])
    assert obj_res.status_code == 200
    assert "v " in obj_res.text and "f " in obj_res.text

    # Verify trimesh parses the downloaded OBJ cleanly
    import io
    loaded = trimesh.load(io.BytesIO(obj_res.text.encode("utf-8")), file_type="obj", force="mesh")
    assert len(loaded.vertices) == data["vertex_count"]
    assert len(loaded.faces) == data["face_count"]


def test_generate_yamaha_rx100_obj():
    res = client.post("/api/generate", json={"prompt": "Classic 1990s Yamaha RX 100 motorcycle"})
    assert res.status_code == 200
    data = res.json()
    assert data["obj_filename"].endswith(".obj")
    assert data["vertex_count"] > 500
    assert data["face_count"] > 500

