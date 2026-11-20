"""
PolyGen Week 4 Pipeline Scaffolding: GLTF / GLB Export Service
Converts generated OBJ meshes and texture atlases into binary GLB files.
"""

import logging
from pathlib import Path
from typing import Optional
import trimesh

logger = logging.getLogger("polygen.export")

def convert_obj_to_glb(obj_path: Path, output_glb_path: Optional[Path] = None) -> Path:
    target_path = output_glb_path or obj_path.with_suffix(".glb")
    mesh = trimesh.load(str(obj_path), file_type="obj", force="mesh")
    mesh.export(str(target_path), file_type="glb")
    logger.info("Successfully converted %s to GLB: %s", obj_path.name, target_path.name)
    return target_path
