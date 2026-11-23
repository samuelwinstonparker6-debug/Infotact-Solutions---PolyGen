"""
PolyGen Week 2 — Mesh Processing: Trimesh Automatic UV-Unwrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone: Week 2 Mesh Processing
- Write a Python script using trimesh to automatically UV-unwrap
  the generated .obj file so textures can be applied without seams.
- Injects 2D texture coordinates (vt) and surface normals (vn) into
  the Wavefront OBJ geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import trimesh

logger = logging.getLogger("polygen.uv")


class MeshUVUnwrapper:
    """
    Automates UV-unwrapping and texture coordinate generation for 3D meshes.
    Supports spherical, cylindrical, and box/planar parameterization projections.
    """

    @classmethod
    def unwrap_obj_file(
        cls,
        input_obj_path: Path,
        output_obj_path: Optional[Path] = None,
        projection: str = "box",
    ) -> Tuple[Path, int]:
        """
        Loads a raw .obj file, computes automatic UV texture coordinates,
        and saves an unwrapped .obj ready for texture mapping.

        Returns:
            - output_path: Path to the UV-unwrapped .obj file
            - uv_count: Number of generated UV coordinates
        """
        target_path = output_obj_path or input_obj_path.with_name(
            f"{input_obj_path.stem}_unwrapped.obj"
        )

        mesh = trimesh.load(str(input_obj_path), file_type="obj", force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object from {input_obj_path} is not a valid Trimesh.")

        # Compute vertex UVs based on projection geometry
        uv_coords = cls.generate_uv_coordinates(mesh, projection=projection)

        # Attach texture visual
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords)

        # Export OBJ with texture coordinates (vt lines)
        obj_content = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_texture=True)
        target_path.write_text(obj_content, encoding="utf-8")

        logger.info(
            "UV-unwrapped %s (%d vertices, %d UV coords) -> %s",
            input_obj_path.name,
            len(mesh.vertices),
            len(uv_coords),
            target_path.name,
        )
        return target_path, len(uv_coords)

    @classmethod
    def generate_uv_coordinates(
        cls,
        mesh: trimesh.Trimesh,
        projection: str = "box",
    ) -> np.ndarray:
        """
        Generates normalized (u, v) in [0.0, 1.0] for every vertex.
        """
        vertices = mesh.vertices - mesh.bounds[0]
        extents = np.maximum(mesh.extents, 1e-6)
        norm_v = vertices / extents

        if projection == "spherical":
            # Spherical coordinate projection
            centered = mesh.vertices - mesh.centroid
            r = np.maximum(np.linalg.norm(centered, axis=1), 1e-6)
            theta = np.arctan2(centered[:, 1], centered[:, 0])  # [-pi, pi]
            phi = np.arccos(np.clip(centered[:, 2] / r, -1.0, 1.0))  # [0, pi]
            u = (theta + np.pi) / (2.0 * np.pi)
            v = phi / np.pi
            return np.column_stack((u, v)).astype(np.float32)

        elif projection == "cylindrical":
            # Cylindrical coordinate projection along vertical Y axis
            centered = mesh.vertices - mesh.centroid
            theta = np.arctan2(centered[:, 2], centered[:, 0])
            u = (theta + np.pi) / (2.0 * np.pi)
            v = norm_v[:, 1]
            return np.column_stack((u, v)).astype(np.float32)

        else:
            # Planar / Box projection (default robust unwrapping for arbitrary game assets)
            u = norm_v[:, 0] * 0.5 + norm_v[:, 2] * 0.5
            v = norm_v[:, 1]
            return np.column_stack((np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0))).astype(np.float32)
