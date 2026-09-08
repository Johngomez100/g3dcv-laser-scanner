"""Reduce duplicate point samples and save coloured point clouds as ASCII PLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: Path, points: np.ndarray, colours_bgr: np.ndarray) -> None:
    """Write an ASCII PLY file that MeshLab and Open3D can read."""
    colours_rgb = colours_bgr[:, ::-1].astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, colour in zip(points, colours_rgb):
            handle.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {colour[0]} {colour[1]} {colour[2]}\n")


def voxel_downsample(points: np.ndarray, colours: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or len(points) == 0:
        return points, colours
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return points[first_indices], colours[first_indices]
