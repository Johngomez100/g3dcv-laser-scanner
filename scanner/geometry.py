"""Camera rays, marker poses, and plane geometry. Coordinates use the marker units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Plane:
    """A plane represented by one point and a unit normal."""

    point: np.ndarray
    normal: np.ndarray


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / length


def fit_plane(points: np.ndarray) -> Plane:
    """Least-squares plane fit using the smallest-variance SVD direction."""
    if len(points) < 3:
        raise ValueError("At least three points are needed to fit a plane")
    centre = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    return Plane(centre, normalize(vh[-1]))


def intersect_rays_plane(directions: np.ndarray, plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    """Intersect camera-centred rays with a plane.

    Returns the 3-D points and a validity mask.  A point is valid only when its
    ray is not parallel to the plane and the intersection lies in front of the
    camera.
    """
    denominator = directions @ plane.normal
    numerator = float(plane.point @ plane.normal)
    valid = np.abs(denominator) > EPS
    distance = np.zeros(len(directions), dtype=np.float64)
    distance[valid] = numerator / denominator[valid]
    valid &= distance > 0.0
    return directions * distance[:, None], valid


def estimate_marker_planes(
    marker_corners: Iterable[np.ndarray], camera_matrix: np.ndarray, marker_width: float, marker_height: float
) -> list[Plane]:
    """Estimate each marker plane with solvePnP from its known physical dimensions."""
    object_corners = np.array(
        [[0.0, 0.0, 0.0], [marker_width, 0.0, 0.0], [marker_width, marker_height, 0.0], [0.0, marker_height, 0.0]],
        dtype=np.float64,
    )
    planes: list[Plane] = []
    for corners in marker_corners:
        success, rotation_vector, translation = cv2.solvePnP(
            object_corners, corners.astype(np.float64), camera_matrix, np.zeros((5, 1)), flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            continue
        rotation, _ = cv2.Rodrigues(rotation_vector)
        planes.append(Plane(translation.reshape(3), normalize(rotation[:, 2])))
    return planes


def backproject(pixels: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    directions = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def p1_reference(corners: np.ndarray, camera_matrix: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space origin and basis: P1 top-left, X right, Y down, Z outward.

    This intentionally left-handed convention follows the user's diagram.
    For row-vector camera points, coordinates are (points - origin) @ basis.
    """
    model = np.array([[0., 0., 0.], [width, 0., 0.],
                      [width, height, 0.], [0., height, 0.]])
    success, rotation_vector, translation = cv2.solvePnP(
        model, np.asarray(corners, dtype=np.float64), camera_matrix,
        np.zeros((5, 1)), flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise RuntimeError("Could not estimate the P1 reference frame")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return translation.reshape(3), rotation @ np.diag([1., 1., -1.])
