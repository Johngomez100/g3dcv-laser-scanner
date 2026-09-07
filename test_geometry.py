"""Small deterministic checks for the project geometry (no video data required)."""

import cv2
import numpy as np

from laser_scanner import Plane, fit_plane, intersect_rays_plane, laser_pixels, marker_candidates


def test_fit_plane_and_intersection() -> None:
    # z=2 plane, sampled at four non-collinear positions.
    plane = fit_plane(np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]))
    assert np.isclose(abs(plane.normal[2]), 1.0)
    directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    intersections, valid = intersect_rays_plane(directions, plane)
    assert valid.all()
    assert np.allclose(intersections[:, 2], 2.0)


def test_parallel_ray_is_rejected() -> None:
    plane = Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]))
    _, valid = intersect_rays_plane(np.array([[1.0, 0.0, 0.0]]), plane)
    assert not valid[0]


def test_image_primitives_on_clean_synthetic_frame() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (70, 100), (240, 280), (255, 255, 255), -1)
    cv2.rectangle(frame, (390, 60), (560, 230), (255, 255, 255), -1)
    cv2.line(frame, (20, 180), (620, 180), (0, 0, 255), 5)
    markers = marker_candidates(frame, min_area=2_000)
    samples = laser_pixels(frame, min_red_excess=50, min_red=120)
    assert len(markers) == 2
    assert len(samples) > 0
