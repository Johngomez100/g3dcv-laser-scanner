#!/usr/bin/env python3
"""Reconstruct a coloured point cloud from a laser-line scanning video.

The program follows the final-project geometry directly:
  1. detect the two printed planar markers and estimate their poses;
  2. back-project laser pixels that fall on either marker;
  3. fit the laser plane through those 3-D points;
  4. intersect every other laser ray with the fitted laser plane.

All coordinates are expressed in the camera coordinate system and in the same
metric unit as --marker-width/--marker-height (metres by default).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class Plane:
    """A plane represented by one point and a unit normal."""

    point: np.ndarray
    normal: np.ndarray


@dataclass
class ScanStats:
    frames_read: int = 0
    frames_with_markers: int = 0
    frames_reconstructed: int = 0
    points_before_downsampling: int = 0


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


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).ravel()  # y - x
    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(differences)]
    ordered[3] = corners[np.argmax(differences)]
    return ordered


def marker_candidates(frame: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Find bright convex quadrilaterals, keeping the two largest likely markers.

    The provided target is a white printed rectangle.  Otsu thresholding makes
    this resilient to a moderate change in room illumination; the optional
    adaptive threshold catches the opposite contrast convention.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 4
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for binary in (binary_otsu, binary_adaptive):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            # Thresholding can turn the full image background into one very
            # large quadrilateral. It is not a physical marker.
            if x <= 1 or y <= 1 or x + width >= frame.shape[1] - 1 or y + height >= frame.shape[0] - 1:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            candidates.append((area, order_corners(polygon.reshape(4, 2))))

    # RETR_LIST and the two threshold methods can find the same sheet repeatedly.
    selected: list[np.ndarray] = []
    for _, corners in sorted(candidates, key=lambda item: item[0], reverse=True):
        centre = np.mean(corners, axis=0)
        if all(np.linalg.norm(centre - np.mean(other, axis=0)) > 20 for other in selected):
            selected.append(corners)
        if len(selected) == 2:
            break
    return selected


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


def laser_pixels(frame: np.ndarray, min_red_excess: int, min_red: int) -> np.ndarray:
    """Extract one sub-pixel-free centre sample per image row from the red laser line."""
    blue, green, red = cv2.split(frame.astype(np.int16))
    red_excess = red - np.maximum(blue, green)
    mask = (red > min_red) & (red_excess > min_red_excess)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    pixels: list[tuple[float, float]] = []
    for y, row in enumerate(mask):
        xs = np.flatnonzero(row)
        if len(xs) == 0:
            continue
        # A red score-weighted centroid avoids emitting every pixel in a thick line.
        weights = red_excess[y, xs].astype(np.float64)
        x = float(np.average(xs, weights=np.maximum(weights, 1.0)))
        pixels.append((x, float(y)))
    return np.asarray(pixels, dtype=np.float64)


def backproject(pixels: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    directions = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def points_inside_polygon(pixels: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    return np.array([cv2.pointPolygonTest(polygon.astype(np.float32), tuple(pixel), False) >= 0 for pixel in pixels])


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


def process_video(args: argparse.Namespace) -> ScanStats:
    camera_matrix = np.loadtxt(args.intrinsics, dtype=np.float64).reshape(3, 3)
    distortion = np.loadtxt(args.distortion, dtype=np.float64).reshape(-1, 1)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    debug_writer: cv2.VideoWriter | None = None
    points_accumulated: list[np.ndarray] = []
    colours_accumulated: list[np.ndarray] = []
    stats = ScanStats()
    cached_corners: list[np.ndarray] | None = None

    while True:
        ok, raw_frame = capture.read()
        if not ok:
            break
        stats.frames_read += 1
        if (stats.frames_read - 1) % args.frame_stride:
            continue
        frame = cv2.undistort(raw_frame, camera_matrix, distortion)
        corners = marker_candidates(frame, args.min_marker_area)
        if len(corners) == 2:
            cached_corners = corners
        elif cached_corners is not None:
            corners = cached_corners
        else:
            continue

        marker_planes = estimate_marker_planes(corners, camera_matrix, args.marker_width, args.marker_height)
        if len(marker_planes) != 2:
            continue
        stats.frames_with_markers += 1
        pixels = laser_pixels(frame, args.min_red_excess, args.min_red)
        if len(pixels) == 0:
            continue
        directions = backproject(pixels, camera_matrix)
        on_marker = [points_inside_polygon(pixels, polygon) for polygon in corners]
        calibration_points: list[np.ndarray] = []
        for membership, marker_plane in zip(on_marker, marker_planes):
            intersections, valid = intersect_rays_plane(directions[membership], marker_plane)
            calibration_points.append(intersections[valid])
        if any(len(group) < args.min_marker_samples for group in calibration_points):
            continue

        laser_plane = fit_plane(np.vstack(calibration_points))
        object_membership = ~(on_marker[0] | on_marker[1])
        object_points, valid = intersect_rays_plane(directions[object_membership], laser_plane)
        object_pixels = pixels[object_membership][valid]
        if len(object_points) == 0:
            continue
        pixel_indices = np.rint(object_pixels).astype(np.int32)
        pixel_indices[:, 0] = np.clip(pixel_indices[:, 0], 0, frame.shape[1] - 1)
        pixel_indices[:, 1] = np.clip(pixel_indices[:, 1], 0, frame.shape[0] - 1)
        colours = frame[pixel_indices[:, 1], pixel_indices[:, 0]]
        points_accumulated.append(object_points)
        colours_accumulated.append(colours)
        stats.frames_reconstructed += 1
        stats.points_before_downsampling += len(object_points)

        if args.debug_video:
            annotated = frame.copy()
            for polygon in corners:
                cv2.polylines(annotated, [polygon.astype(np.int32)], True, (0, 255, 0), 2)
            for x, y in pixels.astype(np.int32):
                cv2.circle(annotated, (x, y), 1, (255, 255, 0), -1)
            if debug_writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS) / args.frame_stride
                debug_writer = cv2.VideoWriter(str(args.debug_video), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (frame.shape[1], frame.shape[0]))
            debug_writer.write(annotated)

    capture.release()
    if debug_writer is not None:
        debug_writer.release()
    if not points_accumulated:
        raise RuntimeError("No 3-D points were reconstructed. Check marker size and laser thresholds.")
    points = np.vstack(points_accumulated)
    colours = np.vstack(colours_accumulated)
    points, colours = voxel_downsample(points, colours, args.voxel_size)
    write_ply(args.output, points, colours)
    args.output.with_suffix(".json").write_text(json.dumps({**stats.__dict__, "points_written": int(len(points))}, indent=2), encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laser-line 3-D scanner for the G3DCV final project")
    parser.add_argument("--video", type=Path, required=True, help="Input cup1.mp4/cup2.mp4/puppet.mp4/soap.mp4")
    parser.add_argument("--intrinsics", type=Path, required=True, help="Path to K.txt")
    parser.add_argument("--distortion", type=Path, required=True, help="Path to dist.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output .ply file")
    parser.add_argument("--marker-width", type=float, required=True, help="Printed marker width, in metres")
    parser.add_argument("--marker-height", type=float, required=True, help="Printed marker height, in metres")
    parser.add_argument("--min-marker-area", type=float, default=2_000, help="Minimum marker area in pixels")
    parser.add_argument("--min-red", type=int, default=120, help="Minimum R intensity for laser extraction")
    parser.add_argument("--min-red-excess", type=int, default=50, help="Minimum R - max(G,B) laser score")
    parser.add_argument("--min-marker-samples", type=int, default=6, help="Laser samples required on each marker")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--voxel-size", type=float, default=0.002, help="Voxel size in marker units; 0 disables downsampling")
    parser.add_argument("--debug-video", type=Path, help="Optional annotated MP4 output")
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    result = process_video(cli_args)
    print(f"Reconstructed {result.points_before_downsampling} samples from {result.frames_reconstructed} frames.")
