"""Process the video: detect, reconstruct, accumulate, and save the results."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .detection import marker_candidates, laser_pixels, points_inside_polygon, sample_laser_free_colours
from .geometry import EPS, estimate_marker_planes, backproject, intersect_rays_plane, fit_plane
from .point_cloud import voxel_downsample, write_ply


@dataclass
class ScanStats:
    frames_read: int = 0
    frames_processed: int = 0
    frames_with_markers: int = 0
    frames_reconstructed: int = 0
    points_before_downsampling: int = 0
    elapsed_processing_seconds: float = 0.0


def process_video(args: argparse.Namespace) -> ScanStats:
    start_time = time.perf_counter()
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
        stats.frames_processed += 1
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
        pixels = laser_pixels(
            frame, args.min_red_excess, args.min_red, args.min_laser_component_pixels, args.max_laser_row_jump
        )
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
        colours = sample_laser_free_colours(
            frame, object_pixels, args.min_red_excess, args.min_red, args.colour_radius
        )
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
    stats.elapsed_processing_seconds = time.perf_counter() - start_time
    report = {
        **stats.__dict__, "points_written": int(len(points)),
        "processed_frames_per_second": stats.frames_processed / max(stats.elapsed_processing_seconds, EPS),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return stats
