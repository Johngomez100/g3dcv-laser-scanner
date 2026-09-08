"""Process the video: detect, reconstruct, accumulate, and save the results."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import calibrate_markers

from .detection import marker_candidates, refine_marker_corners, laser_pixels, points_inside_polygon, sample_laser_free_colours
from .geometry import EPS, estimate_marker_planes, backproject, intersect_rays_plane, fit_plane, p1_reference
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
    fixed_corners = None
    fixed_planes = None
    calibration_count = 0
    reference = None
    reference_frame = getattr(args, "reference_frame", "camera")
    if args.marker_mode == "fixed":
        fixed_corners, calibration_count = calibrate_markers(
            args.video, camera_matrix, distortion, args.min_marker_area,
            args.calibration_frames, not args.no_refine_markers,
        )
        fixed_planes = estimate_marker_planes(fixed_corners, camera_matrix, args.marker_width, args.marker_height)
        if len(fixed_planes) != 2:
            raise RuntimeError("Could not estimate both marker planes")
    marker_red = args.min_red if args.marker_min_red is None else args.marker_min_red
    marker_excess = args.min_red_excess if args.marker_min_red_excess is None else args.marker_min_red_excess
    object_red = args.min_red if args.object_min_red is None else args.object_min_red
    object_excess = args.min_red_excess if args.object_min_red_excess is None else args.object_min_red_excess
    plane_errors = []
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_video:
        args.debug_video.parent.mkdir(parents=True, exist_ok=True)
    debug_writer: cv2.VideoWriter | None = None
    points_accumulated: list[np.ndarray] = []
    colours_accumulated: list[np.ndarray] = []
    stats = ScanStats()
    cached_corners: list[np.ndarray] | None = None

    try:
        while True:
            ok, raw_frame = capture.read()
            if not ok:
                break
            stats.frames_read += 1
            if (stats.frames_read - 1) % args.frame_stride:
                continue
            stats.frames_processed += 1
            frame = cv2.undistort(raw_frame, camera_matrix, distortion)
            corners = fixed_corners if fixed_corners is not None else marker_candidates(frame, args.min_marker_area)
            if fixed_corners is None and len(corners) == 2 and not args.no_refine_markers:
                corners = refine_marker_corners(frame, corners)
            if len(corners) == 2:
                cached_corners = corners
            elif cached_corners is not None:
                corners = cached_corners
            else:
                continue

            marker_planes = fixed_planes if fixed_planes is not None else estimate_marker_planes(corners, camera_matrix, args.marker_width, args.marker_height)
            if len(marker_planes) != 2:
                continue
            if reference_frame == "p1" and reference is None:
                # Establish one fixed world frame, even with dynamic detection.
                # In the supplied scene P1 is the upper (wall) marker.
                p1 = min(corners, key=lambda polygon: float(np.mean(polygon[:, 1])))
                reference = p1_reference(p1, camera_matrix, args.marker_width, args.marker_height)
            stats.frames_with_markers += 1
            if (marker_red, marker_excess) == (object_red, object_excess):
                pixels = laser_pixels(frame, marker_excess, marker_red, args.min_laser_component_pixels, args.max_laser_row_jump)
            else:
                # Mask before component filtering so faint object settings cannot alter
                # the marker samples used to estimate the laser plane.
                marker_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for polygon in corners:
                    cv2.fillConvexPoly(marker_mask, np.rint(polygon).astype(np.int32), 1)
                marker_pixels = laser_pixels(
                    frame, marker_excess, marker_red, args.min_laser_component_pixels,
                    args.max_laser_row_jump, region_mask=marker_mask,
                )
                object_candidates = laser_pixels(
                    frame, object_excess, object_red, args.min_laser_component_pixels,
                    args.max_laser_row_jump, region_mask=1 - marker_mask,
                )
                pixels = np.vstack((marker_pixels, object_candidates))
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

            calibration_array = np.vstack(calibration_points)
            laser_plane = fit_plane(calibration_array)
            plane_errors.append(float(np.sqrt(np.mean(((calibration_array - laser_plane.point) @ laser_plane.normal) ** 2))))
            # Brief section 1, step 2(d): intersect ALL laser rays, including
            # the marker rays, with the fitted laser plane.
            object_points, valid = intersect_rays_plane(directions, laser_plane)
            object_points = object_points[valid]
            object_pixels = pixels[valid]
            if len(object_points) == 0:
                continue
            colours = sample_laser_free_colours(
                frame, object_pixels, object_excess, object_red, args.colour_radius
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
                    if not debug_writer.isOpened():
                        raise RuntimeError(f"Cannot write debug video: {args.debug_video}")
                debug_writer.write(annotated)
    finally:
        capture.release()
        if debug_writer is not None:
            debug_writer.release()
    if not points_accumulated:
        raise RuntimeError("No 3-D points were reconstructed. Check marker size and laser thresholds.")
    points = np.vstack(points_accumulated)
    colours = np.vstack(colours_accumulated)
    if reference is not None:
        origin, basis = reference
        points = (points - origin) @ basis
    points, colours = voxel_downsample(points, colours, args.voxel_size)
    write_ply(args.output, points, colours)
    stats.elapsed_processing_seconds = time.perf_counter() - start_time
    report = {
        **stats.__dict__, "points_written": int(len(points)),
        "marker_mode": args.marker_mode,
        "reference_frame": reference_frame,
        "coordinate_units": "metres",
        "marker_dimensions_metres": [args.marker_width, args.marker_height],
        "voxel_size": args.voxel_size,
        "reference_origin_camera": None if reference is None else reference[0].tolist(),
        "reference_basis_camera": None if reference is None else reference[1].tolist(),
        "calibration_frames_used": calibration_count,
        "marker_corners": None if fixed_corners is None else np.asarray(fixed_corners).tolist(),
        "marker_thresholds": {"min_red": marker_red, "min_red_excess": marker_excess},
        "object_thresholds": {"min_red": object_red, "min_red_excess": object_excess},
        "laser_plane_rms_median": float(np.median(plane_errors)),
        "laser_plane_rms_p95": float(np.percentile(plane_errors, 95)),
        "processed_frames_per_second": stats.frames_processed / max(stats.elapsed_processing_seconds, EPS),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return stats
