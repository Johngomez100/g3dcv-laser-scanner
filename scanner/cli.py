"""Command-line options and the scanner entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laser-line 3-D scanner for the G3DCV final project")
    parser.add_argument("--video", type=Path, required=True, help="Input cup1.mp4/cup2.mp4/puppet.mp4/soap.mp4")
    parser.add_argument("--intrinsics", type=Path, required=True, help="Path to K.txt")
    parser.add_argument("--distortion", type=Path, required=True, help="Path to dist.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output .ply file")
    parser.add_argument("--marker-width", type=float, required=True, help="Printed marker width, in metres")
    parser.add_argument("--marker-height", type=float, required=True, help="Printed marker height, in metres")
    parser.add_argument("--reference-frame", choices=("camera", "p1"), default="p1", help="Export in P1 coordinates: origin top-left, X right, Y down, Z outward. P1 is the upper marker in the image.")
    parser.add_argument("--min-marker-area", type=float, default=2_000, help="Minimum marker area in pixels")
    parser.add_argument("--min-red", type=int, default=60, help="Minimum R intensity for laser extraction")
    parser.add_argument("--min-red-excess", type=int, default=10, help="Minimum R - max(G,B) laser score")
    parser.add_argument("--marker-min-red", type=int, help="Marker R threshold; defaults to --min-red")
    parser.add_argument("--marker-min-red-excess", type=int, help="Marker red-excess threshold; defaults to --min-red-excess")
    parser.add_argument("--object-min-red", type=int, help="Object R threshold; defaults to --min-red")
    parser.add_argument("--object-min-red-excess", type=int, help="Object red-excess threshold; defaults to --min-red-excess")
    parser.add_argument("--marker-mode", choices=("fixed", "dynamic"), default="fixed", help="Fixed calibration for a stationary camera and markers; dynamic re-detects each frame")
    parser.add_argument("--calibration-frames", type=int, default=10, help="Initial frames used for fixed marker calibration")
    parser.add_argument("--no-refine-markers", action="store_true", help="Disable subpixel corner refinement for comparison")
    parser.add_argument("--min-laser-component-pixels", type=int, default=12, help="Reject smaller isolated laser-mask blobs")
    parser.add_argument("--max-laser-row-jump", type=float, default=25.0, help="Largest consistent laser x shift between nearby rows")
    parser.add_argument("--colour-radius", type=int, default=2, help="Radius for laser-free median surface colour; 0 keeps raw laser pixels")
    parser.add_argument("--min-marker-samples", type=int, default=6, help="Laser samples required on each marker")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel size in metres; default 0 retains all reconstructed samples")
    parser.add_argument("--debug-video", type=Path, help="Optional annotated MP4 output")
    return parser


def main() -> None:
    """Read command-line options and run one video reconstruction."""
    cli_args = build_parser().parse_args()
    if cli_args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if cli_args.calibration_frames < 1:
        raise SystemExit("--calibration-frames must be at least 1")
    if cli_args.marker_width <= 0 or cli_args.marker_height <= 0:
        raise SystemExit("Marker dimensions must be positive")
    result = process_video(cli_args)
    print(
        f"Reconstructed {result.points_before_downsampling} samples from {result.frames_reconstructed} frames "
        f"in {result.elapsed_processing_seconds:.2f}s."
    )
