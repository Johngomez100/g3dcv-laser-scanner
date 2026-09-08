#!/usr/bin/env python3
"""Run the course laser scanner with the existing command-line interface.

Implementation lives in scanner/. Re-export the original helpers so existing
notebooks and tests importing laser_scanner continue to work.
"""

from scanner.cli import build_parser, main
from scanner.detection import (
    laser_pixels,
    marker_candidates,
    order_corners,
    points_inside_polygon,
    red_laser_mask,
    sample_laser_free_colours,
)
from scanner.geometry import (
    EPS,
    Plane,
    backproject,
    estimate_marker_planes,
    fit_plane,
    intersect_rays_plane,
    normalize,
)
from scanner.pipeline import ScanStats, process_video
from scanner.point_cloud import voxel_downsample, write_ply


if __name__ == "__main__":
    main()
