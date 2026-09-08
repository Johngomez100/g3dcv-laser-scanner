"""Estimate fixed marker corners from several frames of a stationary scene."""

import cv2
import numpy as np

from .detection import marker_candidates, refine_marker_corners


def calibrate_markers(video, camera_matrix, distortion, min_area, frame_count, refine=True):
    """Use the median of consistent detections to suppress corner jitter.

    Assumes that the camera and both markers remain still throughout the scan.
    Sort markers vertically so a change in contour area cannot swap identities.
    """
    capture = cv2.VideoCapture(str(video))
    observations = []
    reference = None
    try:
        for _ in range(frame_count):
            ok, raw = capture.read()
            if not ok:
                break
            frame = cv2.undistort(raw, camera_matrix, distortion)
            corners = marker_candidates(frame, min_area)
            if len(corners) != 2:
                continue
            corners = sorted(corners, key=lambda polygon: polygon[:, 1].mean())
            if refine:
                corners = refine_marker_corners(frame, corners)
            candidate = np.asarray(corners)
            if reference is None:
                reference = candidate
            if np.max(np.linalg.norm(candidate - reference, axis=2)) <= 5.0:
                observations.append(candidate)
    finally:
        capture.release()
    if not observations:
        raise RuntimeError("No marker pair found during calibration. Check the first frames or increase --calibration-frames.")
    corners = np.median(observations, axis=0).astype(np.float32)
    return list(corners), len(observations)
