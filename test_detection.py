"""Regression checks for thin stripes, separate surfaces, and calibration."""

import cv2
import numpy as np

from scanner.detection import laser_pixels, refine_marker_corners
from scanner.calibration import calibrate_markers


def test_separate_stripes_do_not_create_a_ray_in_the_gap():
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[5:45, 20, 2] = 180
    frame[5:45, 80, 2] = 180
    pixels = laser_pixels(frame, 50, 120)
    # Both one-pixel stripes survive, with no fictitious centre at x=50.
    assert len(pixels) == 80
    assert set(pixels[:, 0]) == {20.0, 80.0}


def test_isolated_red_pixel_is_rejected():
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[5:45, 20, 2] = 180
    frame[25, 80, 2] = 255
    pixels = laser_pixels(frame, 50, 120)
    assert len(pixels) == 40
    assert np.all(pixels[:, 0] == 20)


def test_region_thresholds_recover_faint_object_without_changing_marker():
    frame = np.full((50, 100, 3), 30, dtype=np.uint8)
    frame[5:45, 20, 2] = 100
    frame[5:45, 80, 2] = 45
    marker_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    marker_mask[:, :50] = 1
    marker = laser_pixels(frame, 50, 60, region_mask=marker_mask)
    strict_object = laser_pixels(frame, 50, 60, region_mask=1 - marker_mask)
    faint_object = laser_pixels(frame, 10, 40, region_mask=1 - marker_mask)
    assert len(marker) == 40
    assert strict_object.shape == (0, 2)
    assert len(faint_object) == 40
    assert np.all(faint_object[:, 0] == 80)


def test_corner_refinement_stays_on_the_detected_boundary():
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (120, 80), (255, 255, 255), -1)
    corners = np.array([[21, 21], [119, 21], [119, 79], [21, 79]], dtype=np.float32)
    refined = refine_marker_corners(frame, [corners])[0]
    assert np.isfinite(refined).all()
    assert np.max(np.linalg.norm(refined - corners, axis=1)) <= 5
    assert refined[0, 0] < 21 and refined[2, 0] > 119


def test_fixed_calibration_matches_marker_identities_and_rejects_jump(monkeypatch):
    from scanner import calibration

    top = np.array([[20, 20], [120, 20], [120, 70], [20, 70]], dtype=np.float32)
    bottom = top + [0, 100]
    detections = iter([[top, bottom], [bottom + 1, top + 1], [top + 40, bottom + 40]])

    class Video:
        released = False

        def read(self):
            return True, np.zeros((220, 160, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    video = Video()
    monkeypatch.setattr(calibration.cv2, "VideoCapture", lambda _: video)
    monkeypatch.setattr(calibration, "marker_candidates", lambda *_: next(detections))
    corners, count = calibrate_markers("unused", np.eye(3), np.zeros(5), 100, 3, refine=False)
    assert count == 2
    assert np.allclose(corners[0], top + 0.5)
    assert np.allclose(corners[1], bottom + 0.5)
    assert video.released
def test_marker_dimensions_match_inner_white_boundary():
    import cv2
    import numpy as np
    from scanner.detection import marker_candidates

    frame = np.full((400, 700, 3), 180, dtype=np.uint8)
    expected = []
    for x in (40, 380):
        cv2.rectangle(frame, (x, 60), (x + 250, 210), (0, 0, 0), -1)
        cv2.rectangle(frame, (x + 10, 70), (x + 240, 200), (255, 255, 255), -1)
        expected.append(np.array([[x+10, 70], [x+240, 70], [x+240, 200], [x+10, 200]]))
    found = sorted(marker_candidates(frame, 2000), key=lambda p: p[:, 0].mean())
    assert len(found) == 2
    for actual, target in zip(found, expected):
        assert np.max(np.linalg.norm(actual - target, axis=1)) < 3
