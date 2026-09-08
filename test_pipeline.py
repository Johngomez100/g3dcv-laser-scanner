"""Ensure rejected intersections cannot misalign point positions and colours."""

import json
import numpy as np

from scanner import pipeline
from scanner.cli import build_parser
from scanner.geometry import Plane


def test_invalid_intersections_are_removed_from_points_and_colours(tmp_path, monkeypatch):
    camera = tmp_path / "K.txt"
    distortion = tmp_path / "dist.txt"
    np.savetxt(camera, np.eye(3))
    np.savetxt(distortion, np.zeros(5))
    output = tmp_path / "scan.ply"
    args = build_parser().parse_args([
        "--video", "synthetic.mp4", "--intrinsics", str(camera),
        "--distortion", str(distortion), "--output", str(output),
        "--marker-width", "0.25", "--marker-height", "0.13",
        "--marker-mode", "dynamic", "--no-refine-markers", "--min-marker-samples", "1",
    ])

    class Video:
        count = 0
        released = False

        def isOpened(self):
            return True

        def read(self):
            self.count += 1
            return (True, np.full((80, 80, 3), 30, np.uint8)) if self.count == 1 else (False, None)

        def release(self):
            self.released = True

    video = Video()
    first = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    plane = Plane(np.array([0., 0., 2.]), np.array([0., 0., 1.]))
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _: video)
    monkeypatch.setattr(pipeline.cv2, "undistort", lambda frame, *_: frame)
    monkeypatch.setattr(pipeline, "marker_candidates", lambda *_: [first, first + [20, 0]])
    monkeypatch.setattr(pipeline, "estimate_marker_planes", lambda *_: [plane, plane])
    monkeypatch.setattr(pipeline, "fit_plane", lambda _: plane)
    monkeypatch.setattr(pipeline, "laser_pixels", lambda *_, **kwargs: np.array([[5., 5.], [25., 5.], [50., 5.], [60., 5.]]))
    monkeypatch.setattr(pipeline, "backproject", lambda *_: np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., -1.]]))
    stats = pipeline.process_video(args)
    # Both marker samples and the valid scene sample must survive; the ray
    # behind the camera must be rejected without misaligning the colours.
    assert stats.points_before_downsampling == 3
    assert "element vertex 3" in output.read_text()
    assert json.loads(output.with_suffix(".json").read_text())["points_written"] == 3
    assert video.released
