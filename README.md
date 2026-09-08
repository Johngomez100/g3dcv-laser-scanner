# G3DCV final project - laser-line 3-D scanner

This project reconstructs a coloured point cloud from one of the course laser-sweep videos. It deliberately has no manual click/selection step: every frame is analysed automatically.

## Code structure and reading guide

The scanner is split by responsibility. Start with `scanner/pipeline.py` to follow
the reconstruction steps, then open the module for the step you want to study.

| File | Responsibility |
| --- | --- |
| `laser_scanner.py` | Small launcher; also preserves the original imports used by tests and notebooks. |
| `scanner/cli.py` | Defines command-line arguments and starts the run. |
| `scanner/pipeline.py` | Reads video frames, connects detection to geometry, accumulates points, and writes debug video and timing statistics. |
| `scanner/detection.py` | Detects marker corners and laser pixels, checks marker membership, and samples surface colours. |
| `scanner/calibration.py` | Refines marker corners and combines consistent observations from the initial frames. |
| `scanner/geometry.py` | Defines planes, estimates marker poses, back-projects rays, intersects rays with planes, and fits the laser plane. |
| `scanner/point_cloud.py` | Downsamples points and writes the coloured PLY file. |
| `test_geometry.py`, `test_detection.py`, `test_pipeline.py` | Check geometry, thin and separated stripes, calibration stability, and invalid intersections. |

The execution order is `laser_scanner.py` -> `scanner/cli.py` ->
`scanner/pipeline.py`. The pipeline calls the detection, geometry, and point-cloud
functions as needed. Existing run commands still use `laser_scanner.py`.
The current detection and calibration defaults are explained below.

Keep the entire `scanner` folder alongside `laser_scanner.py` when copying or
submitting the project; the launcher is no longer a standalone file.

## What it implements

1. Undistort the frame with the supplied camera intrinsics and radial distortion coefficients.
2. Detect the two rectangular calibration markers, refine their corners to subpixel precision, and take the median of consistent detections across the first 10 frames. Keep this calibration fixed for the stationary scene.
3. Use the known marker dimensions and `solvePnP` to estimate the markers' 3-D planes in camera coordinates.
4. Segment red laser pixels, reject tiny disconnected blobs and retain one intensity-weighted centre per contiguous stripe segment in each row. Preserve separate segments on different surfaces and thin stripes; require nearby-row support.
5. Back-project marker laser pixels through `K^-1`, intersect them with their known marker planes, and use SVD to fit the laser plane.
6. Intersect all detected laser rays, including those on both markers, with that laser plane, as required by step 2(d) of the brief. Estimate surface colours from nearby non-laser pixels.
7. Transform to the requested P1 reference frame, retain all valid samples by default, report timing, and write an RGB ASCII PLY. Voxel downsampling is optional.

## Setup

Use a Python environment with NumPy and OpenCV:

```powershell
python -m pip install -r requirements.txt
```

Download the final-project data package from Moodle, then keep the videos plus `K.txt` and `dist.txt` outside the submission folder. The exam brief explicitly says not to submit data.

The project brief, page 3, Figure 3 explicitly labels the inner white rectangle as 23 x 13 cm and the outer black boundary as 25 x 15 cm. The detector prioritizes the enclosed white interiors, so use 0.23 x 0.13 metres. This resolves the ambiguous/inconsistent `outer border: 25x13 - th 1` text in the separate template.

## Run

```powershell
python laser_scanner.py `
  --video ..\laser_scanner_data\LaserScanner_project_data\data\cup1.mp4 `
  --intrinsics ..\laser_scanner_data\LaserScanner_project_data\calibration\K.txt `
  --distortion ..\laser_scanner_data\LaserScanner_project_data\calibration\dist.txt `
  --marker-width 0.23 --marker-height 0.13 --reference-frame p1 `
  --output output\cup1_p1.ply `
  --debug-video output\cup1_p1_debug.mp4
```

`run_cup1.ps1` runs this configuration. P1 is the upper wall marker: origin at its inner top-left, X right, Y down, Z outward toward the object, in metres. This intentionally left-handed export follows the user's diagram; internal ray geometry stays in camera coordinates. `--reference-frame camera` exports the camera coordinates used in the brief's equations.

Open `output\cup1_p1.ply` in MeshLab. Inspect the debug MP4: green polygons must follow the inner white boundaries and cyan dots should follow the laser. Tune thresholds if needed. The JSON includes processing time and processed frames per second. `--voxel-size 0` is the default and preserves all reconstructed samples; a positive value explicitly requests a reduced cloud. `--colour-radius 0` retains raw laser-tinted colours for comparison.

## Detection settings and assumptions

- Default laser thresholds are `--min-red 60 --min-red-excess 10`, tested on `cup1.mp4`. They apply to markers and other surfaces unless overridden with `--marker-min-red`, `--marker-min-red-excess`, `--object-min-red`, or `--object-min-red-excess`.
- Marker and object settings can be tuned independently. Keep the marker thresholds conservative because their samples determine the laser plane. Lowering the object excess threshold to 6 admitted extensive desk detections in our comparison; it is not the recommended setting.
- `--marker-mode fixed` (default) assumes the camera and markers remain stationary. `--calibration-frames 10` sets the initial calibration window; observations moving more than 5 pixels from the initial marker pair are excluded. Subpixel refinement is limited to a local 5-pixel displacement.
- `--marker-mode dynamic` re-detects marker corners for comparison. `--no-refine-markers` disables corner refinement. Dynamic mode does not compensate for a moving camera when accumulating the cloud.
- The JSON report includes calibration observations, fixed corner coordinates, effective thresholds, and median/95th-percentile laser-plane RMS residuals, in the marker coordinate units. Residuals measure fit consistency, not ground-truth reconstruction accuracy.
- Debug video includes only successfully reconstructed frames. The PLY includes valid detected laser samples on the wall, desk, object, and both marker interiors. Frames without enough samples on both markers cannot determine the laser plane and are skipped.

Run checks with `python -m pytest -q` if pytest is installed in your development environment; it is not required for reconstruction.

Subpixel corner refinement, stationary calibration, and separate object/marker settings were motivated by comparison with [LolloneS/3D-Laser-Scanner](https://github.com/LolloneS/3D-Laser-Scanner). The implementation here retains NumPy/OpenCV and uses segment-wise sampling with connected-component filtering, without adding DBSCAN or Open3D dependencies.
