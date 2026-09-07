# G3DCV final project - laser-line 3-D scanner

This project reconstructs a coloured point cloud from one of the course laser-sweep videos. It deliberately has no manual click/selection step: every frame is analysed automatically.

## What it implements

1. Undistort the frame with the supplied camera intrinsics and radial distortion coefficients.
2. Detect the two bright rectangular calibration markers with thresholding, contours and polygon approximation.
3. Use the known marker dimensions and `solvePnP` to estimate the markers' 3-D planes in camera coordinates.
4. Segment red laser pixels, retaining one intensity-weighted centre point per image row.
5. Back-project marker laser pixels through `K^-1`, intersect them with their known marker planes, and use SVD to fit the laser plane.
6. Intersect the remaining laser rays with that laser plane; store their original pixel colours.
7. Voxel-downsample the accumulated points and write a MeshLab-compatible ASCII PLY.

## Setup

Use a Python environment with NumPy and OpenCV:

```powershell
python -m pip install -r requirements.txt
```

Download the final-project data package from Moodle, then keep the videos plus `K.txt` and `dist.txt` outside the submission folder. The exam brief explicitly says not to submit data.

The marker dimensions must match `plane.pdf`. Read the values printed on that PDF and pass them in metres. They set the scale of the output point cloud.

## Run

```powershell
python laser_scanner.py `
  --video C:\path\to\cup1.mp4 `
  --intrinsics C:\path\to\K.txt `
  --distortion C:\path\to\dist.txt `
  --marker-width 0.210 --marker-height 0.297 `
  --output output\cup1.ply `
  --debug-video output\cup1_debug.mp4
```

The `0.210 x 0.297` values above are only an A4-sized example, not a replacement for the dimensions stated on the supplied `plane.pdf`.

Open `output\cup1.ply` in MeshLab. Also inspect the debug MP4: green polygons should trace both markers and cyan dots should trace only the laser line. If not, tune `--min-red`, `--min-red-excess`, and (if necessary) `--min-marker-area` for the provided videos.

## Submission checklist

- Run the scanner on at least one of the four provided videos and verify the generated PLY opens in MeshLab.
- Include source code, `requirements.txt`, and this README; do **not** include videos, images, or calibration data.
- Package the code as `<name>_<surname>_exam.zip` at least one week before the booked oral date.
- Bring the laptop, the code, a tested video/data copy for the live demo, and the generated PLY/debug MP4.

See `EXAM_NOTES.md` for the explanation flow and limitations to discuss at the oral exam.
