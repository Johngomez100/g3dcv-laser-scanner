# Oral-exam notes: laser-line scanner

## A two-minute demo narrative

"The scanner uses structured light. I estimate the camera-coordinate poses of two calibration rectangles from refined corners in the initial frames. Since the camera and markers are stationary, those marker planes stay fixed throughout the scan. The laser crosses both markers, so their illuminated pixels provide 3-D points on the current laser plane. Once that plane is known, each other laser pixel back-projects to a camera ray and its intersection with the laser plane is a reconstructed object point. Accumulating those points across the sweep gives a coloured PLY point cloud."

Show the annotated video first, then the PLY in MeshLab. Point out marker contours, laser samples, and the final cloud.

## Be ready to derive these operations

- **Undistortion:** radial distortion bends image positions. I undistort before contour detection and set `distCoeffs=0` in `solvePnP`, since its inputs are already undistorted.
- **Pose from a marker:** with local corners `(0,0,0), (W,0,0), (W,H,0), (0,H,0)`, `solvePnP` gives `R,t`. A marker plane is `p=t`, `n=R(0,0,1)^T`.
- **Back-projection:** pixel `u=(x,y,1)^T` corresponds to camera ray direction `d=K^-1u / ||K^-1u||`, starting at the camera centre.
- **Ray-plane intersection:** for plane `(p,n)` and ray `z d`, solve `z=(p dot n)/(d dot n)`. Discard nearly parallel rays and negative depths, applying the same validity mask to point positions and colour samples.
- **Laser-plane fit:** its point is the centroid of calibration intersections. Its normal is the right-singular vector of the centred point matrix with the smallest singular value. This minimises squared orthogonal distance.
- **Colour:** sampled from the undistorted original frame; stored as RGB in PLY even though OpenCV reads BGR.

## Design choices, strengths, and limitations

| Choice | Why it helps | Limitation / improvement |
| --- | --- | --- |
| Otsu + adaptive threshold, contours, `approxPolyDP` | Simple and fast for the supplied high-contrast paper markers | Sensitive to poor contrast, clutter, or partial occlusion; ArUco tags or temporal tracking would be more robust. |
| Refined corners and fixed median calibration | Reduces detection jitter across a stationary scan | Assumes fixed camera and markers; the initial detections must be correct. |
| `solvePnP` with known metric marker size | Gives metric scale and direct camera-frame planes | Incorrect marker dimensions or corner order corrupts the reconstruction. |
| Red-excess mask and one weighted centre per contiguous row segment | Preserves separate surfaces in the same row and avoids fictitious points between stripes | Fails with red objects/specular highlights; background subtraction or a calibrated laser wavelength filter would improve it. |
| Fit a laser plane every frame | Handles hand movement of the laser projector | Requires visible laser segments on both markers in every accepted frame. |
| Voxel downsampling | Makes point cloud smaller and less redundant | Keeps first point per voxel rather than averaging; statistical outlier removal could improve visual quality. |

## Likely questions

1. **Why two markers?** A single marker gives laser points on only one 3-D line, which cannot determine a plane. Points from two non-parallel planes provide a non-collinear 3-D set.
2. **Why is `solvePnP` needed?** Image pixels alone have no depth. The marker's known dimensions constrain its pose and therefore its 3-D plane.
3. **What happens if the laser ray is parallel to a plane?** The intersection denominator tends to zero, so depth is undefined or numerically unstable; the program rejects it.
4. **Why does the output have real scale?** All reconstructed geometry inherits the unit used for the known marker width and height.
5. **Why undistort first?** The pinhole projection used by `K^-1` and `solvePnP` assumes rectilinear pixel positions.
