# Comparison with G3DCV_final_project_2025.pdf

All six pages were read, including rendered Figure 3. Source: ../G3DCV_final_project_2025.pdf, revision December 10, 2025. This is a development checklist; the brief does not require an additional report.

| Brief reference | Requirement or recommendation | Implementation / verification |
| --- | --- | --- |
| Pages 1-2, steps 1 and 2(a) | Automatically identify two markers and the laser | Thresholding, contour approximation, nested white-interior selection; connected-component filtering and per-segment stripe centres. No user clicks or manual masks. |
| Page 2, step 2(b); A.2 | Back-project with normalized K inverse rays | scanner/geometry.py backproject. |
| Page 2, step 2(c); A.3-A.4 | Intersect with P1/P2 and fit a laser plane every accepted frame | Marker intersections followed by centred SVD. Its smallest right-singular vector is equivalent to the smallest covariance eigenvector prescribed in A.4. |
| Page 2, step 2(d) | Intersect ALL laser rays with PL | Corrected pipeline to include marker samples as well as the object and surrounding surfaces. Invalid/behind-camera intersections are rejected. |
| Page 3, Figure 3 | Known marker dimensions | Explicit inner 23 x 13 cm, outer 25 x 15 cm. Use inner corners and 0.23 x 0.13 m. Earlier 0.23 x 0.11 and 0.25 x 0.13 interpretations were wrong. |
| Section 2.1 | Python processing one supplied video frame by frame | CLI accepts all four supplied videos; repeatable run uses cup1 and supplied K/dist. Default stride is 1. Other videos have not yet been validated. |
| Section 2.1 | Coloured cloud of all reconstructed scene points | BGR neighbourhood colours converted to RGB in PLY. Downsampling now disabled by default. Stripe centre sampling estimates the laser line; it does not emit every pixel across its thickness. |
| Section 2.1 | Save PLY and verify it opens in MeshLab | Verified output/cup1_p1.ply opens in MeshLab 2025.07 on 2026-09-08; viewport reports 444,099 vertices. |
| Section 3 | Visual algorithm feedback encouraged | Annotated video and automatic marker-boundary inspection image; JSON statistics and plane-fit residuals. Debug video includes accepted frames only. |
| Section 3 | Nearly real-time processing encouraged, not mandatory | Timings measured; no claim of real-time operation. Python per-sample colour medians and frame undistortion remain costs. |
| Section 3; page 4 | Comment code and explain choices in oral demo | Modular commented source and EXAM_NOTES.md. Student still needs to understand and demonstrate it. |
| A.1, A.3 | Point/unit normal planes and normalized rays | Plane dataclass, normalized directions, dot-product intersections; tested. |
| A.5 | solvePnP or homography for marker poses | solvePnP plus Rodrigues, plane point T and normal R[:,2]. |
| A.6 | Threshold, findContours, approxPolyDP, four corners | Implemented with convexity, size, hierarchy and subpixel checks. Epsilon=10 is a suggested starting value, not a compulsory constant. |
| A.7 | Undistort before processing; zero distortion in solvePnP | Implemented both during initial fixed calibration and per-frame reconstruction. |
| Section 4 | Source-only named ZIP, Moodle at least a week before oral, arrange date | Submission instructions retained in README. Do not zip the whole project: exclude output/, videos, calibration, images, caches, .venv and .validation_packages. No submission or email performed. |

The user's P1 export frame is applied after reconstruction. It does not change the camera-centred equations in the appendix. Fixed calibration assumes stationary camera and markers; dynamic detection is available but does not perform moving-camera registration. Detector errors, red objects, specular reflections and missing stripe segments can still affect quality. The geometry tests and a valid PLY do not prove ground-truth accuracy.

## Current validation

- 12 tests passed, including inner-boundary selection, P1 axes, thin/separated laser segments and retention of marker intersections.
- All 933 cup1 frames processed; 597 frames yielded reconstructions; 444,099 points written with no voxel reduction.
- Run time 251.57 seconds, 3.71 fps, below the 15 fps input rate. Near-real-time performance remains unmet.
- Every exported coordinate is finite and every RGB channel is within 0-255.
- Initial fixed-calibration overlay inspected: both green polygons trace the inner white boundaries.
- PLY imported in a separate MeshLab project, preserving the user's existing project. Visible outliers remain; import verification does not establish metric accuracy or eliminate detection artefacts.
- Outputs: output/cup1_p1.ply, output/cup1_p1_debug.mp4, output/cup1_p1.json. Previous outputs predate these corrections.
