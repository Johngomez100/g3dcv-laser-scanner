# Cup1 reconstruction comparison

Historical comparison predating the project-brief audit. The dimensions and
marker exclusion used below are superseded: Figure 3 specifies inner 23 x 13 cm
and outer 25 x 15 cm. Current validation, including MeshLab import and all-scene
reconstruction, is recorded in BRIEF_COMPLIANCE.md. Do not use the dimensions
below for a new run.

Baseline: repository commit `7bd5470`, before the detector/calibration changes.
Comparison input: the local course `cup1.mp4` (933 frames), supplied `K.txt` and
`dist.txt`, outer marker dimensions 0.25 x 0.13, and voxel size 0.002.
Metric units assume the PDF dimensions are centimetres, as documented in README.

## Changes tested in sequence

All sampled runs processed every tenth frame (94 frames total), with R > 60
and red excess > 10 unless indicated.

| Stage | Reconstructed frames | Samples before downsampling | PLY points |
| --- | ---: | ---: | ---: |
| Original detector and per-frame marker calibration | 58 | 16,435 | 6,529 |
| Separate row segments, preserving thin stripes | 60 | 20,920 | 8,006 |
| Fixed median marker calibration, no subpixel refinement | 59 | 20,454 | 7,812 |
| Fixed calibration with subpixel refinement | 59 | 20,462 | 7,871 |
| Object R > 50, marker R > 60, excess > 10 for both | 59 | 20,568 | 7,899 |

Refinement reduced median laser-plane RMS residual from 0.001427 to 0.001259
in marker units in the sampled fixed-calibration runs. This measures fit
consistency, not geometric accuracy against a ground-truth scan.

An additional object-threshold experiment (R > 40, excess > 6) admitted extensive
desk detections. It was rejected as a recommended configuration. It also exposed
an existing validity-mask bug: invalid intersections were retained in the point
array after their corresponding pixels had been removed. Positions and colours
now use the same validity mask, with a regression test.

## Full-video comparison

The final default keeps R > 60 and red excess > 10 for both markers and object;
independent overrides remain available. Both runs processed all 933 frames.

| Measurement | Baseline | Improved |
| --- | ---: | ---: |
| Reconstructed frames | 581 | 597 |
| Samples before downsampling | 164,279 | 205,395 |
| PLY points after the same voxel filter | 31,276 | 32,694 |

Local outputs (excluded from submission/Git by the existing output ignore rule):

- `output/improvement_check/cup1_baseline_full.ply`
- `output/cup1_improved.ply`
- `output/cup1_improved_debug.mp4`
- `output/cup1_improved.json`
- `output/cup1_comparison.png`

The comparison image uses the same 25-degree orthographic rotation, scale,
point radius, and uniform colour for both outputs. Shared image limits use
combined 0.5th/99.5th coordinate percentiles; this is a display choice only,
and the PLY files retain all output points. Background fragments remain, and
some isolated detections remain in the debug frame. The handle is more distinct
in this view, but density alone is not evidence of higher geometric accuracy.
No claim of matching the external reference implementation is made.

## Checks

- 10 automated tests pass, including preservation of separate one-pixel stripes,
  isolated-pixel rejection, separate region thresholds, calibration consistency,
  and matching point/colour masks for invalid intersections.
- Both full-run PLY files have finite positions/colours and a vertex count matching
  the JSON report; the improved debug MP4 was decoded and a frame visually checked.
- The new PLY still needs user inspection from multiple angles in MeshLab.
- `--marker-mode fixed` assumes the camera and markers remain stationary.
- Timings were not used as a performance comparison: the full runs ran concurrently
  and only the improved full run wrote debug video.
