$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    # Brief page 3, Figure 3: inner white rectangle is 23 x 13 cm.
    & .\.venv\Scripts\python.exe laser_scanner.py `
        --video ..\laser_scanner_data\LaserScanner_project_data\data\cup1.mp4 `
        --intrinsics ..\laser_scanner_data\LaserScanner_project_data\calibration\K.txt `
        --distortion ..\laser_scanner_data\LaserScanner_project_data\calibration\dist.txt `
        --marker-width 0.23 --marker-height 0.13 --reference-frame p1 `
        --min-red-excess 10 --min-red 60 `
        --output output\cup1_p1.ply `
        --debug-video output\cup1_p1_debug.mp4
    if ($LASTEXITCODE -ne 0) { throw "Scanner failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
