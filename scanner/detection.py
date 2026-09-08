"""Find marker corners and laser pixels, and sample surface colours from a frame."""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).ravel()  # y - x
    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(differences)]
    ordered[3] = corners[np.argmax(differences)]
    return ordered


def marker_candidates(frame: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Find bright convex quadrilaterals, keeping the two largest likely markers.

    The provided target is a white printed rectangle.  Otsu thresholding makes
    this resilient to a moderate change in room illumination; the optional
    adaptive threshold catches the opposite contrast convention.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 4
    )
    candidates: list[tuple[float, np.ndarray]] = []
    inner_candidates: list[tuple[float, np.ndarray]] = []
    for binary in (binary_otsu, binary_adaptive):
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour_index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            # Thresholding can turn the full image background into one very
            # large quadrilateral. It is not a physical marker.
            if x <= 1 or y <= 1 or x + width >= frame.shape[1] - 1 or y + height >= frame.shape[0] - 1:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            candidates.append((area, order_corners(polygon.reshape(4, 2))))
            # White interior enclosed by a black border: an even-depth contour
            # inside a hole. Excludes the larger outer black-border boundary.
            depth = 0
            parent = hierarchy[0, contour_index, 3]
            while parent != -1:
                depth += 1
                parent = hierarchy[0, parent, 3]
            if depth >= 2 and depth % 2 == 0:
                inner_candidates.append(candidates[-1])

    # RETR_LIST and the two threshold methods can find the same sheet repeatedly.
    selected: list[np.ndarray] = []
    for _, corners in sorted(inner_candidates or candidates, key=lambda item: item[0], reverse=True):
        centre = np.mean(corners, axis=0)
        if all(np.linalg.norm(centre - np.mean(other, axis=0)) > 20 for other in selected):
            selected.append(corners)
        if len(selected) == 2:
            break
    return selected


def refine_marker_corners(frame: np.ndarray, corners: list[np.ndarray]) -> list[np.ndarray]:
    """Refine detected boundaries locally without jumping across the black border."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    refined = []
    for polygon in corners:
        candidate = cv2.cornerSubPix(
            gray, polygon.astype(np.float32).reshape(-1, 1, 2).copy(),
            (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        ).reshape(4, 2)
        if (np.isfinite(candidate).all()
                and np.max(np.linalg.norm(candidate - polygon, axis=1)) <= 5
                and cv2.isContourConvex(candidate)):
            refined.append(candidate)
        else:
            refined.append(polygon.copy())
    return refined


def red_laser_mask(frame: np.ndarray, min_red_excess: int, min_red: int) -> np.ndarray:
    """Threshold red excess without eroding thin, genuine laser stripes.

    Connected-component filtering in laser_pixels removes isolated detections.
    A 3x3 opening here would erase stripes only one or two pixels wide.
    """
    blue, green, red = cv2.split(frame.astype(np.int16))
    return ((red > min_red) & (red - np.maximum(blue, green) > min_red_excess)).astype(np.uint8)


def laser_pixels(
    frame: np.ndarray, min_red_excess: int, min_red: int,
    min_component_pixels: int = 12, max_row_jump: float = 25.0,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return one centre per contiguous stripe segment per row.

    Separate surfaces can produce several stripes in the same row. Never
    average across the gaps between them: that creates false camera rays.
    The optional region mask lets marker and object thresholds be independent.
    """
    mask = red_laser_mask(frame, min_red_excess, min_red)
    if region_mask is not None:
        mask &= region_mask.astype(np.uint8)
    _, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = statistics[:, cv2.CC_STAT_AREA] >= min_component_pixels
    keep[0] = False
    mask = keep[labels]
    blue, green, red = cv2.split(frame.astype(np.int16))
    score = red - np.maximum(blue, green)
    rows: dict[int, list[float]] = {}
    for y in np.flatnonzero(mask.any(axis=1)):
        xs = np.flatnonzero(mask[y])
        segments = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        rows[int(y)] = [float(np.average(segment, weights=np.maximum(score[y, segment], 1)))
                        for segment in segments]
    pixels = []
    for y, centres in rows.items():
        neighbours = [x for dy in (-2, -1, 1, 2) for x in rows.get(y + dy, [])]
        for x in centres:
            if any(abs(x - other) <= max_row_jump for other in neighbours):
                pixels.append((x, float(y)))
    return np.asarray(pixels, dtype=np.float64).reshape(-1, 2)


def sample_laser_free_colours(
    frame: np.ndarray, pixels: np.ndarray, min_red_excess: int, min_red: int, radius: int
) -> np.ndarray:
    """Estimate surface colour from nearby non-laser pixels, avoiding red laser tint."""
    if radius <= 0:
        indices = np.rint(pixels).astype(np.int32)
        indices[:, 0] = np.clip(indices[:, 0], 0, frame.shape[1] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, frame.shape[0] - 1)
        return frame[indices[:, 1], indices[:, 0]]

    laser = red_laser_mask(frame, min_red_excess, min_red).astype(bool)
    colours: list[np.ndarray] = []
    for x_float, y_float in pixels:
        x, y = int(round(x_float)), int(round(y_float))
        x0, x1 = max(0, x - radius), min(frame.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(frame.shape[0], y + radius + 1)
        neighbourhood = frame[y0:y1, x0:x1]
        usable = neighbourhood[~laser[y0:y1, x0:x1]]
        # If the laser fills the small neighbourhood, retaining the centre
        # colour is preferable to dropping a valid 3-D sample.
        colours.append(np.median(usable, axis=0) if len(usable) else frame[y, x])
    return np.asarray(colours, dtype=np.uint8)


def points_inside_polygon(pixels: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    return np.array([cv2.pointPolygonTest(polygon.astype(np.float32), tuple(pixel), False) >= 0 for pixel in pixels])
