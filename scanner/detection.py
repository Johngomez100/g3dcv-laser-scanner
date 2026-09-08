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
    for binary in (binary_otsu, binary_adaptive):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
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

    # RETR_LIST and the two threshold methods can find the same sheet repeatedly.
    selected: list[np.ndarray] = []
    for _, corners in sorted(candidates, key=lambda item: item[0], reverse=True):
        centre = np.mean(corners, axis=0)
        if all(np.linalg.norm(centre - np.mean(other, axis=0)) > 20 for other in selected):
            selected.append(corners)
        if len(selected) == 2:
            break
    return selected


def red_laser_mask(frame: np.ndarray, min_red_excess: int, min_red: int) -> np.ndarray:
    """Return a cleaned binary mask of pixels that are likely part of the laser."""
    blue, green, red = cv2.split(frame.astype(np.int16))
    red_excess = red - np.maximum(blue, green)
    mask = (red > min_red) & (red_excess > min_red_excess)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def laser_pixels(
    frame: np.ndarray, min_red_excess: int, min_red: int, min_component_pixels: int = 12, max_row_jump: float = 25.0
) -> np.ndarray:
    """Extract one weighted centre sample per row from spatially consistent laser components.

    Tiny red blobs are removed with connected-component filtering.  Samples that
    jump abruptly between adjacent rows are discarded, which rejects isolated
    red reflections while retaining the continuous projected laser stripe.
    """
    mask = red_laser_mask(frame, min_red_excess, min_red)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep_labels = np.flatnonzero(statistics[:, cv2.CC_STAT_AREA] >= min_component_pixels)
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels) == 0:
        return np.empty((0, 2), dtype=np.float64)
    mask = np.isin(labels, keep_labels)
    blue, green, red = cv2.split(frame.astype(np.int16))
    red_excess = red - np.maximum(blue, green)

    pixels: list[tuple[float, float]] = []
    for y, row in enumerate(mask):
        xs = np.flatnonzero(row)
        if len(xs) == 0:
            continue
        # A red score-weighted centroid avoids emitting every pixel in a thick line.
        weights = red_excess[y, xs].astype(np.float64)
        x = float(np.average(xs, weights=np.maximum(weights, 1.0)))
        pixels.append((x, float(y)))
    samples = np.asarray(pixels, dtype=np.float64)
    if len(samples) < 3:
        return samples

    # Keep a row only when its position agrees with a nearby row.  The test on
    # both sides avoids preserving a single outlying red reflection.
    keep = np.ones(len(samples), dtype=bool)
    for index in range(1, len(samples) - 1):
        previous_y, next_y = samples[index - 1, 1], samples[index + 1, 1]
        if next_y - previous_y <= 2.0:
            close_to_previous = abs(samples[index, 0] - samples[index - 1, 0]) <= max_row_jump
            close_to_next = abs(samples[index, 0] - samples[index + 1, 0]) <= max_row_jump
            keep[index] = close_to_previous or close_to_next
    return samples[keep]


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
