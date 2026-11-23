"""Manual and automatic air-hockey table calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import cv2
import numpy as np

from core.coordinate_contract import TABLE_LENGTH_M, TABLE_WIDTH_M

if TYPE_CHECKING:
    from core.sim_bridge import SimBridge


Point = tuple[float, float]
TableCorners = tuple[Point, Point, Point, Point]

TABLE_LENGTH_MM = TABLE_LENGTH_M * 1000.0
TABLE_WIDTH_MM = TABLE_WIDTH_M * 1000.0
CIRCLE_OFFSET_LENGTH_MM = 410.0
CIRCLE_OFFSET_WIDTH_MM = 240.0


@dataclass(frozen=True)
class AutoCalibrationDetection:
    """Detected circle centers and the projected physical table corners."""

    circle_centers_px: TableCorners
    circle_radii_px: tuple[float, float, float, float]
    corners_px: TableCorners


@overload
def auto_detect_table_corners(
    frame: np.ndarray,
    is_rgb: bool = False,
    circle_offset_length_mm: float = CIRCLE_OFFSET_LENGTH_MM,
    circle_offset_width_mm: float = CIRCLE_OFFSET_WIDTH_MM,
    *,
    return_details: Literal[False] = False,
) -> TableCorners | None: ...


@overload
def auto_detect_table_corners(
    frame: np.ndarray,
    is_rgb: bool = False,
    circle_offset_length_mm: float = CIRCLE_OFFSET_LENGTH_MM,
    circle_offset_width_mm: float = CIRCLE_OFFSET_WIDTH_MM,
    *,
    return_details: Literal[True],
) -> AutoCalibrationDetection | None: ...


def auto_detect_table_corners(
    frame: np.ndarray,
    is_rgb: bool = False,
    circle_offset_length_mm: float = CIRCLE_OFFSET_LENGTH_MM,
    circle_offset_width_mm: float = CIRCLE_OFFSET_WIDTH_MM,
    *,
    return_details: bool = False,
) -> TableCorners | AutoCalibrationDetection | None:
    """Detect four black face-off circles and project corners in Orion order.

    Camera frames are BGR by default, matching both Orion camera workers.
    Set ``is_rgb`` when supplying an RGB frame. Circle offsets are measured
    inward from the full table edges. ``None`` means that one or more circles
    could not be detected reliably.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        raise ValueError("frame must be a color image with at least 32 pixels per side")
    offsets = np.asarray(
        [circle_offset_length_mm, circle_offset_width_mm], dtype=np.float64
    )
    if (
        not np.isfinite(offsets).all()
        or circle_offset_length_mm <= 0.0
        or circle_offset_length_mm >= TABLE_LENGTH_MM / 2.0
        or circle_offset_width_mm <= 0.0
        or circle_offset_width_mm >= TABLE_WIDTH_MM / 2.0
    ):
        raise ValueError("circle offsets must be finite and inside the table bounds")

    # Normalize channel order before using the red-emphasis projection. This
    # prevents a BGR camera frame from treating its blue channel as red.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    _, green, red = cv2.split(bgr)
    red_i16 = red.astype(np.int16)
    green_i16 = green.astype(np.int16)
    gray = np.clip(
        red_i16 + np.maximum(0, red_i16 - green_i16), 0, 255
    ).astype(np.uint8)

    height, width = gray.shape
    quadrants = (
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (width // 2, height // 2, width, height),
        (0, height // 2, width // 2, height),
    )
    detected: list[tuple[float, float, float]] = []
    for x0, y0, x1, y1 in quadrants:
        region = gray[y0:y1, x0:x1]
        region_min = min(region.shape)
        min_radius = max(4, round(region_min * 0.012))
        max_radius = max(min_radius + 1, round(region_min * 0.22))
        circles = None
        for accumulator_threshold in (30.0, 24.0, 18.0):
            circles = cv2.HoughCircles(
                region,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(1.0, region_min * 0.35),
                param1=100.0,
                param2=accumulator_threshold,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is not None and circles.size:
                break
        if circles is None or not circles.size:
            return None

        candidates = circles[0]
        region_center = np.asarray(
            [region.shape[1] / 2.0, region.shape[0] / 2.0]
        )
        circle = min(
            candidates,
            key=lambda item: float(
                np.sum((item[:2].astype(np.float64) - region_center) ** 2)
            ),
        )
        refined_x, refined_y = _refine_circle_center(
            gray, x0 + float(circle[0]), y0 + float(circle[1]), float(circle[2])
        )
        detected.append((refined_x, refined_y, float(circle[2])))

    # The quadrant order above is image TL, TR, BR, BL. SimBridge's canonical
    # table axes are width across X and length along Y; landscape cameras see
    # the long axis horizontally, so rotate the sequence by one corner.
    if width > height:
        detected = [detected[3], detected[0], detected[1], detected[2]]

    circle_centers = np.asarray(
        [[x, y] for x, y, _ in detected], dtype=np.float32
    )
    half_width = TABLE_WIDTH_MM / 2.0
    half_length = TABLE_LENGTH_MM / 2.0
    circle_width = half_width - circle_offset_width_mm
    circle_length = half_length - circle_offset_length_mm
    circle_points_mm = np.asarray(
        [
            [-circle_width, -circle_length],
            [circle_width, -circle_length],
            [circle_width, circle_length],
            [-circle_width, circle_length],
        ],
        dtype=np.float32,
    )
    table_points_mm = np.asarray(
        [
            [-half_width, -half_length],
            [half_width, -half_length],
            [half_width, half_length],
            [-half_width, half_length],
        ],
        dtype=np.float32,
    )
    circle_to_mm = cv2.getPerspectiveTransform(circle_centers, circle_points_mm)
    mm_to_circle = np.linalg.inv(circle_to_mm)
    corners_px_array = cv2.perspectiveTransform(
        table_points_mm.reshape(1, 4, 2), mm_to_circle
    )[0]
    corner_points = [
        (float(point[0]), float(point[1])) for point in corners_px_array
    ]
    corners: TableCorners = (
        corner_points[0], corner_points[1], corner_points[2], corner_points[3]
    )
    center_points = [
        (float(point[0]), float(point[1])) for point in circle_centers
    ]
    centers: TableCorners = (
        center_points[3], center_points[2], center_points[1], center_points[0]
    )
    radii_values = [float(circle[2]) for circle in detected]
    radii = (
        radii_values[3], radii_values[2], radii_values[1], radii_values[0]
    )
    corners = (corners[3], corners[2], corners[1], corners[0])
    if return_details:
        return AutoCalibrationDetection(
            circle_centers_px=centers,
            circle_radii_px=radii,
            corners_px=corners,
        )
    return corners


def _refine_circle_center(
    gray: np.ndarray, center_x: float, center_y: float, radius: float
) -> Point:
    """Use a local Otsu contour centroid to refine a Hough center."""
    search_radius = max(6, int(round(radius * 1.6)))
    height, width = gray.shape
    x0 = max(0, int(np.floor(center_x - search_radius)))
    y0 = max(0, int(np.floor(center_y - search_radius)))
    x1 = min(width, int(np.ceil(center_x + search_radius + 1)))
    y1 = min(height, int(np.ceil(center_y + search_radius + 1)))
    patch = gray[y0:y1, x0:x1]
    _, threshold = cv2.threshold(
        patch, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    expected_area = np.pi * radius**2
    center_local = np.asarray([center_x - x0, center_y - y0])
    best: tuple[float, Point] | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not expected_area * 0.3 <= area <= expected_area * 2.2:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        centroid = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
        distance = float(np.linalg.norm(np.asarray(centroid) - center_local))
        if distance > radius * 0.2:
            continue
        if best is None or distance < best[0]:
            best = (distance, (centroid[0] + x0, centroid[1] + y0))

    rows, columns = np.nonzero(threshold)
    if len(rows) >= 12:
        local_x = columns.astype(np.float64) + x0
        local_y = rows.astype(np.float64) + y0
        distance = np.hypot(local_x - center_x, local_y - center_y)
        annulus = (distance >= radius * 0.62) & (distance <= radius * 1.38)
        edge_x, edge_y = local_x[annulus], local_y[annulus]
        if len(edge_x) >= 12:
            design = np.column_stack(
                (2.0 * edge_x, 2.0 * edge_y, np.ones(len(edge_x)))
            )
            target = edge_x**2 + edge_y**2
            solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            fitted_radius_squared = (
                solution[2] + solution[0] ** 2 + solution[1] ** 2
            )
            fitted_center = (float(solution[0]), float(solution[1]))
            fitted_radius = (
                float(np.sqrt(fitted_radius_squared))
                if fitted_radius_squared > 0.0
                else 0.0
            )
            fit_distance = float(
                np.hypot(
                    fitted_center[0] - center_x,
                    fitted_center[1] - center_y,
                )
            )
            if (
                rank == 3
                and radius * 0.55 <= fitted_radius <= radius * 1.5
                and fit_distance <= radius * 0.25
            ):
                if best is not None:
                    fitted_center = (
                        fitted_center[0] * 0.9 + best[1][0] * 0.1,
                        fitted_center[1] * 0.9 + best[1][1] * 0.1,
                    )
                return fitted_center
    return best[1] if best is not None else (center_x, center_y)


class ManualTableCalibrator:
    """Collect the four table corners in TL, TR, BR, BL click order."""

    CORNER_LABELS = ("TL", "TR", "BR", "BL")

    def __init__(self) -> None:
        self._corners: list[tuple[float, float]] = []

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        """Return all currently selected points in click order."""
        return tuple(self._corners)

    @property
    def next_corner_label(self) -> str | None:
        """Return the label expected for the next click, or None when done."""
        if len(self._corners) >= len(self.CORNER_LABELS):
            return None
        return self.CORNER_LABELS[len(self._corners)]

    @property
    def is_complete(self) -> bool:
        """Whether all four corners have been selected."""
        return len(self._corners) == 4

    def add_corner(self, point: tuple[float, float]) -> str:
        """Add one finite source-frame pixel coordinate and return its label."""
        if self.is_complete:
            raise RuntimeError("all four table corners are already selected")
        coordinates = np.asarray(point, dtype=np.float64)
        if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
            raise ValueError("corner must be a finite (x, y) point")
        label = self.CORNER_LABELS[len(self._corners)]
        self._corners.append((float(coordinates[0]), float(coordinates[1])))
        return label

    def apply(self, sim_bridge: SimBridge) -> None:
        """Commit the four selected corners to SimBridge's perspective map."""
        if not self.is_complete:
            raise RuntimeError("exactly four table corners are required")
        sim_bridge.set_table_corners(*self._corners)

    def reset(self) -> None:
        """Discard any selected or committed-preview points."""
        self._corners.clear()
