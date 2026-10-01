"""Manual four-click air-hockey table calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.sim_bridge import SimBridge


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
