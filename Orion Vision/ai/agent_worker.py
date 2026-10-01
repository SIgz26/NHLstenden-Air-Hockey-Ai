"""100 Hz background inference worker for a live SAC controller."""

from __future__ import annotations

import time
from collections.abc import Callable

from ai.sac_controller import LiveSACController, check_sac_runtime
from PyQt5.QtCore import QThread, pyqtSignal

ActionSink = Callable[[float, float], None]


class SACAgentWorker(QThread):
    """Run policy inference on a dedicated thread at a configured rate.

    ``action_sink`` is an optional actuator adapter, such as a non-blocking
    enqueue operation for a PLC/OPC UA worker. Do not perform synchronous
    network I/O in the callback. Every predicted action is also emitted for
    the GUI and external consumers.
    """

    action_updated = pyqtSignal(float, float)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        controller: LiveSACController,
        action_sink: ActionSink | None = None,
        frequency_hz: float = 100.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be greater than zero")
        self.controller = controller
        self._action_sink = action_sink
        self._period_ns = int(1_000_000_000 / frequency_hz)
        self._running = False
        self.completed_steps = 0
        self.deadline_misses = 0

    def set_action_sink(self, action_sink: ActionSink | None) -> None:
        """Set the non-blocking action consumer used by an actuator adapter."""
        self._action_sink = action_sink

    def stop(self) -> None:
        """Request the inference loop to stop after its current iteration."""
        self._running = False

    def run(self) -> None:
        if self.controller.model is None:
            try:
                check_sac_runtime()
            except RuntimeError as exc:
                self.error_occurred.emit(str(exc))
                return
            self.error_occurred.emit("No SAC model is loaded")
            return

        self._running = True
        deadline_ns = time.perf_counter_ns()
        while self._running:
            try:
                target_vx, target_vy = self.controller.predict_action()
                action_sink = self._action_sink
                if action_sink is not None:
                    action_sink(target_vx, target_vy)
                self.action_updated.emit(target_vx, target_vy)
                self.completed_steps += 1
            except Exception as exc:
                self.error_occurred.emit(f"SAC inference stopped: {exc}")
                break

            deadline_ns += self._period_ns
            now_ns = time.perf_counter_ns()
            remaining_ns = deadline_ns - now_ns
            if remaining_ns > 0:
                self.usleep(max(1, int(remaining_ns / 1_000)))
            else:
                self.deadline_misses += 1
                if -remaining_ns >= self._period_ns:
                    deadline_ns = now_ns
        self._running = False
