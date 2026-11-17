"""Background worker for the live rule-based opponent controllers."""

from __future__ import annotations

import time
import threading
from collections.abc import Callable

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ai.opponents.attacker import AttackerOpponent, ComplexOpponent
from ai.opponents.base_opponent import BaseOpponent
from ai.opponents.defender import DefenderOpponent
from core.opcua_worker import limit_velocity
from core.sim_bridge import SimBridge

ActionSink = Callable[[float, float], None]

OPPONENTS: dict[str, type[BaseOpponent]] = {
    "Complex Opponent": ComplexOpponent,
    "Attacker": AttackerOpponent,
    "Defender": DefenderOpponent,
}


class ScriptedOpponentWorker(QThread):
    """Run a scripted opponent against the newest calibrated live observation."""

    action_updated = pyqtSignal(float, float)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        sim_bridge: SimBridge,
        opponent_name: str,
        action_sink: ActionSink | None = None,
        frequency_hz: float = 50.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if opponent_name not in OPPONENTS:
            raise ValueError(f"Unknown scripted opponent: {opponent_name}")
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be greater than zero")
        self.sim_bridge = sim_bridge
        self.opponent: BaseOpponent = OPPONENTS[opponent_name]()
        self.opponent_name = opponent_name
        self._action_sink = action_sink
        self._period_ns = int(1_000_000_000 / frequency_hz)
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        deadline_ns = time.perf_counter_ns()
        try:
            while not self._stop_requested.is_set():
                action = self._calculate_action()
                if self._action_sink is not None:
                    self._action_sink(*action)
                self.action_updated.emit(*action)

                deadline_ns += self._period_ns
                remaining_ns = deadline_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    self.usleep(max(1, int(remaining_ns / 1_000)))
                elif -remaining_ns >= self._period_ns:
                    deadline_ns = time.perf_counter_ns()
        except Exception as exc:
            self.error_occurred.emit(
                f"{self.opponent_name} controller stopped: {exc}"
            )
            if self._action_sink is not None:
                try:
                    self._action_sink(0.0, 0.0)
                except Exception as stop_exc:
                    self.error_occurred.emit(
                        f"{self.opponent_name} safe-stop enqueue failed: {stop_exc}"
                    )
            self.action_updated.emit(0.0, 0.0)
        finally:
            self._stop_requested.set()

    def _calculate_action(self) -> tuple[float, float]:
        if (
            not self.sim_bridge.physical_table_calibrated
            or not self.sim_bridge.has_fresh_puck_detection(max_age_s=0.25)
        ):
            return 0.0, 0.0

        physical_state = self.sim_bridge.get_opponent_physical_state()
        if physical_state is None:
            return 0.0, 0.0
        puck_pos_m, puck_vel_mps, mallet_pos_m = physical_state
        field_length_mm = SimBridge.PHYSICAL_TABLE_LENGTH_MM
        field_width_mm = SimBridge.PHYSICAL_TABLE_WIDTH_MM

        desired_velocity_mps = np.asarray(
            self.opponent.get_action(
                puck_pos=puck_pos_m,
                puck_vel=puck_vel_mps,
                opponent_pos=mallet_pos_m,
                field_length=field_length_mm / 1000.0,
                field_width=field_width_mm / 1000.0,
                mallet_radius=0.048,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if desired_velocity_mps.shape != (2,) or not np.isfinite(desired_velocity_mps).all():
            raise ValueError("Opponent must return a finite 2D velocity in m/s")

        # Policies return (depth, width); PLC Y decreases as depth x advances.
        vx_mm_s = float(desired_velocity_mps[1] * 1000.0)
        vy_mm_s = float(desired_velocity_mps[0] * 1000.0)
        return limit_velocity(vx_mm_s, vy_mm_s)
