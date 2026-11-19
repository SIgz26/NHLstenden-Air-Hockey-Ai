"""Non-blocking OPC UA velocity and PLC-pose worker."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import deque

from asyncua import Client, ua
from PyQt5.QtCore import QThread, pyqtSignal

from Python_OPCUA import NODE_PREFIX, PLC_URL

GVL_MACHINE_PREFIX = "|var|CODESYS Control for Linux SL.Application.GVL_Machine."
VISION_SEQ_NODE = f"ns=4;s={NODE_PREFIX}nVisionSeqRx"
POSE_X_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}pose.rX"
POSE_Y_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}pose.rY"
POSE_VALID_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}pose.bValid"
HOMED_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}bHomed"
TRACKING_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}bTracking"
ERROR_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}bError"
STATE_TEXT_NODE = f"ns=4;s={GVL_MACHINE_PREFIX}sStateText"
TARGET_VX_NODE = f"ns=4;s={NODE_PREFIX}rTargetVx"
TARGET_VY_NODE = f"ns=4;s={NODE_PREFIX}rTargetVy"

MAX_LATERAL_SPEED_MM_S = 850.0
MAX_REVERSE_SPEED_MM_S = 850.0
MAX_FORWARD_SPEED_MM_S = 1000.0
MAX_MOTOR_SPEED_MM_S = 1000.0
MAX_VISION_SEQUENCE = 0xFFFFFFFF
WRITE_INTERVAL_S = 0.01


def limit_velocity(vx_mm_s: float, vy_mm_s: float) -> tuple[float, float]:
    """Apply asymmetric PLC axis limits and the motor-axis speed limit."""
    vx = float(vx_mm_s)
    vy = float(vy_mm_s)
    if not math.isfinite(vx) or not math.isfinite(vy):
        raise ValueError("Target velocity components must be finite")

    vx = max(-MAX_LATERAL_SPEED_MM_S, min(MAX_LATERAL_SPEED_MM_S, vx))
    vy_limit = MAX_FORWARD_SPEED_MM_S if vy < 0.0 else MAX_REVERSE_SPEED_MM_S
    vy = max(-vy_limit, min(vy_limit, vy))
    motor_speed = max(abs(vx - vy), abs(vx + vy))
    scale = MAX_MOTOR_SPEED_MM_S / motor_speed if motor_speed > MAX_MOTOR_SPEED_MM_S else 1.0
    if scale < 1.0:
        vx *= scale
        vy *= scale
    return vx, vy


class OpcUaWorker(QThread):
    """Keep one OPC UA session alive on a background asyncio event loop."""

    connection_status = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    plc_pose_updated = pyqtSignal(float, float)
    plc_pose_validity_updated = pyqtSignal(bool)
    plc_status_updated = pyqtSignal(bool, bool, bool, str)
    zero_written = pyqtSignal(object)
    cycle_metrics_updated = pyqtSignal(float, float)

    def __init__(
        self,
        url: str = PLC_URL,
        pose_x_node: str = POSE_X_NODE,
        pose_y_node: str = POSE_Y_NODE,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.url = url
        self.pose_x_node_id = pose_x_node
        self.pose_y_node_id = pose_y_node
        self._mailbox_lock = threading.Lock()
        self._target_velocity = (0.0, 0.0)
        self._target_revision = 0
        self._stop_requested = threading.Event()
        self._seq = 0
        self._pose_error_reported = False
        self._roundtrip_samples_ms: deque[float] = deque(maxlen=50)
        self._loop_period_samples_s: deque[float] = deque(maxlen=50)
        self._last_cycle_start: float | None = None
        self._last_metrics_emit = 0.0
        self.roundtrip_ms_avg = 0.0
        self.loop_hz = 0.0

    def set_target_velocity(self, vx_mm_s: float, vy_mm_s: float) -> int:
        """Replace the pending setpoint without waiting for network I/O."""
        target = limit_velocity(vx_mm_s, vy_mm_s)
        with self._mailbox_lock:
            self._target_velocity = target
            self._target_revision = (self._target_revision + 1) & 0x7FFFFFFF
            return self._target_revision

    @property
    def accepting_commands(self) -> bool:
        return not self._stop_requested.is_set()

    def stop(self) -> None:
        """Request a final zero write and orderly worker shutdown."""
        self.set_target_velocity(0.0, 0.0)
        self._stop_requested.set()

    def run(self) -> None:
        try:
            asyncio.run(self._connection_loop())
        except Exception as exc:
            self.error_occurred.emit(f"OPC UA worker stopped: {exc}")
            self.connection_status.emit("Error")

    async def _connection_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                self.connection_status.emit(f"Connecting to {self.url}")
                async with Client(url=self.url, timeout=1.0) as client:
                    self.connection_status.emit("Connected")
                    await self._write_loop(client)
            except Exception as exc:
                if not self._stop_requested.is_set():
                    self.error_occurred.emit(f"OPC UA connection error: {exc}")
                    self.connection_status.emit("Disconnected")
                    await asyncio.sleep(0.5)
        self.connection_status.emit("Disconnected")

    async def _write_loop(self, client: Client) -> None:
        node_vx = client.get_node(TARGET_VX_NODE)
        node_vy = client.get_node(TARGET_VY_NODE)
        node_seq = client.get_node(VISION_SEQ_NODE)
        node_pose_x = client.get_node(self.pose_x_node_id)
        node_pose_y = client.get_node(self.pose_y_node_id)
        node_pose_valid = client.get_node(POSE_VALID_NODE)
        node_homed = client.get_node(HOMED_NODE)
        node_tracking = client.get_node(TRACKING_NODE)
        node_error = client.get_node(ERROR_NODE)
        node_state_text = client.get_node(STATE_TEXT_NODE)
        write_nodes = [node_vx, node_vy, node_seq]
        read_nodes = [
            node_pose_x,
            node_pose_y,
            node_pose_valid,
            node_homed,
            node_tracking,
            node_error,
            node_state_text,
        ]

        try:
            first_cycle = True
            while not self._stop_requested.is_set():
                loop = asyncio.get_running_loop()
                cycle_start = loop.time()
                if self._last_cycle_start is not None:
                    self._loop_period_samples_s.append(
                        cycle_start - self._last_cycle_start
                    )
                    self.loop_hz = len(self._loop_period_samples_s) / sum(
                        self._loop_period_samples_s
                    )
                self._last_cycle_start = cycle_start
                with self._mailbox_lock:
                    vx, vy = self._target_velocity
                    revision = self._target_revision
                requested_zero = vx == 0.0 and vy == 0.0
                if first_cycle:
                    vx = vy = 0.0
                    first_cycle = False
                self._seq = (self._seq % MAX_VISION_SEQUENCE) + 1
                roundtrip_start = loop.time()
                await asyncio.wait_for(
                    client.write_values(
                        write_nodes,
                        [
                            ua.Variant(float(vx), ua.VariantType.Float),
                            ua.Variant(float(vy), ua.VariantType.Float),
                            ua.Variant(int(self._seq), ua.VariantType.UInt32),
                        ],
                    ),
                    timeout=1.0,
                )
                if requested_zero:
                    self.zero_written.emit(revision)

                try:
                    values = await asyncio.wait_for(
                        client.read_values(read_nodes),
                        timeout=1.0,
                    )
                    plc_state = self._read_plc_state(values)
                except Exception as exc:
                    if not self._pose_error_reported:
                        self.error_occurred.emit(
                            "OPC UA machine pose read failed "
                            f"({self.pose_x_node_id}, {self.pose_y_node_id}): {exc}"
                        )
                        self._pose_error_reported = True
                else:
                    roundtrip_ms = (loop.time() - roundtrip_start) * 1000.0
                    self._roundtrip_samples_ms.append(roundtrip_ms)
                    self.roundtrip_ms_avg = float(
                        sum(self._roundtrip_samples_ms)
                        / len(self._roundtrip_samples_ms)
                    )
                    (
                        pose_x,
                        pose_y,
                        pose_valid,
                        homed,
                        tracking,
                        has_error,
                        state_text,
                    ) = plc_state
                    self._pose_error_reported = False
                    self.plc_status_updated.emit(
                        homed, tracking, has_error, state_text
                    )
                    self.plc_pose_validity_updated.emit(pose_valid)
                    if pose_valid:
                        self.plc_pose_updated.emit(pose_x, pose_y)

                now = loop.time()
                if now - self._last_metrics_emit >= 1.0 and self._roundtrip_samples_ms:
                    self.cycle_metrics_updated.emit(
                        self.roundtrip_ms_avg, self.loop_hz
                    )
                    self._last_metrics_emit = now

                remaining = WRITE_INTERVAL_S - (
                    now - cycle_start
                )
                if remaining > 0.0 and not self._stop_requested.is_set():
                    await asyncio.sleep(remaining)
        finally:
            self._seq = (self._seq % MAX_VISION_SEQUENCE) + 1
            try:
                await asyncio.wait_for(
                    client.write_values(
                        write_nodes,
                        [
                            ua.Variant(0.0, ua.VariantType.Float),
                            ua.Variant(0.0, ua.VariantType.Float),
                            ua.Variant(int(self._seq), ua.VariantType.UInt32),
                        ],
                    ),
                    timeout=1.0,
                )
            except Exception as exc:
                self.error_occurred.emit(f"OPC UA final zero write failed: {exc}")

    @staticmethod
    def _read_plc_state(
        values: list[object],
    ) -> tuple[float, float, bool, bool, bool, bool, str]:
        if len(values) != 7:
            raise ValueError(f"Expected 7 PLC state values, received {len(values)}")
        pose_x, pose_y = float(values[0]), float(values[1])
        if not math.isfinite(pose_x) or not math.isfinite(pose_y):
            raise ValueError("PLC pose must contain finite coordinates")
        if not isinstance(values[6], str):
            raise TypeError("GVL_Machine.sStateText must be a string")
        return (
            pose_x,
            pose_y,
            bool(values[2]),
            bool(values[3]),
            bool(values[4]),
            bool(values[5]),
            values[6],
        )
