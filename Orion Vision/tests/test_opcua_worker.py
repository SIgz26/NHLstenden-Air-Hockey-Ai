import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from asyncua import ua

from core.opcua_worker import (
    ERROR_NODE,
    HOMED_NODE,
    POSE_VALID_NODE,
    POSE_X_NODE,
    POSE_Y_NODE,
    STATE_TEXT_NODE,
    TRACKING_NODE,
    VISION_SEQ_NODE,
    WRITE_INTERVAL_S,
    OpcUaWorker,
    limit_velocity,
)


class _Node:
    def __init__(self, value=None, on_read=None) -> None:
        self.value = value
        self.on_read = on_read
        self.writes = []
        self.write_value = AsyncMock(side_effect=self._write)

    async def _write(self, data_value) -> None:
        self.writes.append(data_value)

    async def read_value(self):
        if self.on_read is not None:
            self.on_read()
        return self.value


class _Client:
    def __init__(self, worker: OpcUaWorker) -> None:
        self.pose_read_count = 0
        self.write_batches = []
        self.read_batches = []

        def stop_after_second_pose_read() -> None:
            self.pose_read_count += 1
            if self.pose_read_count >= 2:
                worker.stop()

        self.nodes = {
            worker.pose_x_node_id: _Node(12.5),
            worker.pose_y_node_id: _Node(98.25),
            POSE_VALID_NODE: _Node(True),
            HOMED_NODE: _Node(True),
            TRACKING_NODE: _Node(False),
            ERROR_NODE: _Node(False),
            STATE_TEXT_NODE: _Node("Ready"),
        }
        self.nodes["vx"] = _Node()
        self.nodes["vy"] = _Node()
        self.nodes["seq"] = _Node()
        self.nodes[worker.pose_x_node_id].on_read = stop_after_second_pose_read

    def get_node(self, node_id: str) -> _Node:
        if node_id.endswith("rTargetVx"):
            return self.nodes["vx"]
        if node_id.endswith("rTargetVy"):
            return self.nodes["vy"]
        if node_id.endswith("nVisionSeqRx"):
            return self.nodes["seq"]
        return self.nodes[node_id]

    async def write_values(self, nodes, values):
        self.write_batches.append((nodes, values))
        for node, value in zip(nodes, values, strict=True):
            node.writes.append(ua.DataValue(value))

    async def read_values(self, nodes):
        self.read_batches.append(nodes)
        return [await node.read_value() for node in nodes]


def test_gvl_node_ids_match_codesys_structure() -> None:
    assert VISION_SEQ_NODE == (
        "ns=4;s=|var|CODESYS Control for Linux SL.Application.GVL.nVisionSeqRx"
    )
    assert POSE_X_NODE == (
        "ns=4;s=|var|CODESYS Control for Linux SL.Application.GVL_Machine.pose.rX"
    )
    assert POSE_Y_NODE == (
        "ns=4;s=|var|CODESYS Control for Linux SL.Application.GVL_Machine.pose.rY"
    )
    assert POSE_VALID_NODE.endswith("GVL_Machine.pose.bValid")
    assert HOMED_NODE.endswith("GVL_Machine.bHomed")
    assert TRACKING_NODE.endswith("GVL_Machine.bTracking")
    assert ERROR_NODE.endswith("GVL_Machine.bError")
    assert STATE_TEXT_NODE.endswith("GVL_Machine.sStateText")


def test_limit_velocity_allows_forward_limit() -> None:
    vx, vy = limit_velocity(0.0, -1200.0)

    assert (vx, vy) == pytest.approx((0.0, -1000.0))


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ((1200.0, 0.0), (850.0, 0.0)),
        ((0.0, 1200.0), (0.0, 850.0)),
    ],
)
def test_limit_velocity_caps_lateral_and_reverse_axes(requested, expected) -> None:
    assert limit_velocity(*requested) == pytest.approx(expected)


def test_limit_velocity_respects_motor_axis_limit() -> None:
    vx, vy = limit_velocity(850.0, -850.0)

    assert max(abs(vx - vy), abs(vx + vy)) == pytest.approx(1000.0)
    assert vx > 0.0
    assert vy < 0.0


def test_opcua_write_interval_is_100_hz() -> None:
    assert WRITE_INTERVAL_S == pytest.approx(1.0 / 100.0)


def test_opcua_write_loop_writes_float_setpoints_and_sequence() -> None:
    worker = OpcUaWorker()
    worker.set_target_velocity(150.0, -50.0)
    client = _Client(worker)
    statuses = []
    validities = []
    worker.plc_status_updated.connect(
        lambda homed, tracking, has_error, state_text: statuses.append(
            (homed, tracking, has_error, state_text)
        )
    )
    worker.plc_pose_validity_updated.connect(validities.append)

    asyncio.run(worker._write_loop(client))

    initial_x = client.nodes["vx"].writes[0].Value
    initial_y = client.nodes["vy"].writes[0].Value
    first_x = client.nodes["vx"].writes[1].Value
    first_y = client.nodes["vy"].writes[1].Value
    assert initial_x.Value == 0.0
    assert initial_y.Value == 0.0
    assert first_x.VariantType == ua.VariantType.Float
    assert first_y.VariantType == ua.VariantType.Float
    assert first_x.Value == 150.0
    assert first_y.Value == -50.0
    assert client.nodes["vx"].writes[-1].Value.Value == 0.0
    assert client.nodes["vy"].writes[-1].Value.Value == 0.0
    assert [item.Value.Value for item in client.nodes["seq"].writes] == [1, 2, 3]
    assert all(
        item.Value.VariantType == ua.VariantType.UInt32
        for item in client.nodes["seq"].writes
    )
    assert statuses == [(True, False, False, "Ready")] * 2
    assert validities == [True, True]
    assert len(client.write_batches) == 3
    assert len(client.read_batches) == 2
    assert worker.roundtrip_ms_avg >= 0.0
    assert worker.loop_hz > 0.0
