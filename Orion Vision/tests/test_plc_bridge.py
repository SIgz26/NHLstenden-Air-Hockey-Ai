"""Unit and integration tests for core/plc_bridge.py without physical hardware."""

import socket
import struct
import time
from unittest.mock import MagicMock, patch

import pytest

# Pas de import aan naar de werkelijke locatie van je PLCBridge klasse
from core.plc_bridge import PLCBridge


@pytest.fixture
def plc_ip() -> str:
    return "192.168.2.10"


@pytest.fixture
def udp_port() -> int:
    return 5005


@pytest.fixture
def bridge(plc_ip: str, udp_port: int) -> PLCBridge:
    """Fixture initializing PLCBridge with dummy socket."""
    with patch("socket.socket"):
        return PLCBridge(plc_ip, udp_port)


def test_initialization(plc_ip: str, udp_port: int):
    """Test standard initialization and setup."""
    with patch("socket.socket") as mock_socket_cls:
        bridge = PLCBridge(plc_ip, udp_port)
        assert bridge.plc_address == (plc_ip, udp_port)
        assert bridge.seq == 0
        mock_socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)


def test_binary_packing_structure(bridge: PLCBridge, plc_ip: str, udp_port: int):
    """Validate struct binary layout, byte size, and data types."""
    mock_sock = MagicMock()
    bridge.sock = mock_sock

    target_x, target_y = 0.45, -0.12
    vx, vy = 1.25, -0.80
    mode = 2
    valid = True

    fixed_time_ms = 1700000000000
    with patch("time.time", return_value=fixed_time_ms / 1000.0):
        bridge.send_target(target_x, target_y, vx, vy, mode, valid)

    mock_sock.sendto.assert_called_once()
    payload, address = mock_sock.sendto.call_args[0]

    # Controleren dat de payload exact 26 bytes lang is
    assert len(payload) == 26
    assert address == (plc_ip, udp_port)

    # De gepackte bytes weer uitpakken om exact de velden te controleren
    unpacked = struct.unpack("<HHffffBBI", payload)
    seq_unpacked, reserved, tx_u, ty_u, vx_u, vy_u, flags_u, mode_u, ts_u = unpacked

    assert seq_unpacked == 1
    assert reserved == 0
    assert pytest.approx(tx_u, rel=1e-5) == target_x
    assert pytest.approx(ty_u, rel=1e-5) == target_y
    assert pytest.approx(vx_u, rel=1e-5) == vx
    assert pytest.approx(vy_u, rel=1e-5) == vy
    assert flags_u == 1
    assert mode_u == mode
    assert ts_u == (fixed_time_ms & 0xFFFFFFFF)


def test_sequence_overflow(bridge: PLCBridge):
    """Test 16-bit sequence roll-over logic (0xFFFF -> 0x0000)."""
    bridge.sock = MagicMock()
    bridge.seq = 65535  # Max 16-bit unsigned short

    bridge.send_target(0.0, 0.0, 0.0, 0.0, mode=1, valid=True)
    assert bridge.seq == 0  # Moet gereset zijn naar 0 via & 0xFFFF


def test_invalid_flag(bridge: PLCBridge):
    """Verify validity flag encoding (False -> 0)."""
    bridge.sock = MagicMock()
    bridge.send_target(0.0, 0.0, 0.0, 0.0, mode=1, valid=False)

    payload = bridge.sock.sendto.call_args[0][0]
    unpacked = struct.unpack("<HHffffBBI", payload)
    flags_u = unpacked[6]
    assert flags_u == 0


def test_loopback_udp_communication(plc_ip: str):
    """Integration test: actual UDP socket send & receive via local loopback."""
    # Gebruik poort 0 om een willekeurige vrije poort toe te wijzen op 127.0.0.1
    receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_sock.bind(("127.0.0.1", 0))
    receiver_port = receiver_sock.getsockname()[1]
    receiver_sock.settimeout(1.0)

    # Initialiseer de echte PLCBridge verbonden met de lokale loopback receiver
    real_bridge = PLCBridge("127.0.0.1", receiver_port)

    try:
        real_bridge.send_target(0.5, -0.3, 2.0, -1.5, mode=3, valid=True)

        data, addr = receiver_sock.recvfrom(1024)
        assert len(data) == 26

        seq, _, x, y, vx, vy, flags, mode, ts = struct.unpack("<HHffffBBI", data)
        assert seq == 1
        assert pytest.approx(x, abs=1e-4) == 0.5
        assert pytest.approx(y, abs=1e-4) == -0.3
        assert pytest.approx(vx, abs=1e-4) == 2.0
        assert pytest.approx(vy, abs=1e-4) == -1.5
        assert flags == 1
        assert mode == 3
    finally:
        receiver_sock.close()
        real_bridge.sock.close()