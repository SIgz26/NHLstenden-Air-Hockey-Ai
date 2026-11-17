import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from asyncua import ua
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Python_OPCUA import write_target_velocities


def test_write_target_velocities_sends_float_array_to_plc() -> None:
    client = AsyncMock()
    client.connect_and_get_server_endpoints.return_value = []
    node = AsyncMock()
    node.read_value.return_value = [150.5, -50.0]
    client.get_node = MagicMock(return_value=node)

    with patch("Python_OPCUA.Client", return_value=client):
        asyncio.run(write_target_velocities(150.5, -50.0))

    client.get_node.assert_called_once_with(
        "ns=4;s=|var|CODESYS Control for Linux SL.Application.GVL.aTargetVelocity"
    )
    data_value = node.write_value.await_args.args[0]
    assert data_value.Value.Value == [150.5, -50.0]
    assert data_value.Value.VariantType == ua.VariantType.Float
    assert data_value.Value.is_array is True
    node.read_value.assert_awaited_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
