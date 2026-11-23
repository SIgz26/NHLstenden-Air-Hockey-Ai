import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ai.opponents.attacker import ComplexOpponent
from ai.scripted_worker import ScriptedOpponentWorker
from core.coordinate_contract import MALLET_RADIUS_M
from core.sim_bridge import SimBridge


def _action(
    opponent: ComplexOpponent,
    puck_pos: tuple[float, float],
    puck_vel: tuple[float, float],
    opponent_pos: tuple[float, float],
) -> np.ndarray:
    return opponent.get_action(
        puck_pos=np.asarray(puck_pos, dtype=np.float64),
        puck_vel=np.asarray(puck_vel, dtype=np.float64),
        opponent_pos=np.asarray(opponent_pos, dtype=np.float64),
        field_length=2.0,
        field_width=1.0,
        mallet_radius=MALLET_RADIUS_M,
    )


def test_complex_opponent_defaults_preserve_existing_tuning():
    opponent = ComplexOpponent()

    assert opponent.max_speed == 3.5
    assert opponent.kp_scale == 1.0
    assert opponent.min_x_ratio == 0.0
    assert opponent.strike_px_min == 0.1
    assert opponent.strike_vx_threshold == -0.8


def test_complex_opponent_custom_minimum_x_clips_strike_target():
    opponent = ComplexOpponent(
        max_speed=10.0,
        kp_scale=100.0,
        min_x_ratio=0.75,
    )

    action = _action(
        opponent,
        puck_pos=(0.4, 0.0),
        puck_vel=(0.0, 0.0),
        opponent_pos=(0.6, 0.0),
    )

    assert opponent.state == "STRIKE"
    assert action[0] > 0.0


def test_complex_opponent_strike_threshold_keeps_incoming_puck_in_defend():
    opponent = ComplexOpponent(
        strike_px_min=0.28,
        strike_vx_threshold=-0.25,
    )

    _action(
        opponent,
        puck_pos=(0.4, 0.0),
        puck_vel=(-0.3, 0.0),
        opponent_pos=(0.8, 0.0),
    )

    assert opponent.state == "DEFEND"


def test_complex_opponent_kp_scale_and_per_mode_speed_limit():
    args = ((-0.8, 0.0), (-1.0, 0.0), (0.5, 0.0))
    default_action = _action(ComplexOpponent(), *args)
    scaled_action = _action(
        ComplexOpponent(kp_scale=0.5, max_speed=0.6),
        *args,
    )
    capped_action = _action(
        ComplexOpponent(kp_scale=0.5, max_speed=0.4),
        *args,
    )

    np.testing.assert_allclose(scaled_action, default_action * 0.5)
    assert np.linalg.norm(capped_action) <= 0.4
    np.testing.assert_allclose(capped_action, [0.4, 0.0])


def test_live_complex_opponent_uses_plc_tuning():
    bridge = SimBridge()

    worker = ScriptedOpponentWorker(bridge, "Complex Opponent")

    assert isinstance(worker.opponent, ComplexOpponent)
    assert worker.opponent.max_speed == 1.0
    assert worker.opponent.kp_scale == 0.60
    assert worker.opponent.min_x_ratio == 0.25
    assert worker.opponent.strike_px_min == 0.28
    assert worker.opponent.strike_vx_threshold == -0.25
