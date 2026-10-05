import numpy as np
from .base_opponent import BaseOpponent

class DefenderOpponent(BaseOpponent):
    """Simpele verdediger die alleen de Y-positie van de puck volgt op de doellijn."""

    def get_action(self, puck_pos: np.ndarray, puck_vel: np.ndarray, opponent_pos: np.ndarray, field_length: float, field_width: float, mallet_radius: float) -> np.ndarray:
        half_length = field_length / 2.0
        target = np.asarray(
            [half_length * 0.72, float(np.clip(puck_pos[1], -0.28, 0.28))],
            dtype=np.float64,
        )
        desired_velocity = np.clip((target - opponent_pos) / 0.25, -1.5, 1.5)
        return desired_velocity