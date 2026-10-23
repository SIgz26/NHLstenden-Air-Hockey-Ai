from abc import ABC, abstractmethod
import numpy as np

class BaseOpponent(ABC):
    """Abstracte basisklasse voor alle Air Hockey tegenstanders."""

    @abstractmethod
    def get_action(self, puck_pos: np.ndarray, puck_vel: np.ndarray, opponent_pos: np.ndarray, field_length: float, field_width: float, mallet_radius: float) -> np.ndarray:
        """Geeft de gewenste doelsnelheid (desired_velocity) terug."""
        pass