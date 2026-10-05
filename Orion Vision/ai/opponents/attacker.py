import numpy as np
from .base_opponent import BaseOpponent

class AttackerOpponent(BaseOpponent):
    """Agressieve tegenstander die de puck opzoekt zodra deze op zijn helft komt."""

    def get_action(self, puck_pos: np.ndarray, puck_vel: np.ndarray, opponent_pos: np.ndarray, field_length: float, field_width: float, mallet_radius: float) -> np.ndarray:
        half_length = field_length / 2.0
        puck_x, puck_y = puck_pos[0], puck_pos[1]

        if puck_x > 0:
            target_x = np.clip(puck_x, mallet_radius, half_length - mallet_radius)
            target_y = np.clip(puck_y, -0.38, 0.38)
        else:
            target_x = half_length * 0.75
            target_y = float(np.clip(puck_y, -0.28, 0.28))

        target = np.asarray([target_x, target_y], dtype=np.float64)
        desired_velocity = np.clip((target - opponent_pos) / 0.10, -2.5, 2.5)
        return desired_velocity

import numpy as np

class ComplexOpponent(BaseOpponent):
    """
    Geavanceerde Airhockey-tegenstander.
    Verbeteringen: Dynamische interceptie, anti-eigen-doelpunt paden, 
    en adaptieve reactiesnelheden.
    """

    PUCK_RADIUS = 0.031
    WALL_MARGIN = 0.01
    
    def __init__(self) -> None:
        self.state = "DEFEND"
        self.last_target = np.array([0.0, 0.0])

    def _predict_puck_intercept(
        self,
        puck_pos: np.ndarray,
        puck_vel: np.ndarray,
        intercept_x: float,
        field_width: float,
    ) -> float:
        px, py = puck_pos
        vx, vy = puck_vel
        y_limit = (field_width / 2.0) - self.PUCK_RADIUS

        if vx <= 0.05: # Puck komt niet naar ons toe
            return float(np.clip(py, -y_limit, y_limit))

        time_to_x = (intercept_x - px) / vx
        if time_to_x < 0:
            return float(np.clip(py, -y_limit, y_limit))

        predicted_y = py + vy * time_to_x

        # Reflectie-logica (modulo rekenen voor oneindige bounces)
        period = 4.0 * y_limit
        phase = (predicted_y + y_limit) % period
        if phase <= 2.0 * y_limit:
            res = -y_limit + phase
        else:
            res = 3.0 * y_limit - phase
        return float(np.clip(res, -y_limit, y_limit))

    def get_action(
        self,
        puck_pos: np.ndarray,
        puck_vel: np.ndarray,
        opponent_pos: np.ndarray,
        field_length: float,
        field_width: float,
        mallet_radius: float,
    ) -> np.ndarray:
        # Constanten en grenzen
        half_len = field_length / 2.0
        y_limit = (field_width / 2.0) - mallet_radius
        x_min, x_max = mallet_radius, half_len - mallet_radius
        
        px, py = puck_pos
        vx, vy = puck_vel
        ox, oy = opponent_pos

        # 1. DYNAMISCHE STATE MACHINE
        dist_to_puck = np.linalg.norm(puck_pos - opponent_pos)
        
        # Bepaal staat met hysterese
        if px > ox - 0.02: 
            self.state = "RETREAT"
        elif px > 0.1 and vx > -0.8:
            self.state = "STRIKE"
        else:
            self.state = "DEFEND"

        # 2. TARGET BEREKENING
        target = np.array([ox, oy])

        if self.state == "RETREAT":
            # VOORKOM EIGEN DOELPUNT: Als de puck tussen ons en het doel ligt, 
            # beweeg dan eerst naar de zijkant (Y-as) voordat we naar achteren (X-as) gaan.
            side_threshold = mallet_radius + self.PUCK_RADIUS + 0.05
            if abs(oy - py) < side_threshold and px > ox:
                # Ga opzij
                target_y = y_limit * np.sign(py if py != 0 else 1.0)
                target_x = ox # Blijf op huidige X
            else:
                target_x = half_len * 0.85
                target_y = py * 0.5 # Blijf enigszins voor het doel
            target = np.array([target_x, target_y])

        elif self.state == "STRIKE":
            # PREDICTIVE STRIKE: Mik op het vijandelijke doel
            # We mikken op de hoeken van het doel om de keeper te passeren
            target_goal_y = 0.15 if py < 0 else -0.15
            goal_vec = np.array([-half_len, target_goal_y]) - puck_pos
            goal_vec /= np.linalg.norm(goal_vec)
            
            # Bereken waar de puck zal zijn over 0.05 seconden voor een 'lead' shot
            future_puck = puck_pos + puck_vel * 0.05
            
            # Positioneer achter de toekomstige puck
            strike_pos = future_puck - goal_vec * (mallet_radius + self.PUCK_RADIUS)
            target = strike_pos

        else: # DEFEND
            # Dynamische verdedigingslijn: als puck ver weg is, kom iets naar voren
            # Als puck dichtbij is, trek terug naar 0.8
            dynamic_intercept_x = half_len * np.clip(0.6 + (px / half_len) * 0.2, 0.6, 0.85)
            
            pred_y = self._predict_puck_intercept(puck_pos, puck_vel, dynamic_intercept_x, field_width)
            
            # Blijf binnen de 'goalie box'
            target = np.array([dynamic_intercept_x, np.clip(pred_y, -0.25, 0.25)])

        # 3. BEWEGINGSCONTROLE (SMOOTHING)
        # Clip targets binnen het speelveld
        target[0] = np.clip(target[0], x_min, x_max)
        target[1] = np.clip(target[1], -y_limit, y_limit)

        # Snelheidsberekening
        diff = target - opponent_pos
        
        # Gebruik hogere gains voor aanval, lagere voor verdediging (stabiliteit)
        if self.state == "STRIKE":
            kp = 18.0 
            max_speed = 3.5
        elif self.state == "RETREAT":
            kp = 12.0
            max_speed = 3.0
        else:
            kp = 10.0
            max_speed = 2.2

        desired_vel = diff * kp
        
        # Vectoriale snelheidsbegrenzing
        speed = np.linalg.norm(desired_vel)
        if speed > max_speed:
            desired_vel = (desired_vel / speed) * max_speed

        return desired_vel