"""Eenvoudige, instelbare slag-opponent om slagvormen op de fysieke H-bot te testen."""

from __future__ import annotations

import numpy as np

from core.coordinate_contract import PUCK_RADIUS_M as CONTRACT_PUCK_RADIUS_M

from .base_opponent import BaseOpponent


class SimpleStrikerOpponent(BaseOpponent):
    """Wacht achterin, en slaat zodra de puck voor de mallet komt."""

    # Kies slagvorm om te testen op de tafel:
    # - "CHARGE_AND_SNAP": eerst uitlijnen in breedte, dan recht naar voren uithalen (1.0 m/s H-bot voordeel)
    # - "DIRECT_THROUGH" : direct diagonaal door de puck heen slaan richting tegenstandersdoel
    STRIKE_STYLE = "CHARGE_AND_SNAP"
    AIM_AT_GOAL = False  # False = recht vooruit slaan (makkelijkst te tunen), True = richten op (-0.97, 0.0)

    # Werkruimte (in meters vanaf middenlijn = 0.0 m; eigen achterwand = +0.7358 m)
    HOME_LENGTH_M = 0.56          # Wachtpositie achterin (geeft aanloopruimte)
    MIN_LENGTH_M = 0.11           # Voorste grens (blijft veilig voor de middenlijn-remzone)
    MAX_LENGTH_M = 0.66           # Achterste grens
    MAX_WIDTH_M = 0.42            # Zijwaartse grens

    # Trigger-gebied: "als de puck voor hem is"
    STRIKE_ZONE_MIN_LENGTH_M = 0.08   # Niet vlak op de middenlijn nog aanzetten
    STRIKE_ZONE_MAX_LENGTH_M = 0.54   # Alleen slaan als puck echt vóór de wachtpositie ligt
    PUCK_FRONT_CLEARANCE_M = 0.03     # Puck moet minstens 3 cm vóór de mallet liggen

    # Snelheden & acceleratie (veilig voor tandriemen)
    HOME_SPEED_MPS = 0.65
    ALIGN_SPEED_MPS = 0.80
    STRIKE_FORWARD_SPEED_MPS = 0.95   # Voorwaartse slagsnelheid (max 1.0 m/s op PLC)
    STRIKE_SIDE_SPEED_MPS = 0.80      # Zijwaartse slagsnelheid (max 0.85 m/s op PLC)
    MAX_ACCEL_MPS2 = 2.6
    CONTROL_DT_S = 0.01

    # Slag-parameters om mee te experimenteren
    FOLLOW_THROUGH_M = 0.14           # Hoe ver voorbij de puck de slag doorloopt
    SNAP_WIDTH_TOLERANCE_M = 0.035    # Bij CHARGE_AND_SNAP: hoe strak uitgelijnd vóór hij uithaalt
    LOOKAHEAD_TIME_S = 0.12           # Voorspelt waar de puck over X seconden is
    HOME_TRACK_GAIN = 0.35            # 0.0 = blijft stil in midden wachten, 0.35 = volgt puck licht in breedte

    PUCK_RADIUS_M = CONTRACT_PUCK_RADIUS_M

    def __init__(self) -> None:
        self.mode = "HOME"  # "HOME" -> "STRIKE" -> "RECOVER"
        self._strike_target: np.ndarray | None = None
        self._prev_velocity = np.zeros(2, dtype=np.float64)
        self._prev_mallet_pos: np.ndarray | None = None

    def get_action(
        self,
        puck_pos: np.ndarray,
        puck_vel: np.ndarray,
        opponent_pos: np.ndarray,
        field_length: float,
        field_width: float,
        mallet_radius: float,
    ) -> np.ndarray:
        del field_length
        puck_pos = np.asarray(puck_pos, dtype=np.float64)
        puck_vel = np.asarray(puck_vel, dtype=np.float64)
        opponent_pos = np.asarray(opponent_pos, dtype=np.float64)

        puck_length, puck_width = float(puck_pos[0]), float(puck_pos[1])
        puck_v_len, puck_v_wid = float(puck_vel[0]), float(puck_vel[1])

        width_limit = min(self.MAX_WIDTH_M, field_width / 2.0 - mallet_radius)
        mallet_length = float(np.clip(opponent_pos[0], self.MIN_LENGTH_M, self.MAX_LENGTH_M))
        mallet_width = float(np.clip(opponent_pos[1], -width_limit, width_limit))
        current = np.asarray([mallet_length, mallet_width], dtype=np.float64)

        # Reset snelheidsramp bij simulatie-reset
        if (
            self._prev_mallet_pos is not None
            and float(np.linalg.norm(current - self._prev_mallet_pos)) > 0.15
        ):
            self._prev_velocity = np.zeros(2, dtype=np.float64)
        self._prev_mallet_pos = current.copy()

        # Voorspel puckpositie kort vooruit zodat we niet achter een bewegende puck aan slaan
        pred_length = puck_length + puck_v_len * self.LOOKAHEAD_TIME_S
        pred_width = float(
            np.clip(puck_width + puck_v_wid * self.LOOKAHEAD_TIME_S, -width_limit, width_limit)
        )

        puck_in_front_zone = (
            self.STRIKE_ZONE_MIN_LENGTH_M <= pred_length <= self.STRIKE_ZONE_MAX_LENGTH_M
            and pred_length < mallet_length - self.PUCK_FRONT_CLEARANCE_M
            and abs(pred_width) <= width_limit
        )

        # --- 1. Toestandsmachine: HOME -> STRIKE -> RECOVER ---
        if self.mode == "HOME":
            if puck_in_front_zone:
                self.mode = "STRIKE"
                self._strike_target = self._compute_strike_target(
                    pred_length, pred_width, width_limit, mallet_radius
                )

        elif self.mode == "STRIKE":
            puck_hit_away = puck_v_len < -0.35 or puck_length < self.STRIKE_ZONE_MIN_LENGTH_M
            puck_passed_us = puck_length > mallet_length + 0.02
            reached_end = (
                self._strike_target is not None
                and mallet_length <= self._strike_target[0] + 0.015
            )
            if puck_hit_away or puck_passed_us or reached_end:
                self.mode = "RECOVER"
                self._strike_target = None

        elif self.mode == "RECOVER":
            # Pas weer opnieuw slaan nadat de mallet terug is op de wachtlijn
            if mallet_length >= self.HOME_LENGTH_M - 0.03:
                self.mode = "HOME"

        # --- 2. Snelheid berekenen voor actieve modus ---
        if self.mode == "STRIKE":
            desired_vel = self._compute_strike_velocity(
                current=current,
                pred_length=pred_length,
                pred_width=pred_width,
                width_limit=width_limit,
                mallet_radius=mallet_radius,
            )
        else:
            # HOME of RECOVER: rustig terug naar wachtpositie achterin
            idle_width = float(
                np.clip(puck_width * self.HOME_TRACK_GAIN, -width_limit * 0.6, width_limit * 0.6)
            )
            home_target = np.asarray([self.HOME_LENGTH_M, idle_width], dtype=np.float64)
            desired_vel = (home_target - current) * 4.5
            speed = float(np.linalg.norm(desired_vel))
            if speed > self.HOME_SPEED_MPS:
                desired_vel *= self.HOME_SPEED_MPS / speed

        ramped = self._ramp_velocity(desired_vel)
        self._prev_velocity = ramped
        return ramped

    def _compute_strike_target(
        self,
        puck_length: float,
        puck_width: float,
        width_limit: float,
        mallet_radius: float,
    ) -> np.ndarray:
        """Berekent het eindpunt (follow-through) voorbij de puck."""
        contact_offset = mallet_radius + self.PUCK_RADIUS_M
        target_length = float(
            np.clip(
                puck_length - self.FOLLOW_THROUGH_M,
                self.MIN_LENGTH_M,
                self.MAX_LENGTH_M,
            )
        )

        if not self.AIM_AT_GOAL:
            target_width = float(np.clip(puck_width, -width_limit, width_limit))
            return np.asarray([target_length, target_width], dtype=np.float64)

        # Richt door het middelpunt van de puck richting het tegenstandersdoel (-0.97, 0.0)
        goal = np.asarray([-0.97, 0.0], dtype=np.float64)
        puck = np.asarray([puck_length, puck_width], dtype=np.float64)
        shot_dir = goal - puck
        norm = float(np.linalg.norm(shot_dir))
        if norm > 1e-6:
            shot_dir /= norm
            contact_center = puck - shot_dir * contact_offset
            slope = shot_dir[1] / min(shot_dir[0], -1e-6)
            target_width = contact_center[1] + slope * (target_length - contact_center[0])
        else:
            target_width = puck_width

        target_width = float(np.clip(target_width, -width_limit, width_limit))
        return np.asarray([target_length, target_width], dtype=np.float64)

    def _compute_strike_velocity(
        self,
        current: np.ndarray,
        pred_length: float,
        pred_width: float,
        width_limit: float,
        mallet_radius: float,
    ) -> np.ndarray:
        """Geeft een constante doorslag-snelheid richting het doel i.p.v. vroegtijdig af te remmen."""
        target = self._compute_strike_target(pred_length, pred_width, width_limit, mallet_radius)
        self._strike_target = target

        if self.STRIKE_STYLE == "CHARGE_AND_SNAP":
            width_error = target[1] - current[1]
            # Fase A: eerst snel op de juiste breedte komen terwijl we achterin blijven
            if abs(width_error) > self.SNAP_WIDTH_TOLERANCE_M and current[0] > pred_length + 0.10:
                v_len = (self.HOME_LENGTH_M - current[0]) * 4.0
                v_wid = np.clip(width_error * 9.0, -self.ALIGN_SPEED_MPS, self.ALIGN_SPEED_MPS)
                return np.asarray([v_len, v_wid], dtype=np.float64)

            # Fase B: uitgelijnd -> vol gas recht naar voren door de puck heen!
            v_len = -self.STRIKE_FORWARD_SPEED_MPS
            v_wid = np.clip(width_error * 6.0, -0.35, 0.35)
            return np.asarray([v_len, v_wid], dtype=np.float64)

        # DIRECT_THROUGH: één vloeiende vector met constante slagsnelheid door het raakpunt
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float64)
        direction = delta / dist
        v_len = direction[0] * self.STRIKE_FORWARD_SPEED_MPS
        v_wid = direction[1] * self.STRIKE_SIDE_SPEED_MPS
        return np.asarray([v_len, v_wid], dtype=np.float64)

    def _ramp_velocity(self, desired_velocity: np.ndarray) -> np.ndarray:
        """Vloeiende acceleratieramp zonder plotselinge snelheidssprongen."""
        max_delta = self.MAX_ACCEL_MPS2 * self.CONTROL_DT_S
        delta = desired_velocity - self._prev_velocity
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta > 0.0:
            delta *= max_delta / delta_norm
        velocity = self._prev_velocity + delta

        vx = float(np.clip(velocity[0], -self.STRIKE_FORWARD_SPEED_MPS, self.HOME_SPEED_MPS))
        vy = float(np.clip(velocity[1], -self.STRIKE_SIDE_SPEED_MPS, self.STRIKE_SIDE_SPEED_MPS))
        return np.asarray([vx, vy], dtype=np.float64)


# Backwards-compatible name used by existing callers and saved configurations.
StrikerDefenderOpponent = SimpleStrikerOpponent
