import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from core.match_simulator import (
    AI_VS_AI,
    AI_VS_SCRIPTED,
    HUMAN_VS_AI,
    MatchSimulator,
    mirror_action_for_right_player,
    mirror_observation_for_right_player,
    mouse_position_to_world,
)
from ui.playground_dashboard import PlaygroundDashboard
from ui.main_window import OrionMainWindow


def test_mouse_position_maps_into_legal_left_side_mallet_area():
    left_edge = mouse_position_to_world(10, 70, 10, 20, 200, 100)
    right_edge = mouse_position_to_world(210, 20, 10, 20, 200, 100)

    assert left_edge[0] == -0.925
    assert right_edge[0] == -0.15
    assert np.isclose(left_edge[1], 0.0)
    assert np.isclose(right_edge[1], 0.442)


def test_right_player_observation_and_action_mirroring_are_consistent():
    observation = np.asarray([0.3, -0.2, 0.4, 0.5, 0.8, 0.1, -0.7, -0.1], dtype=np.float32)
    action = np.asarray([0.6, -0.4], dtype=np.float32)

    mirrored = mirror_observation_for_right_player(observation)
    restored = mirror_observation_for_right_player(mirrored)

    np.testing.assert_array_equal(restored, observation)
    np.testing.assert_array_equal(
        mirror_action_for_right_player(mirror_action_for_right_player(action)),
        action,
    )


def test_ai_vs_ai_steps_both_policies_and_keeps_mallets_on_their_sides(monkeypatch):
    class DummyModel:
        def __init__(self, action):
            self.action = np.asarray(action, dtype=np.float32)
            self.observations = []

        def predict(self, observation, deterministic=True):
            self.observations.append(np.asarray(observation).copy())
            return self.action, None

    left_model = DummyModel([0.4, -0.2])
    right_model = DummyModel([0.3, 0.1])
    models = iter((left_model, right_model))
    monkeypatch.setattr("core.match_simulator.load_policy", lambda _path: next(models))

    simulator = MatchSimulator(AI_VS_AI, model_a_path="left.zip", model_b_path="right.zip", seed=5)
    simulator.step()

    assert len(left_model.observations) == 1
    assert len(right_model.observations) == 1
    np.testing.assert_array_equal(
        right_model.observations[0],
        mirror_observation_for_right_player(left_model.observations[0]),
    )
    assert simulator.env.robot_position_m[0] <= -0.15
    assert simulator.env._opponent_position[0] >= simulator.env.MALLET_RADIUS_M
    simulator.close()


def test_human_vs_ai_clamps_mouse_target_to_left_half(monkeypatch):
    class DummyModel:
        def predict(self, observation, deterministic=True):
            return np.zeros(2, dtype=np.float32), None

    monkeypatch.setattr("core.match_simulator.load_policy", lambda _path: DummyModel())
    simulator = MatchSimulator(HUMAN_VS_AI, model_b_path="right.zip", seed=7)
    simulator.set_mouse_target(0.8, 0.8)

    assert simulator.mouse_target_m[0] == -0.15
    assert simulator.mouse_target_m[1] < simulator.env.FIELD_WIDTH_M / 2.0
    simulator.step()
    assert simulator.env.robot_position_m[0] <= -0.15
    simulator.close()


def test_ai_vs_scripted_uses_complex_opponent(monkeypatch):
    class DummyModel:
        def predict(self, observation, deterministic=True):
            return np.zeros(2, dtype=np.float32), None

    monkeypatch.setattr("core.match_simulator.load_policy", lambda _path: DummyModel())
    simulator = MatchSimulator(AI_VS_SCRIPTED, model_a_path="left.zip", seed=9)
    assert simulator.env.opponent_type == "complex"
    simulator.close()


def test_playground_mode_selection_exposes_only_required_model_controls(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    model_file = tmp_path / "demo.zip"
    model_file.write_bytes(b"test")
    monkeypatch.setattr("ui.playground_dashboard.MODEL_ROOT", tmp_path)
    dashboard = PlaygroundDashboard()

    assert dashboard.mode_selector.count() == 3
    assert dashboard.model_a_selector.count() == 1
    assert dashboard.model_a_selector.currentData() == str(model_file)

    dashboard.mode_selector.setCurrentIndex(0)
    assert not dashboard.model_a_selector.isEnabled()
    assert dashboard.model_b_selector.isEnabled()
    dashboard.mode_selector.setCurrentIndex(1)
    assert dashboard.model_a_selector.isEnabled()
    assert dashboard.model_b_selector.isEnabled()
    dashboard.mode_selector.setCurrentIndex(2)
    assert dashboard.model_a_selector.isEnabled()
    assert not dashboard.model_b_selector.isEnabled()
    assert dashboard.SIM_DT == 1.0 / 120.0
    assert dashboard._simulation_timer.interval() == 4

    dashboard.close()
    app.processEvents()


def test_playground_locks_setup_during_match_and_cleans_up_on_back(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class FakeSimulator:
        def __init__(self, mode, **_kwargs):
            self.mode = mode
            self.observation = np.zeros(8, dtype=np.float32)
            self.elapsed_seconds = 0.0
            self.puck_speed_mps = 0.0
            self.closed = False

        def close(self):
            self.closed = True

    back_calls = []
    monkeypatch.setattr("ui.playground_dashboard.MatchSimulator", FakeSimulator)
    dashboard = PlaygroundDashboard(on_back_callback=lambda: back_calls.append(True))

    dashboard.start_match()

    active_simulator = dashboard.simulator
    assert not dashboard.mode_selector.isEnabled()
    assert not dashboard.model_b_selector.isEnabled()
    assert dashboard._simulation_timer.isActive()

    dashboard._back_to_menu()

    assert active_simulator.closed
    assert dashboard.simulator is None
    assert dashboard.mode_selector.isEnabled()
    assert back_calls == [True]
    dashboard.close()
    app.processEvents()


def test_main_window_opens_playground_module():
    app = QApplication.instance() or QApplication([])
    window = OrionMainWindow()

    window.launch_module("playground")

    assert isinstance(window.playground_dashboard_widget, PlaygroundDashboard)
    assert window.stack.currentWidget() is window.playground_dashboard_widget
    window.close()
    app.processEvents()
