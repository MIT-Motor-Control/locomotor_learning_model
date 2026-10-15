"""Regression tests for the MATLAB-reference comparison gate."""

import importlib.util
from pathlib import Path

import numpy as np


def test_parity_rejects_nonfinite_difference(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/validate_matlab_python_parity.py"
    spec = importlib.util.spec_from_file_location("parity_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = {
        "objectiveValue": np.array([np.nan]),
        "pInput": np.zeros(10),
        "stateVar0": np.zeros(5),
        "contextNow": np.zeros(2),
        "controllerStore8D": np.zeros((8, 2)),
    }
    monkeypatch.setattr(module, "loadmat", lambda path: reference)
    monkeypatch.setattr(module, "load_learnable_parameters_initial", lambda fixed: np.zeros(10))
    monkeypatch.setattr(module, "load_initial_body_state", lambda controller: np.zeros(5))
    monkeypatch.setattr(module, "get_treadmill_speed", lambda *args: (0.0, 0.0))
    monkeypatch.setattr(module, "f_objective_asymmetric_nominal", lambda *args: (0.0,))
    monkeypatch.setattr(module, "simulate_learning_step_by_step", lambda *args: np.zeros((8, 2)))
    assert module.main(["unused-reference.mat"]) == 1
