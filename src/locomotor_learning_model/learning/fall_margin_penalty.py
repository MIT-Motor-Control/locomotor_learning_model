"""Fall-risk penalty terms for the reward.

This module supplies the optional explicit fall-risk term of the objective:
a penalty added to the per-stride reward the RL learner descends. A zero
weight (``fallPenaltyWeight == 0``) reproduces the baseline model exactly.

The **primary** term is a *local center-of-mass (COM) margin* penalty
(``com_margin``). During each simulated step the vertical COM height is
``leglength * cos(theta)`` (nondimensional units, ``leglength == 1`` in this
model). The penalty measures how far the *lowest* COM height in each step
falls below a documented "safe" height, normalized to the range between that
safe height and a ground reference, so the value is dimensionless and bounded
to ``[0, 1]``:

    margin_i = clip((h_safe - min_i(h)) / (h_safe - h_ground), 0, 1)
    penalty  = mean_i( margin_i ** power )

Rationale / consistency with the paper: the manuscript detects a fall when the
COM height drops below a near-ground threshold. This penalty uses the *same*
COM-height quantity, but as a graded, bounded margin so it can enter the
reward the learner descends (a per-stride signal) rather than a terminal
event. It is deterministic (no RNG), so logging it never perturbs the noise
stream and ``fallPenaltyWeight == 0`` reproduces the baseline model exactly.

A ``capture_point`` alternative (bounded XCoM excess) is kept for robustness
checks. A global *probability of falling* term is not a per-stride quantity —
it requires uncapped rollouts and a fall-termination event (a physics
change) — and is therefore not implemented here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_step_min_com_heights(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> np.ndarray:
    """Minimum COM height reached in each simulated step (``leglength*cos theta``)."""
    leg_length = float(param_fixed["leglength"])
    min_heights: list[float] = []
    for step_state in state_store:
        if step_state is None or len(step_state) == 0:
            continue
        theta = np.asarray(step_state)[:, 0]
        min_heights.append(float(np.min(leg_length * np.cos(theta))))
    return np.asarray(min_heights, dtype=float)


def compute_trial_min_com_height(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Lowest COM height reached across the simulated trial/iteration."""
    min_heights = compute_step_min_com_heights(state_store, param_fixed)
    if min_heights.size == 0:
        return float("nan")
    return float(np.min(min_heights))


def compute_step_max_abs_capture_points(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> np.ndarray:
    """Each step's maximum absolute extrapolated-COM (capture point / XCoM)."""
    leg_length = float(param_fixed["leglength"])
    omega = np.sqrt(float(param_fixed["gravg"]) / leg_length)
    angle_slope = float(param_fixed.get("angleSlope", 0.0))
    max_capture_points: list[float] = []
    for step_state in state_store:
        if step_state is None or len(step_state) == 0:
            continue
        step_state = np.asarray(step_state)
        theta = step_state[:, 0]
        theta_dot = step_state[:, 1]
        com_position = leg_length * np.sin(theta + angle_slope)
        com_velocity = leg_length * np.cos(theta + angle_slope) * theta_dot
        capture_point = com_position + com_velocity / omega
        max_capture_points.append(float(np.max(np.abs(capture_point))))
    return np.asarray(max_capture_points, dtype=float)


def compute_com_margin_penalty(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Primary local penalty: bounded per-step COM-height margin (see module docstring)."""
    power = float(param_fixed.get("fallPenaltyPower", 2.0))
    eps = float(param_fixed.get("fallPenaltyEpsilon", 1e-6))
    safe_height = float(param_fixed.get("fallSafeComHeight", 0.95))
    # Fallback matches the pre-registered loaded constant (load_biped_model_parameters)
    # so the function's isolated default is self-consistent with every actual run.
    ground_height = float(param_fixed.get("fallGroundComHeight", 0.90))
    span = max(safe_height - ground_height, eps)

    min_heights = compute_step_min_com_heights(state_store, param_fixed)
    if min_heights.size == 0:
        return 0.0
    margins = np.clip((safe_height - min_heights) / span, 0.0, 1.0)
    return float(np.mean(margins**power))


def compute_capture_point_penalty(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Robustness alternative: bounded XCoM excess beyond a stability limit."""
    power = float(param_fixed.get("fallPenaltyPower", 2.0))
    eps = float(param_fixed.get("fallPenaltyEpsilon", 1e-6))
    capture_points = compute_step_max_abs_capture_points(state_store, param_fixed)
    if capture_points.size == 0:
        return 0.0
    limit = float(param_fixed.get("fallCapturePointLimit", 1.0))
    scale = float(param_fixed.get("fallCapturePointScale", 1.0))
    deficits = np.maximum(0.0, capture_points - limit)
    bounded = np.clip(deficits / max(scale, eps), 0.0, 1.0)
    return float(np.mean(bounded**power))


def compute_fall_margin_penalty(
    state_store: Iterable[np.ndarray],
    param_fixed: dict,
) -> float:
    """Dispatch on ``fallPenaltyMode``.

    - ``com_margin``      : primary bounded COM-height margin (default).
    - ``capture_point``   : robustness XCoM-excess alternative.
    """
    mode = str(param_fixed.get("fallPenaltyMode", "com_margin"))
    if mode == "com_margin":
        return compute_com_margin_penalty(state_store, param_fixed)
    if mode == "capture_point":
        return compute_capture_point_penalty(state_store, param_fixed)
    raise ValueError(f"Unknown fallPenaltyMode: {mode!r} (use 'com_margin' or 'capture_point')")
