from __future__ import annotations

import numpy as np

from locomotor_learning_model.learning.simulate_one_step_midstance_to_midstance_with_energy import (
    simulate_one_step_midstance_to_midstance_with_energy,
)


def simulate_many_steps_asymmetric_control(
    state_var0: np.ndarray,
    param_controller,
    param_fixed,
    t_start: float,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Simulate asymmetric walking and return trajectories and step-wise energies.

    The initial state contains stance angle, angular velocity, foot position,
    accumulated foot position, and swing-foot velocity at midstance.
    """

    num_steps = int(param_fixed['num_steps'])

    t_store: list[np.ndarray] = [np.empty(0) for _ in range(num_steps + 1)]
    state_store: list[np.ndarray] = [np.empty((0, 3)) for _ in range(num_steps + 1)]

    emet_total_store = np.zeros(num_steps)
    emet_per_time_store = np.zeros(num_steps)
    t_step_store = np.zeros(num_steps)

    ework_pushoff_store = np.zeros(num_steps)
    ework_heelstrike_store = np.zeros(num_steps)

    for i_step in range(1, num_steps + 1):
        if i_step % 2 == 0:
            param_controller['theta_end_nominal'] = param_controller['Even']['theta_end_nominal']
            param_controller['ydot_at_midstance_nominal_beltframe'] = (
                param_controller['Even']['ydot_at_midstance_nominal_beltframe']
            )
            param_controller['pushoff_impulse_magnitude_nominal'] = (
                param_controller['Even']['pushoff_impulse_magnitude_nominal']
            )
            param_controller['y_at_midstance_nominal_slopeframe'] = (
                param_controller['Even']['y_at_midstance_nominal_slopeframe']
            )
            param_controller['sumy_at_midstance_nominal_slopeframe'] = (
                param_controller['Even']['sumy_at_midstance_nominal_slopeframe']
            )
        else:
            param_controller['theta_end_nominal'] = param_controller['Odd']['theta_end_nominal']
            param_controller['ydot_at_midstance_nominal_beltframe'] = (
                param_controller['Odd']['ydot_at_midstance_nominal_beltframe']
            )
            param_controller['pushoff_impulse_magnitude_nominal'] = (
                param_controller['Odd']['pushoff_impulse_magnitude_nominal']
            )
            # Odd-leg position targets define the reference frame.
            param_controller['y_at_midstance_nominal_slopeframe'] = 0.0
            param_controller['sumy_at_midstance_nominal_slopeframe'] = 0.0

        (
            state_var0,
            tlist_till_endstance,
            statevarlist_till_endstance,
            tlist_till_midstance,
            statevarlist_midstance,
            emet_total_now,
            emet_per_time,
            t_total,
            ework_pushoff,
            ework_heelstrike,
        ) = simulate_one_step_midstance_to_midstance_with_energy(
            state_var0,
            i_step,
            t_start,
            param_controller,
            param_fixed,
        )

        # Each midstance-to-midstance interval spans two stance trajectories.
        t_store[i_step - 1] = np.concatenate(
            [t_store[i_step - 1], np.asarray(tlist_till_endstance).flatten()]
        )
        state_store[i_step - 1] = np.vstack(
            [state_store[i_step - 1], np.asarray(statevarlist_till_endstance)]
        )

        if i_step < num_steps:
            t_store[i_step] = np.concatenate(
                [t_store[i_step], np.asarray(tlist_till_midstance[:-1]).flatten()]
            )
            state_store[i_step] = np.vstack(
                [state_store[i_step], np.asarray(statevarlist_midstance[:-1])]
            )
        else:
            t_store[i_step] = np.concatenate(
                [t_store[i_step], np.asarray(tlist_till_midstance).flatten()]
            )
            state_store[i_step] = np.vstack(
                [state_store[i_step], np.asarray(statevarlist_midstance)]
            )

        t_start = float(np.asarray(tlist_till_midstance).flatten()[-1])

        emet_total_store[i_step - 1] = float(emet_total_now)
        emet_per_time_store[i_step - 1] = float(emet_per_time)
        t_step_store[i_step - 1] = float(t_total)

        ework_pushoff_store[i_step - 1] = float(ework_pushoff)
        ework_heelstrike_store[i_step - 1] = float(ework_heelstrike)

    return (
        state_var0,
        t_store,
        state_store,
        emet_total_store,
        emet_per_time_store,
        t_step_store,
        ework_pushoff_store,
        ework_heelstrike_store,
    )
