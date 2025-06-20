"""Simple plotting helper for post-processing results.

This module provides basic plotting functionality to visualize the results
of the learning simulation, inspired by the MATLAB `postProcessHelper_JustThePlots.m` function.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def post_process_helper_plots(
    state_var0: np.ndarray,
    t_store: List,
    state_store: List,
    emet_total_store: np.ndarray,
    emet_per_time_store: np.ndarray,
    t_step_store: np.ndarray,
    param_controller,
    param_fixed,
    do_animate: bool = False,
    ework_pushoff_store: np.ndarray = None,
    ework_heelstrike_store: np.ndarray = None,
    edot_store_iteration_average: np.ndarray = None,
    t_total_iteration_store: np.ndarray = None,
) -> None:
    """Create basic plots of the simulation results.
    
    Parameters
    ----------
    state_var0 : np.ndarray
        Final state
    t_store : List
        Time trajectories for each step
    state_store : List  
        State trajectories for each step
    emet_total_store : np.ndarray
        Total metabolic energy per step
    emet_per_time_store : np.ndarray
        Metabolic energy rate per step
    t_step_store : np.ndarray
        Step duration for each step
    param_controller : dict
        Controller parameters
    param_fixed : dict
        Fixed parameters
    do_animate : bool, optional
        Whether to create animations (not implemented yet)
    ework_pushoff_store : np.ndarray, optional
        Pushoff work per step
    ework_heelstrike_store : np.ndarray, optional
        Heelstrike work per step
    edot_store_iteration_average : np.ndarray, optional
        Average energy consumption per iteration
    t_total_iteration_store : np.ndarray, optional
        Total time per iteration
    """
    
    num_steps = len(emet_total_store)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Learning Simulation Results', fontsize=16)
    
    # Plot 1: Energy consumption over steps
    axes[0, 0].plot(range(1, num_steps + 1), emet_total_store, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Step Number')
    axes[0, 0].set_ylabel('Total Metabolic Energy')
    axes[0, 0].set_title('Energy Consumption per Step')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Step duration over steps
    axes[0, 1].plot(range(1, num_steps + 1), t_step_store, 'g-', linewidth=1.5)
    axes[0, 1].set_xlabel('Step Number')
    axes[0, 1].set_ylabel('Step Duration')
    axes[0, 1].set_title('Step Duration over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Energy rate over steps  
    axes[1, 0].plot(range(1, num_steps + 1), emet_per_time_store, 'r-', linewidth=1.5)
    axes[1, 0].set_xlabel('Step Number')
    axes[1, 0].set_ylabel('Metabolic Energy Rate')
    axes[1, 0].set_title('Energy Rate per Step')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning progress (if available)
    if edot_store_iteration_average is not None:
        axes[1, 1].plot(range(1, len(edot_store_iteration_average) + 1), 
                       edot_store_iteration_average, 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Learning Iteration')
        axes[1, 1].set_ylabel('Average Energy Consumption')
        axes[1, 1].set_title('Learning Progress')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Plot body angle trajectory for last few steps
        if len(state_store) > 0:
            for i in range(max(0, len(state_store) - 5), len(state_store)):
                if state_store[i].size > 0:
                    theta_list = state_store[i][:, 0]
                    t_list = t_store[i]
                    color = 'r' if i % 2 == 1 else 'b'  # Red for odd steps, blue for even
                    axes[1, 1].plot(t_list, theta_list, color=color, alpha=0.7, linewidth=1)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Body Angle (rad)')
            axes[1, 1].set_title('Body Angle Trajectories (Last 5 Steps)')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Additional energy work plot if available
    if ework_pushoff_store is not None and ework_heelstrike_store is not None:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(range(1, num_steps + 1), ework_pushoff_store, 'b-', linewidth=1.5, label='Pushoff Work')
        plt.plot(range(1, num_steps + 1), ework_heelstrike_store, 'r-', linewidth=1.5, label='Heelstrike Work')
        plt.xlabel('Step Number')
        plt.ylabel('Work')
        plt.title('Mechanical Work per Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        total_work = ework_pushoff_store + ework_heelstrike_store
        plt.plot(range(1, num_steps + 1), total_work, 'k-', linewidth=2, label='Total Work')
        plt.xlabel('Step Number')
        plt.ylabel('Total Work')
        plt.title('Total Mechanical Work per Step')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    # Summary statistics
    print("\\n=== Simulation Summary ===")
    print(f"Total steps: {num_steps}")
    print(f"Total simulation time: {sum(t_step_store):.3f}")
    print(f"Average step duration: {np.mean(t_step_store):.4f}")
    print(f"Average energy per step: {np.mean(emet_total_store):.6f}")
    print(f"Average energy rate: {np.mean(emet_per_time_store):.6f}")
    
    if edot_store_iteration_average is not None:
        print(f"Initial average energy: {edot_store_iteration_average[0]:.6f}")
        print(f"Final average energy: {edot_store_iteration_average[-1]:.6f}")
        improvement = (edot_store_iteration_average[0] - edot_store_iteration_average[-1]) / edot_store_iteration_average[0] * 100
        print(f"Energy improvement: {improvement:.2f}%")
    
    plt.show()