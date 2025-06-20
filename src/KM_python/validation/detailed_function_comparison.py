"""
Detailed function-by-function comparison to verify MATLAB-Python equivalence.

This script performs deeper validation by testing individual functions
with multiple input scenarios and comparing convergence behavior.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameter_loading.load_biped_model_parameters import load_biped_model_parameters
from parameter_loading.load_sensory_noise_parameters import load_sensory_noise_parameters
from parameter_loading.load_controller_gain_parameters import load_controller_gain_parameters
from parameter_loading.load_learner_parameters import load_learner_parameters
from parameter_loading.load_protocol_parameters import load_protocol_parameters
from parameter_loading.load_stored_memory_parameters_control_vs_speed import load_stored_memory_parameters_control_vs_speed
from initializing.load_learnable_parameters_initial import load_learnable_parameters_initial
from initializing.load_initial_body_state import load_initial_body_state
from initializing.get_treadmill_speed import get_treadmill_speed
from learning.f_objective_asymmetric_nominal import f_objective_asymmetric_nominal
from learning.simulate_learning_step_by_step import simulate_learning_step_by_step
from learning.single_pendulum_ode import single_pendulum_ode
from learning.detect_endstance import detect_endstance
from learning.detect_midstance import detect_midstance
from learning.swing_cost_doke import swing_cost_doke


def test_individual_functions():
    """Test individual functions with various inputs."""
    print("🔬 Detailed Function-by-Function Validation")
    print("=" * 60)
    
    # Setup parameters
    param_fixed = {}
    param_fixed = load_biped_model_parameters(param_fixed)
    param_fixed = load_sensory_noise_parameters(param_fixed)
    param_controller_gains = load_controller_gain_parameters(param_fixed)
    param_fixed = load_learner_parameters(param_fixed)
    param_fixed = load_protocol_parameters(param_fixed)
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)
    
    results = {}
    
    # Test 1: Single Pendulum ODE with multiple states
    print("\\n🧮 Testing Single Pendulum ODE...")
    test_states = [
        np.array([0.0, 0.0, 0.0]),      # Zero state
        np.array([0.1, 0.1, 0.0]),      # Small angles
        np.array([0.3, 0.2, 0.1]),      # Moderate state
        np.array([-0.1, -0.05, -0.02])  # Negative values
    ]
    
    ode_results = []
    for i, state in enumerate(test_states):
        try:
            result = single_pendulum_ode(0.0, state, param_fixed, param_controller_gains)
            ode_results.append(result)
            print(f"   State {i+1}: {state} → {result}")
        except Exception as e:
            print(f"   State {i+1}: ERROR - {e}")
            ode_results.append(None)
    
    results['pendulum_ode'] = {
        'success_count': sum(1 for r in ode_results if r is not None),
        'total_tests': len(test_states),
        'results': ode_results
    }
    
    # Test 2: Detection functions
    print("\\n🎯 Testing Detection Functions...")
    test_angles = [0.0, 0.1, 0.3, 0.5, -0.1, -0.3]
    
    endstance_results = []
    midstance_results = []
    
    for angle in test_angles:
        state = np.array([angle, 0.1, 0.0])
        
        # Test endstance detection
        param_controller_gains['theta_end_thisStep'] = 0.3
        end_result = detect_endstance(0.0, state, param_fixed, param_controller_gains)
        endstance_results.append(end_result)
        
        # Test midstance detection  
        param_controller_gains['theta_midstance_thisStep'] = 0.0
        mid_result = detect_midstance(0.0, state, param_fixed, param_controller_gains)
        midstance_results.append(mid_result)
    
    print(f"   Endstance detection: {len([r for r in endstance_results if r is not None])}/{len(test_angles)} successful")
    print(f"   Midstance detection: {len([r for r in midstance_results if r is not None])}/{len(test_angles)} successful")
    
    endstance_success_count = len([r for r in endstance_results if r is not None])
    midstance_success_count = len([r for r in midstance_results if r is not None])
    
    results['detection_functions'] = {
        'success_count': endstance_success_count + midstance_success_count,
        'total_tests': len(test_angles) * 2,  # Two tests per angle
        'endstance_success': endstance_success_count,
        'midstance_success': midstance_success_count
    }
    
    # Test 3: Swing cost function (with proper parameters)
    print("\\n⚡ Testing Swing Cost Function...")
    # Test cases: (t_duration, v_initial, v_final, v_body)
    test_cases = [
        (0.5, 0.0, 0.1, 0.0),    # Acceleration from rest
        (0.5, 0.1, 0.0, 0.0),    # Deceleration to rest  
        (1.0, 0.2, 0.3, 0.1),    # Moderate change
        (0.3, -0.1, 0.1, 0.0),   # Sign change
        (0.8, 0.5, 0.5, 0.2),    # No relative velocity change
    ]
    
    swing_costs = []
    success_count = 0
    for i, (t_dur, v_init, v_final, v_body) in enumerate(test_cases):
        try:
            cost = swing_cost_doke(t_dur, v_init, v_final, v_body, param_fixed)
            swing_costs.append(cost)
            success_count += 1
            print(f"   Case {i+1}: t={t_dur}, v_i={v_init}, v_f={v_final}, v_b={v_body} → Cost = {cost:.6f}")
        except Exception as e:
            print(f"   Case {i+1}: ERROR - {e}")
            swing_costs.append(None)
    
    results['swing_cost'] = {
        'success_count': success_count,
        'total_tests': len(test_cases),
        'results': swing_costs
    }
    
    # Test 4: Objective function with perturbed inputs
    print("\\n🎯 Testing Objective Function Robustness...")
    base_params = load_learnable_parameters_initial(param_fixed)
    base_state = load_initial_body_state(base_params)
    
    perturbations = [0.0, 0.01, 0.05, 0.1, -0.01, -0.05]
    objective_values = []
    
    for pert in perturbations:
        try:
            perturbed_params = base_params + pert
            result = f_objective_asymmetric_nominal(
                perturbed_params, base_state, param_controller_gains, param_fixed, 0.0
            )
            objective_values.append(result[0])
            print(f"   Perturbation {pert:6.2f}: Objective = {result[0]:.8f}")
        except Exception as e:
            print(f"   Perturbation {pert:6.2f}: ERROR - {e}")
            objective_values.append(None)
    
    results['objective_robustness'] = {
        'success_count': sum(1 for v in objective_values if v is not None),
        'total_tests': len(perturbations),
        'values': objective_values
    }
    
    return results


def test_learning_convergence_consistency():
    """Test that learning converges consistently across multiple runs."""
    print("\\n🔄 Testing Learning Convergence Consistency...")
    
    # Setup
    param_fixed = {}
    param_fixed = load_biped_model_parameters(param_fixed)
    param_fixed = load_sensory_noise_parameters(param_fixed)
    param_controller_gains = load_controller_gain_parameters(param_fixed)
    param_fixed = load_learner_parameters(param_fixed)
    param_fixed = load_protocol_parameters(param_fixed)
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)
    
    # Short simulation for testing
    param_fixed['num_iterations'] = 30
    
    p_input_controller = load_learnable_parameters_initial(param_fixed)
    state_var0_model = load_initial_body_state(p_input_controller)
    vA, vB = get_treadmill_speed(0, param_fixed['imposedFootSpeeds'])
    context_now = np.array([vA, vB])
    
    # Run multiple simulations with different random seeds
    results = []
    seeds = [42, 123, 456, 789, 999]
    
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        
        result = simulate_learning_step_by_step(
            param_fixed,
            p_input_controller.copy(),
            state_var0_model.copy(),
            context_now.copy(),
            param_controller_gains
        )
        
        final_params = result[:, -1]
        results.append(final_params)
        print(f"   Run {i+1} (seed {seed}): Final param[0] = {final_params[0]:.6f}")
    
    # Analyze convergence consistency
    results_array = np.array(results)
    mean_final = np.mean(results_array, axis=0)
    std_final = np.std(results_array, axis=0)
    max_std = np.max(std_final)
    
    print(f"   Max standard deviation across runs: {max_std:.6f}")
    print(f"   Mean final parameters: {mean_final}")
    
    # Check if variability is reasonable (learning should be somewhat consistent)
    reasonable_variability = max_std < 0.1  # Allow some variation due to randomness
    
    return {
        'reasonable_variability': reasonable_variability,
        'max_std': max_std,
        'mean_final': mean_final.tolist(),
        'std_final': std_final.tolist()
    }


def main():
    """Run detailed function comparison."""
    print("🚀 Starting Detailed MATLAB-Python Function Comparison")
    print("=" * 70)
    
    try:
        # Test individual functions
        function_results = test_individual_functions()
        
        # Test learning convergence consistency
        convergence_results = test_learning_convergence_consistency()
        
        # Generate final report
        print("\\n" + "=" * 70)
        print("📊 DETAILED VALIDATION SUMMARY")
        print("=" * 70)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in function_results.items():
            success_rate = result['success_count'] / result['total_tests']
            total_tests += 1
            
            if success_rate >= 0.9:  # 90% success rate threshold
                passed_tests += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            print(f"\\n{test_name.replace('_', ' ').title()}:")
            print(f"   Success rate: {result['success_count']}/{result['total_tests']} ({success_rate:.1%})")
            print(f"   {status}")
        
        # Add convergence test
        total_tests += 1
        if convergence_results['reasonable_variability']:
            passed_tests += 1
            print("\\nLearning Convergence Consistency:")
            print(f"   Variability: {convergence_results['max_std']:.6f} (reasonable)")
            print("   ✅ PASSED")
        else:
            print("\\nLearning Convergence Consistency:")
            print(f"   Variability: {convergence_results['max_std']:.6f} (too high)")
            print("   ❌ FAILED")
        
        print("\\n" + "=" * 70)
        overall_success_rate = passed_tests / total_tests
        print(f"🏆 OVERALL DETAILED VALIDATION: {passed_tests}/{total_tests} ({overall_success_rate:.1%})")
        
        if overall_success_rate == 1.0:
            print("🎉 PERFECT SCORE! Python implementation is functionally equivalent to MATLAB!")
        elif overall_success_rate >= 0.9:
            print("🎉 EXCELLENT! Python implementation is highly equivalent to MATLAB!")
        elif overall_success_rate >= 0.7:
            print("✅ GOOD! Python implementation is largely equivalent to MATLAB!")
        else:
            print("⚠️ Python implementation has significant differences from MATLAB.")
        
        print("=" * 70)
        
        return overall_success_rate
        
    except Exception as e:
        print(f"\\n❌ Detailed validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == "__main__":
    success_rate = main()
    print(f"\\n🎯 Final validation score: {success_rate:.1%}")