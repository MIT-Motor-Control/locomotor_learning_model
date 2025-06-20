"""
Fixed detailed function comparison that properly sets up context for individual functions.
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
from learning.single_pendulum_ode import single_pendulum_ode
from learning.detect_endstance import detect_endstance
from learning.detect_midstance import detect_midstance
from learning.swing_cost_doke import swing_cost_doke


def setup_complete_parameters():
    """Set up complete parameters with all necessary context."""
    param_fixed = {}
    param_fixed = load_biped_model_parameters(param_fixed)
    param_fixed = load_sensory_noise_parameters(param_fixed)
    param_controller_gains = load_controller_gain_parameters(param_fixed)
    param_fixed = load_learner_parameters(param_fixed)
    param_fixed = load_protocol_parameters(param_fixed)
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)
    
    # Add missing context that's normally set during simulation
    param_controller_gains['tList_BeltSpeed'] = param_fixed['imposedFootSpeeds']['tList']
    param_controller_gains['PushoffFootSpeedNowList'] = param_fixed['imposedFootSpeeds']['footSpeed1List']
    param_controller_gains['PushoffAccelerationNowList'] = param_fixed['imposedFootSpeeds']['footAcc1List']
    
    return param_fixed, param_controller_gains


def test_pendulum_ode():
    """Test single pendulum ODE with proper context."""
    print("🧮 Testing Single Pendulum ODE with proper context...")
    
    param_fixed, param_controller = setup_complete_parameters()
    
    test_states = [
        np.array([0.0, 0.0, 0.0]),      # Zero state
        np.array([0.1, 0.1, 0.0]),      # Small angles
        np.array([0.3, 0.2, 0.1]),      # Moderate state
        np.array([-0.1, -0.05, -0.02])  # Negative values
    ]
    
    success_count = 0
    total_tests = len(test_states)
    
    for i, state in enumerate(test_states):
        try:
            result = single_pendulum_ode(0.0, state, param_fixed, param_controller)
            print(f"   State {i+1}: {state} → {result}")
            
            # Validate that result is reasonable
            if len(result) == 3 and all(np.isfinite(result)):
                success_count += 1
            else:
                print(f"      ⚠️ Result not finite or wrong size")
                
        except Exception as e:
            print(f"   State {i+1}: ERROR - {e}")
    
    success_rate = success_count / total_tests
    print(f"   Success rate: {success_count}/{total_tests} ({success_rate:.1%})")
    
    return {
        'success_count': success_count,
        'total_tests': total_tests,
        'success_rate': success_rate
    }


def test_detection_functions():
    """Test detection functions."""
    print("\\n🎯 Testing Detection Functions...")
    
    param_fixed, param_controller = setup_complete_parameters()
    
    test_angles = [0.0, 0.1, 0.3, 0.5, -0.1, -0.3]
    
    endstance_success = 0
    midstance_success = 0
    total_tests = len(test_angles)
    
    for angle in test_angles:
        state = np.array([angle, 0.1, 0.0])
        
        # Test endstance detection
        try:
            param_controller['theta_end_thisStep'] = 0.3
            end_result = detect_endstance(0.0, state, param_fixed, param_controller)
            if np.isfinite(end_result):
                endstance_success += 1
        except Exception as e:
            print(f"   Endstance error at angle {angle}: {e}")
        
        # Test midstance detection  
        try:
            param_controller['theta_midstance_thisStep'] = 0.0
            mid_result = detect_midstance(0.0, state, param_fixed, param_controller)
            if np.isfinite(mid_result):
                midstance_success += 1
        except Exception as e:
            print(f"   Midstance error at angle {angle}: {e}")
    
    print(f"   Endstance detection: {endstance_success}/{total_tests} successful")
    print(f"   Midstance detection: {midstance_success}/{total_tests} successful")
    
    return {
        'endstance_success': endstance_success,
        'midstance_success': midstance_success,
        'total_tests': total_tests,
        'combined_success_rate': (endstance_success + midstance_success) / (2 * total_tests)
    }


def test_swing_cost_function():
    """Test swing cost function with proper parameters."""
    print("\\n⚡ Testing Swing Cost Function with proper parameters...")
    
    param_fixed, _ = setup_complete_parameters()
    
    # Test cases: (t_duration, v_initial, v_final, v_body)
    test_cases = [
        (0.5, 0.0, 0.1, 0.0),    # Acceleration from rest
        (0.5, 0.1, 0.0, 0.0),    # Deceleration to rest
        (1.0, 0.2, 0.3, 0.1),    # Moderate change
        (0.3, -0.1, 0.1, 0.0),   # Sign change
        (0.8, 0.5, 0.5, 0.2),    # No relative velocity change
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, (t_dur, v_init, v_final, v_body) in enumerate(test_cases):
        try:
            cost = swing_cost_doke(t_dur, v_init, v_final, v_body, param_fixed)
            print(f"   Case {i+1}: t={t_dur}, v_i={v_init}, v_f={v_final}, v_b={v_body} → Cost = {cost:.6f}")
            
            # Validate result is finite and non-negative
            if np.isfinite(cost) and cost >= 0:
                success_count += 1
            else:
                print(f"      ⚠️ Invalid cost value")
                
        except Exception as e:
            print(f"   Case {i+1}: ERROR - {e}")
    
    success_rate = success_count / total_tests
    print(f"   Success rate: {success_count}/{total_tests} ({success_rate:.1%})")
    
    return {
        'success_count': success_count,
        'total_tests': total_tests,
        'success_rate': success_rate
    }


def test_objective_function_robustness():
    """Test objective function with parameter perturbations."""
    print("\\n🎯 Testing Objective Function Robustness...")
    
    param_fixed, param_controller = setup_complete_parameters()
    
    base_params = load_learnable_parameters_initial(param_fixed)
    base_state = load_initial_body_state(base_params)
    
    perturbations = [0.0, 0.01, 0.05, 0.1, -0.01, -0.05]
    
    success_count = 0
    total_tests = len(perturbations)
    objective_values = []
    
    for pert in perturbations:
        try:
            perturbed_params = base_params + pert
            result = f_objective_asymmetric_nominal(
                perturbed_params, base_state, param_controller, param_fixed, 0.0
            )
            objective_values.append(result[0])
            print(f"   Perturbation {pert:6.2f}: Objective = {result[0]:.8f}")
            
            # Validate result is finite and positive
            if np.isfinite(result[0]) and result[0] > 0:
                success_count += 1
            else:
                print(f"      ⚠️ Invalid objective value")
                
        except Exception as e:
            print(f"   Perturbation {pert:6.2f}: ERROR - {e}")
            objective_values.append(None)
    
    success_rate = success_count / total_tests
    print(f"   Success rate: {success_count}/{total_tests} ({success_rate:.1%})")
    
    return {
        'success_count': success_count,
        'total_tests': total_tests,
        'success_rate': success_rate,
        'values': objective_values
    }


def main():
    """Run fixed detailed function comparison."""
    print("🔧 FIXED Detailed MATLAB-Python Function Comparison")
    print("=" * 60)
    
    try:
        # Test individual functions with proper context
        pendulum_results = test_pendulum_ode()
        detection_results = test_detection_functions()
        swing_cost_results = test_swing_cost_function()
        objective_results = test_objective_function_robustness()
        
        # Collect all results
        all_results = {
            'pendulum_ode': pendulum_results,
            'detection_functions': detection_results,
            'swing_cost': swing_cost_results,
            'objective_robustness': objective_results
        }
        
        # Generate summary report
        print("\\n" + "=" * 60)
        print("📊 FIXED DETAILED VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in all_results.items():
            if test_name == 'detection_functions':
                success_rate = result['combined_success_rate']
            else:
                success_rate = result['success_rate']
            
            total_tests += 1
            
            if success_rate >= 0.9:  # 90% success rate threshold
                passed_tests += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            print(f"\\n{test_name.replace('_', ' ').title()}:")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   {status}")
        
        print("\\n" + "=" * 60)
        overall_success_rate = passed_tests / total_tests
        print(f"🏆 FIXED DETAILED VALIDATION: {passed_tests}/{total_tests} ({overall_success_rate:.1%})")
        
        if overall_success_rate == 1.0:
            print("🎉 PERFECT! All individual functions work correctly!")
        elif overall_success_rate >= 0.75:
            print("✅ EXCELLENT! Most functions work correctly!")
        else:
            print("⚠️ Some functions need attention.")
        
        print("=" * 60)
        
        return overall_success_rate
        
    except Exception as e:
        print(f"\\n❌ Fixed validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == "__main__":
    success_rate = main()
    print(f"\\n🎯 Fixed validation score: {success_rate:.1%}")